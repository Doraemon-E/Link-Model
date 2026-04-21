import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import torch
from torch.export import Dim
from transformers import AutoModelForCausalLM

from helper.coreml_bundle_helpers import (
    build_named_coreml_paths,
    copy_runtime_files,
    package_coreml_bundle,
    write_translation_manifest,
)
from helper.coreml_quantization_helpers import (
    assert_coreml_weight_quantization_ops,
    assert_torch_quantization_metadata,
)
from helper.stateful_hunyuan_for_coreml import StatefulHunYuanForCoreML
from helper.stateless_hunyuan_for_coreml import StatelessHunYuanForCoreML


DEFAULT_MODEL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")
DEFAULT_COREML_OUTPUT_ROOT = Path("models/translation/converted/coreml-int8")
DEFAULT_COREML_ARTIFACT_STEM = "hy-mt1.5-1.8b-coreml-int8"
DEFAULT_COREML_PACKAGED_ROOT = Path("models/translation/packaged")

DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_OPTIMIZED_CONTEXT_LENGTH = 128
DEFAULT_COREML_Q_BITS = 8
DEFAULT_COREML_Q_GROUP_SIZE = 64
DEFAULT_COREML_Q_MODE = "affine"
DEFAULT_COREML_DECODE_ONLY = True
DEFAULT_USE_STATE_CACHE = True
DEFAULT_CACHE_TIER_CONTEXTS = "256,192,128,96,64"
DEFAULT_CACHE_TIER_PREFIX = "cache-c"
DEFAULT_COMPILE_MLMODELC = True
DEFAULT_PACKAGE_INCLUDE_MLMODELC = True

MODEL_PACKAGE_FILE_NAME = "hy_mt_w8_from_torch.mlpackage"
MODEL_COMPILED_DIR_NAME = "hy_mt_w8_from_torch.mlmodelc"
PROFILE_CACHE = "cache"
PROFILE_NOCACHE = "nocache"
PROFILE_CACHE_OPT = "cache-opt"
PROFILE_CACHE_TIERS = "cache-tiers"
PROFILE_ALL = "all"


@dataclass(frozen=True)
class ExportProfile:
    name: str
    use_state_cache: bool
    decode_only: bool
    context_length: int


# ---------- model load / quantization ----------
def _load_base_model(model_dir: Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"

    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.rope_type = "default"

    return model


def _resolve_coreml_quantization_options(
    q_bits: int,
    q_mode: str,
) -> tuple[str, str]:
    if q_bits == 8:
        weight_dtype = "int8"
    elif q_bits == 4:
        weight_dtype = "int4"
    else:
        raise ValueError(f"unsupported q_bits={q_bits}, only 4 or 8 are supported")

    normalized_q_mode = q_mode.lower()
    if normalized_q_mode not in {"affine", "symmetric"}:
        raise ValueError(
            f"unsupported q_mode={q_mode}, expected one of: affine, symmetric"
        )
    return weight_dtype, normalized_q_mode


def _build_linear_ptq_config(
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> cto.torch.quantization.PostTrainingQuantizerConfig:
    if q_group_size <= 0:
        raise ValueError(f"q_group_size must be > 0, got {q_group_size}")

    weight_dtype, quantization_scheme = _resolve_coreml_quantization_options(
        q_bits=q_bits,
        q_mode=q_mode,
    )
    return cto.torch.quantization.PostTrainingQuantizerConfig.from_dict(
        {
            "module_type_configs": {
                torch.nn.Linear: {
                    "weight_dtype": weight_dtype,
                    "granularity": "per_block",
                    "block_size": q_group_size,
                    "quantization_scheme": quantization_scheme,
                },
            },
        }
    )


def _load_quantized_torch_model(
    model_dir: Path,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> torch.nn.Module:
    model = _load_base_model(model_dir)
    config = _build_linear_ptq_config(
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )

    quantizer = cto.torch.quantization.PostTrainingQuantizer(model, config)
    quantized_model = quantizer.compress()
    quantized_model.eval()
    assert_torch_quantization_metadata(quantized_model)
    return quantized_model


# ---------- convert ----------
def _convert_coreml(
    model_dir: Path,
    output_dir: Path,
    context_length: int,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
    decode_only: bool,
    use_state_cache: bool,
) -> tuple[Path, str]:
    quantized_torch_model = _load_quantized_torch_model(
        model_dir=model_dir,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )

    if use_state_cache:
        wrapper = StatefulHunYuanForCoreML(
            model=quantized_torch_model,
            max_cache_len=context_length,
            decode_only=decode_only,
        )
        wrapper.reset_cache()
    else:
        wrapper = StatelessHunYuanForCoreML(model=quantized_torch_model)
    wrapper.eval()

    if decode_only and use_state_cache:
        sample_input_ids = torch.ones((1, 1), dtype=torch.int32)
        exported_program = torch.export.export(
            wrapper,
            args=(sample_input_ids,),
            strict=False,
        )
    else:
        sample_len = min(8, context_length)
        sample_input_ids = torch.ones((1, sample_len), dtype=torch.int32)
        dynamic_shapes = {
            "input_ids": {1: Dim("query_length", min=1, max=context_length)}
        }
        exported_program = torch.export.export(
            wrapper,
            args=(sample_input_ids,),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

    exported_program = exported_program.run_decompositions({})

    output_dir.mkdir(parents=True, exist_ok=True)
    coreml_path = output_dir / MODEL_PACKAGE_FILE_NAME

    default_passes = list(ct.PassPipeline.DEFAULT.passes)
    custom_pass_pipeline = ct.PassPipeline(
        pass_names=default_passes,
        pipeline_name="hy_mt_coreml_export",
    )
    if use_state_cache:
        custom_pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

    convert_kwargs: dict[str, object] = {
        "convert_to": "mlprogram",
        "minimum_deployment_target": ct.target.iOS18,
        "pass_pipeline": custom_pass_pipeline,
        "package_dir": str(coreml_path),
        "skip_model_load": True,
    }
    if use_state_cache:
        convert_kwargs["states"] = _build_coreml_states(wrapper)

    coreml_model = ct.convert(
        exported_program,
        **convert_kwargs,
    )
    assert_coreml_weight_quantization_ops(coreml_model)
    if not coreml_path.exists():
        raise RuntimeError(f"coreml package was not created at expected path: {coreml_path}")

    output_names = [out.name for out in coreml_model.get_spec().description.output]
    if not output_names:
        raise RuntimeError("coreml model has no outputs")
    if "logits" in output_names:
        output_name = "logits"
    elif len(output_names) == 1:
        output_name = output_names[0]
    else:
        raise RuntimeError(f"unexpected multiple outputs: {output_names}")

    return coreml_path, output_name


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _compile_mlpackage_to_mlmodelc(
    *,
    mlpackage_path: Path,
    compiled_model_path: Path,
) -> Path:
    if not mlpackage_path.exists():
        raise RuntimeError(f"mlpackage does not exist: {mlpackage_path}")

    _remove_path(compiled_model_path)
    compiled_path = Path(
        ct.models.utils.compile_model(
            str(mlpackage_path),
            destination_path=str(compiled_model_path),
        )
    )
    if not compiled_path.exists():
        raise RuntimeError(f"compiled mlmodelc was not created: {compiled_path}")
    return compiled_path


# ---------- coreml states ----------
def _make_state(name: str, shape: tuple[int, ...], dtype) -> ct.StateType:
    return ct.StateType(
        wrapped_type=ct.TensorType(
            shape=shape,
            dtype=dtype,
        ),
        name=name,
    )


def _build_coreml_states(
    wrapper,
    cache_dtype=np.float16,
    position_dtype=np.float16,
) -> list[ct.StateType]:
    states: list[ct.StateType] = []

    for layer_idx in range(wrapper.num_layers):
        key_cache = getattr(wrapper, f"key_cache_{layer_idx}")
        states.append(
            _make_state(
                name=f"key_cache_{layer_idx}",
                shape=tuple(key_cache.shape),
                dtype=cache_dtype,
            )
        )

    for layer_idx in range(wrapper.num_layers):
        value_cache = getattr(wrapper, f"value_cache_{layer_idx}")
        states.append(
            _make_state(
                name=f"value_cache_{layer_idx}",
                shape=tuple(value_cache.shape),
                dtype=cache_dtype,
            )
        )

    states.append(
        _make_state(
            name="cache_position",
            shape=tuple(wrapper.cache_position.shape),
            dtype=position_dtype,
        )
    )

    return states


# ---------- cli helpers ----------
def _resolve_profiles(args: argparse.Namespace) -> list[ExportProfile]:
    def _parse_cache_tier_contexts(raw: str) -> list[int]:
        values: list[int] = []
        seen: set[int] = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value <= 0:
                raise ValueError(f"cache tier context must be > 0, got {value}")
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
        if not values:
            raise ValueError("cache tier contexts cannot be empty")
        return values

    profiles = {
        PROFILE_CACHE: ExportProfile(
            name=PROFILE_CACHE,
            use_state_cache=True,
            decode_only=args.decode_only,
            context_length=args.context_length,
        ),
        PROFILE_NOCACHE: ExportProfile(
            name=PROFILE_NOCACHE,
            use_state_cache=False,
            decode_only=False,
            context_length=args.context_length,
        ),
        PROFILE_CACHE_OPT: ExportProfile(
            name=PROFILE_CACHE_OPT,
            use_state_cache=True,
            decode_only=args.decode_only,
            context_length=args.optimized_context_length,
        ),
    }

    if args.profile == PROFILE_CACHE_TIERS:
        contexts = _parse_cache_tier_contexts(args.cache_tier_contexts)
        return [
            ExportProfile(
                name=f"{args.cache_tier_prefix}{context_length}",
                use_state_cache=True,
                decode_only=args.decode_only,
                context_length=context_length,
            )
            for context_length in contexts
        ]

    if args.profile == PROFILE_ALL:
        return [profiles[PROFILE_NOCACHE], profiles[PROFILE_CACHE], profiles[PROFILE_CACHE_OPT]]
    return [profiles[args.profile]]


def _infer_coreml_kind_from_path(path: str | None) -> str | None:
    if not path:
        return None
    if path.endswith(".mlmodelc"):
        return "mlmodelc"
    if path.endswith(".mlpackage"):
        return "mlpackage"
    return None


def _load_existing_manifest_coreml_fields(
    output_dir: Path,
) -> tuple[str | None, str | None, str | None]:
    manifest_path = output_dir / "translation-manifest.json"
    if not manifest_path.is_file():
        return None, None, None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    coreml = payload.get("coreml", {})
    output_name = coreml.get("outputName")
    model_path = coreml.get("path")
    model_kind = coreml.get("kind")
    output_name_value = output_name if isinstance(output_name, str) and output_name else None
    model_path_value = model_path if isinstance(model_path, str) and model_path else None
    model_kind_value = model_kind if isinstance(model_kind, str) and model_kind else None
    return output_name_value, model_path_value, model_kind_value


def _resolve_existing_model_artifact(
    output_dir: Path,
    *,
    manifest_model_path: str | None,
    manifest_model_kind: str | None,
) -> tuple[str, str]:
    if manifest_model_path is not None and not Path(manifest_model_path).is_absolute():
        candidate = output_dir / manifest_model_path
        if candidate.exists():
            inferred_kind = _infer_coreml_kind_from_path(manifest_model_path)
            return manifest_model_path, manifest_model_kind or inferred_kind or "mlpackage"

    default_compiled_path = output_dir / MODEL_COMPILED_DIR_NAME
    if default_compiled_path.exists():
        return MODEL_COMPILED_DIR_NAME, "mlmodelc"

    default_package_path = output_dir / MODEL_PACKAGE_FILE_NAME
    if default_package_path.exists():
        return MODEL_PACKAGE_FILE_NAME, "mlpackage"

    raise RuntimeError(
        "unable to resolve existing CoreML model artifact; "
        f"expected one of {MODEL_COMPILED_DIR_NAME} or {MODEL_PACKAGE_FILE_NAME} in {output_dir}"
    )


def _ensure_compiled_model_artifact(
    output_dir: Path,
    *,
    manifest_model_path: str | None,
) -> tuple[str, str]:
    if manifest_model_path is not None and not Path(manifest_model_path).is_absolute():
        manifest_artifact = output_dir / manifest_model_path
        if manifest_artifact.exists() and manifest_artifact.name.endswith(".mlmodelc"):
            return manifest_model_path, "mlmodelc"
        if manifest_artifact.exists() and manifest_artifact.name.endswith(".mlpackage"):
            sibling_compiled = manifest_artifact.with_suffix(".mlmodelc")
            if sibling_compiled.exists():
                try:
                    relative_compiled = sibling_compiled.relative_to(output_dir).as_posix()
                except ValueError:
                    relative_compiled = sibling_compiled.name
                return relative_compiled, "mlmodelc"

    compiled_model_path = output_dir / MODEL_COMPILED_DIR_NAME
    if compiled_model_path.exists():
        return MODEL_COMPILED_DIR_NAME, "mlmodelc"

    source_mlpackage: Path | None = None
    default_package_path = output_dir / MODEL_PACKAGE_FILE_NAME
    if default_package_path.exists():
        source_mlpackage = default_package_path
    elif manifest_model_path is not None and not Path(manifest_model_path).is_absolute():
        manifest_path = output_dir / manifest_model_path
        if manifest_path.exists() and manifest_path.name.endswith(".mlpackage"):
            source_mlpackage = manifest_path

    if source_mlpackage is None:
        package_candidates = sorted(
            [
                child
                for child in output_dir.iterdir()
                if child.name.endswith(".mlpackage")
            ]
        )
        if len(package_candidates) == 1:
            source_mlpackage = package_candidates[0]

    if source_mlpackage is None:
        raise RuntimeError(
            "cannot compile mlmodelc because no source .mlpackage was found in "
            f"{output_dir}"
        )

    compiled_path = _compile_mlpackage_to_mlmodelc(
        mlpackage_path=source_mlpackage,
        compiled_model_path=compiled_model_path,
    )
    try:
        compiled_relative_path = compiled_path.relative_to(output_dir).as_posix()
    except ValueError:
        compiled_relative_path = compiled_path.name
    return compiled_relative_path, "mlmodelc"


def _resolve_packaging_model_artifacts(
    output_dir: Path,
    *,
    selected_model_path: str,
    package_include_mlmodelc: bool,
) -> tuple[str, tuple[str, ...]]:
    if not package_include_mlmodelc:
        return selected_model_path, ()

    compiled_model_path, _ = _ensure_compiled_model_artifact(
        output_dir,
        manifest_model_path=selected_model_path,
    )
    if compiled_model_path == selected_model_path:
        return selected_model_path, ()
    return selected_model_path, (compiled_model_path,)


def _run_one_profile(args: argparse.Namespace, profile: ExportProfile) -> dict[str, object]:
    artifact_name = f"{args.artifact_stem}-{profile.name}"
    output_dir, packaged_zip = build_named_coreml_paths(
        output_root=args.output_root,
        packaged_root=args.packaged_root,
        artifact_name=artifact_name,
    )

    reused_existing = output_dir.exists() and not args.force_rebuild
    manifest_model_path: str | None = None
    manifest_model_kind: str | None = None

    if reused_existing:
        output_name, manifest_model_path, manifest_model_kind = _load_existing_manifest_coreml_fields(
            output_dir
        )
        if output_name is None:
            raise RuntimeError(
                f"existing artifact is missing outputName in manifest: {output_dir}"
            )

        if args.compile_mlmodelc:
            selected_model_path, selected_model_kind = _ensure_compiled_model_artifact(
                output_dir,
                manifest_model_path=manifest_model_path,
            )
        else:
            selected_model_path, selected_model_kind = _resolve_existing_model_artifact(
                output_dir,
                manifest_model_path=manifest_model_path,
                manifest_model_kind=manifest_model_kind,
            )
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        package_model_path, output_name = _convert_coreml(
            model_dir=args.model_dir,
            output_dir=output_dir,
            context_length=profile.context_length,
            q_bits=args.q_bits,
            q_group_size=args.q_group_size,
            q_mode=args.q_mode,
            decode_only=profile.decode_only,
            use_state_cache=profile.use_state_cache,
        )
        if package_model_path.name != MODEL_PACKAGE_FILE_NAME:
            raise RuntimeError(f"unexpected model file name: {package_model_path.name}")

        copy_runtime_files(
            model_dir=args.model_dir,
            output_dir=output_dir,
        )
        if args.compile_mlmodelc:
            compiled_model_path = _compile_mlpackage_to_mlmodelc(
                mlpackage_path=package_model_path,
                compiled_model_path=output_dir / MODEL_COMPILED_DIR_NAME,
            )
            selected_model_path = compiled_model_path.name
            selected_model_kind = "mlmodelc"
        else:
            selected_model_path = package_model_path.name
            selected_model_kind = "mlpackage"

    if (
        (not reused_existing)
        or manifest_model_path != selected_model_path
        or manifest_model_kind != selected_model_kind
    ):
        write_translation_manifest(
            output_dir=output_dir,
            coreml_kind=selected_model_kind,
            coreml_path=selected_model_path,
            context_length=profile.context_length,
            output_name=output_name,
        )

    packaged_primary_model_path, packaged_extra_model_paths = _resolve_packaging_model_artifacts(
        output_dir,
        selected_model_path=selected_model_path,
        package_include_mlmodelc=args.package_include_mlmodelc,
    )

    package_coreml_bundle(
        source_dir=output_dir,
        model_file_name=packaged_primary_model_path,
        extra_model_file_names=packaged_extra_model_paths,
        zip_path=packaged_zip,
    )

    packaged_model_files = [packaged_primary_model_path, *packaged_extra_model_paths]

    return {
        "profile": profile.name,
        "reused_existing": reused_existing,
        "use_state_cache": profile.use_state_cache,
        "decode_only": profile.decode_only,
        "context_length": profile.context_length,
        "model_dir": str(output_dir),
        "model_path": str(output_dir / selected_model_path),
        "model_kind": selected_model_kind,
        "packaged_zip": str(packaged_zip),
        "packaged_model_files": packaged_model_files,
        "output_name": output_name,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {args.context_length}")
    if args.optimized_context_length <= 0:
        raise ValueError(
            f"optimized_context_length must be > 0, got {args.optimized_context_length}"
        )

    profiles = _resolve_profiles(args)
    results = [_run_one_profile(args, profile) for profile in profiles]

    return {
        "status": "completed",
        "profiles": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HY-MT to CoreML W8 artifacts.")
    parser.add_argument(
        "--profile",
        choices=[
            PROFILE_CACHE,
            PROFILE_NOCACHE,
            PROFILE_CACHE_OPT,
            PROFILE_CACHE_TIERS,
            PROFILE_ALL,
        ],
        default=PROFILE_CACHE,
        help="which export profile to generate",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_COREML_OUTPUT_ROOT)
    parser.add_argument("--packaged-root", type=Path, default=DEFAULT_COREML_PACKAGED_ROOT)
    parser.add_argument("--artifact-stem", default=DEFAULT_COREML_ARTIFACT_STEM)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument(
        "--optimized-context-length",
        type=int,
        default=DEFAULT_OPTIMIZED_CONTEXT_LENGTH,
        help="context length used by cache-opt profile",
    )
    parser.add_argument(
        "--cache-tier-contexts",
        default=DEFAULT_CACHE_TIER_CONTEXTS,
        help="comma-separated context lengths used by cache-tiers profile",
    )
    parser.add_argument(
        "--cache-tier-prefix",
        default=DEFAULT_CACHE_TIER_PREFIX,
        help="variant name prefix for cache tiers; final variant is <prefix><context>",
    )
    parser.add_argument("--q-bits", type=int, default=DEFAULT_COREML_Q_BITS)
    parser.add_argument("--q-group-size", type=int, default=DEFAULT_COREML_Q_GROUP_SIZE)
    parser.add_argument("--q-mode", default=DEFAULT_COREML_Q_MODE)
    parser.add_argument(
        "--decode-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COREML_DECODE_ONLY,
        help="decode-only export for cache profiles",
    )
    parser.add_argument(
        "--force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rebuild artifact even if fixed-name output already exists",
    )
    parser.add_argument(
        "--compile-mlmodelc",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COMPILE_MLMODELC,
        help="compile exported .mlpackage to .mlmodelc and write manifest to the compiled path",
    )
    parser.add_argument(
        "--package-include-mlmodelc",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PACKAGE_INCLUDE_MLMODELC,
        help="ensure packaged zip contains a sibling .mlmodelc so runtime import can skip compilation",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


COREML_ROOT_DIR = Path("models/translation/converted/coreml-int8")
COREML_ARTIFACT_STEM = "hy-mt1.5-1.8b-coreml-int8"
COREML_VARIANT = "cache"
TOKENIZER_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
TARGET_LANGUAGE = "English"
SOURCE_TEXT = "今天下午三点半在5A会议室开会。"
SYSTEM_PROMPT = "You are a translation engine."
MAX_NEW_TOKENS = 64
CONTEXT_LENGTH = 256
COMPUTE_UNIT = "cpuAndNeuralEngine"
PRINT_GENERATED_TEXT = True


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024.0 * 1024.0), 3)


def _current_rss_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
    except Exception:
        return None

    if not output:
        return None
    try:
        # ps rss is in KB
        return int(output) * 1024
    except ValueError:
        return None


def _load_manifest_coreml_fields(
    coreml_dir: Path,
) -> tuple[str | None, str | None, str | None]:
    manifest_path = coreml_dir / "translation-manifest.json"
    if not manifest_path.is_file():
        return None, None, None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    coreml = payload.get("coreml", {})
    model_path = coreml.get("path")
    output_name = coreml.get("outputName")
    model_kind = coreml.get("kind")

    model_path_value = model_path if isinstance(model_path, str) and model_path else None
    output_name_value = output_name if isinstance(output_name, str) and output_name else None
    model_kind_value = model_kind if isinstance(model_kind, str) and model_kind else None
    return model_path_value, output_name_value, model_kind_value


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _manifest_relative_model_path(coreml_dir: Path, model_path: Path) -> str:
    try:
        return model_path.relative_to(coreml_dir).as_posix()
    except ValueError:
        return str(model_path)


def _update_manifest_coreml_target(
    *,
    coreml_dir: Path,
    model_path: Path,
) -> bool:
    manifest_path = coreml_dir / "translation-manifest.json"
    if not manifest_path.is_file():
        return False

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    coreml = payload.get("coreml")
    if not isinstance(coreml, dict):
        coreml = {}
        payload["coreml"] = coreml

    expected_path = _manifest_relative_model_path(coreml_dir, model_path)
    expected_kind = "mlmodelc" if model_path.name.endswith(".mlmodelc") else "mlpackage"
    changed = False

    if coreml.get("path") != expected_path:
        coreml["path"] = expected_path
        changed = True
    if coreml.get("kind") != expected_kind:
        coreml["kind"] = expected_kind
        changed = True

    if changed:
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return changed


def _materialize_compiled_model_in_place(
    *,
    coreml_dir: Path,
    model_path: Path,
) -> tuple[Path, bool]:
    if model_path.suffix != ".mlpackage":
        return model_path, False

    compiled_path = model_path.with_suffix(".mlmodelc")
    compiled_created = False
    if not compiled_path.exists():
        _remove_path(compiled_path)
        compiled_output = Path(
            ct.models.utils.compile_model(
                str(model_path),
                destination_path=str(compiled_path),
            )
        )
        if not compiled_output.exists():
            raise RuntimeError(
                f"compiled model was not created in model dir: {compiled_output}"
            )
        compiled_created = True

    _update_manifest_coreml_target(
        coreml_dir=coreml_dir,
        model_path=compiled_path,
    )
    return compiled_path, compiled_created


def _resolve_model_path(
    coreml_dir: Path,
    *,
    manifest_model_path: str | None,
) -> Path:
    if manifest_model_path:
        manifest_candidate = Path(manifest_model_path)
        candidate = (
            manifest_candidate.expanduser().resolve()
            if manifest_candidate.is_absolute()
            else (coreml_dir / manifest_candidate)
        )
        if candidate.exists():
            return candidate

    fallback_candidates = [
        coreml_dir / "hy_mt_w8_from_torch.mlmodelc",
        coreml_dir / "hy_mt_w8_from_torch.mlpackage",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "missing model artifact under "
        f"{coreml_dir}; checked manifest path={manifest_model_path}"
    )


def _resolve_coreml_dir(
    *,
    coreml_dir: Path | None,
    coreml_root_dir: Path,
    artifact_stem: str,
    variant: str,
) -> Path:
    if coreml_dir is not None:
        resolved = coreml_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"coreml dir does not exist: {resolved}")
        return resolved

    root = coreml_root_dir.expanduser().resolve()
    preferred = root / f"{artifact_stem}-{variant}"
    if preferred.is_dir():
        return preferred

    legacy_fixed = root / "hy-mt1.5-1.8b-coreml"
    if legacy_fixed.is_dir():
        return legacy_fixed

    legacy_prefix = f"{artifact_stem}-"
    if root.is_dir():
        candidates = sorted(
            [
                child
                for child in root.iterdir()
                if child.is_dir() and child.name.startswith(legacy_prefix)
            ]
        )
        if candidates:
            return candidates[-1]

    raise RuntimeError(
        f"no coreml output dir found under: {root}; expected {preferred}"
    )


def _resolve_compute_unit(name: str) -> ct.ComputeUnit:
    if name == "cpuAndNeuralEngine":
        return ct.ComputeUnit.CPU_AND_NE
    if name == "all":
        return ct.ComputeUnit.ALL
    if name == "cpuAndGPU":
        return ct.ComputeUnit.CPU_AND_GPU
    if name == "cpuOnly":
        return ct.ComputeUnit.CPU_ONLY
    raise RuntimeError(f"unsupported compute unit: {name}")


def _build_prompt(target_language: str, source_text: str) -> str:
    return (
        f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
        f"{source_text}"
    )


def _load_eos_ids(tokenizer_dir: Path, tokenizer) -> set[int]:
    eos_ids: set[int] = set()
    generation_config_path = tokenizer_dir / "generation_config.json"
    if generation_config_path.is_file():
        payload = json.loads(generation_config_path.read_text(encoding="utf-8"))
        eos_raw = payload.get("eos_token_id")
        if isinstance(eos_raw, int):
            eos_ids.add(eos_raw)
        elif isinstance(eos_raw, list):
            eos_ids.update(int(v) for v in eos_raw if isinstance(v, int))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    return eos_ids


def _load_coreml_predictor(model_path: Path, compute_unit: ct.ComputeUnit):
    if model_path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(
            str(model_path),
            compute_units=compute_unit,
        )
    return ct.models.MLModel(
        str(model_path),
        compute_units=compute_unit,
    )


def _resolve_output_name(predictor, manifest_output_name: str | None) -> str:
    spec_output_names: list[str] = []
    get_spec = getattr(predictor, "get_spec", None)
    if callable(get_spec):
        spec = get_spec()
        spec_output_names = [out.name for out in spec.description.output]

    if manifest_output_name and manifest_output_name in spec_output_names:
        return manifest_output_name
    if "logits" in spec_output_names:
        return "logits"
    if len(spec_output_names) == 1:
        return spec_output_names[0]
    if manifest_output_name:
        return manifest_output_name
    return "logits"


def _resolve_max_query_tokens(predictor, input_name: str = "input_ids") -> int | None:
    get_spec = getattr(predictor, "get_spec", None)
    if not callable(get_spec):
        return None
    spec = get_spec()
    for feature in spec.description.input:
        if feature.name != input_name:
            continue
        multi_array = feature.type.multiArrayType
        if (
            multi_array.HasField("shapeRange")
            and len(multi_array.shapeRange.sizeRanges) >= 2
        ):
            seq_range = multi_array.shapeRange.sizeRanges[1]
            if seq_range.upperBound > 0:
                return int(seq_range.upperBound)
            return int(seq_range.lowerBound)
        shape = list(multi_array.shape)
        if len(shape) >= 2 and shape[1] > 0:
            return int(shape[1])
    return None


def _model_has_states(predictor) -> bool:
    make_state = getattr(predictor, "make_state", None)
    if callable(make_state):
        return True
    get_spec = getattr(predictor, "get_spec", None)
    if not callable(get_spec):
        return False
    spec = get_spec()
    return len(spec.description.state) > 0


def _predict_with_optional_state(predictor, inputs: dict[str, np.ndarray], state):
    if state is None:
        return predictor.predict(inputs)
    try:
        return predictor.predict(inputs, state=state)
    except TypeError:
        return predictor.predict(inputs)


def _run(args: argparse.Namespace) -> dict[str, object]:
    end_to_end_start = time.perf_counter()

    coreml_dir = _resolve_coreml_dir(
        coreml_dir=args.coreml_dir,
        coreml_root_dir=args.coreml_root_dir,
        artifact_stem=args.artifact_stem,
        variant=args.variant,
    )
    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    manifest_model_path, manifest_output_name, _manifest_model_kind = _load_manifest_coreml_fields(
        coreml_dir
    )
    model_path = _resolve_model_path(
        coreml_dir,
        manifest_model_path=manifest_model_path,
    )
    compiled_materialized = False
    if args.materialize_compiled_model:
        model_path, compiled_materialized = _materialize_compiled_model_in_place(
            coreml_dir=coreml_dir,
            model_path=model_path,
        )
    compute_unit = _resolve_compute_unit(args.compute_unit)

    tmpdir = coreml_dir / "CoreMLTemp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = str(tmpdir)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    eos_ids = _load_eos_ids(tokenizer_dir, tokenizer)
    prompt = _build_prompt(args.target_language, args.source_text)
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = prompt_ids[-args.context_length :]
    if not prompt_ids:
        raise RuntimeError("prompt ids is empty")

    translation_start = time.perf_counter()
    memory_before_load = _current_rss_bytes()

    load_start = time.perf_counter()
    predictor = _load_coreml_predictor(model_path, compute_unit)
    load_elapsed = time.perf_counter() - load_start
    output_name = _resolve_output_name(predictor, manifest_output_name)
    max_query_tokens = _resolve_max_query_tokens(predictor)

    state = None
    state_error: str | None = None
    has_states = _model_has_states(predictor)
    make_state = getattr(predictor, "make_state", None)
    if has_states and callable(make_state):
        try:
            state = make_state()
        except Exception as exc:
            state_error = str(exc)

    if has_states and max_query_tokens is None:
        # CompiledMLModel may not expose spec; stateful decode-only export expects [1, 1].
        max_query_tokens = 1

    memory_after_load = _current_rss_bytes()

    prefill_seconds = 0.0
    stateful_runtime = has_states and state is not None

    if stateful_runtime and max_query_tokens == 1 and len(prompt_ids) > 1:
        prefill_start = time.perf_counter()
        for token_id in prompt_ids[:-1]:
            _predict_with_optional_state(
                predictor,
                {"input_ids": np.asarray([[token_id]], dtype=np.int32)},
                state,
            )
        prefill_seconds = time.perf_counter() - prefill_start
        current_input = np.asarray([[prompt_ids[-1]]], dtype=np.int32)
    else:
        if max_query_tokens is not None and len(prompt_ids) > max_query_tokens:
            prompt_ids = prompt_ids[-max_query_tokens:]
        current_input = np.asarray([prompt_ids], dtype=np.int32)
    token_history = list(prompt_ids)

    generated_ids: list[int] = []
    step_times: list[float] = []

    for _ in range(args.max_new_tokens):
        step_start = time.perf_counter()
        output = _predict_with_optional_state(
            predictor,
            {"input_ids": current_input},
            state,
        )
        step_times.append(time.perf_counter() - step_start)

        if output_name not in output:
            if len(output) == 1:
                output_name = next(iter(output.keys()))
            else:
                raise RuntimeError(
                    f"output missing {output_name}, keys={list(output.keys())}"
                )
        logits = np.asarray(output[output_name])
        if logits.ndim == 3:
            next_scores = logits[0, -1, :]
        elif logits.ndim == 2:
            next_scores = logits[0, :]
        else:
            raise RuntimeError(f"unexpected logits shape: {logits.shape}")

        next_token = int(np.argmax(next_scores))
        if next_token in eos_ids:
            break
        generated_ids.append(next_token)
        if stateful_runtime:
            current_input = np.asarray([[next_token]], dtype=np.int32)
        else:
            token_history.append(next_token)
            if max_query_tokens is not None and len(token_history) > max_query_tokens:
                token_history = token_history[-max_query_tokens:]
            current_input = np.asarray([token_history], dtype=np.int32)

    memory_after_generate = _current_rss_bytes()
    translation_total_seconds = time.perf_counter() - translation_start

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("generated empty text")

    end_to_end_seconds = time.perf_counter() - end_to_end_start

    memory_delta_load = None
    if memory_before_load is not None and memory_after_load is not None:
        memory_delta_load = memory_after_load - memory_before_load

    result = {
        "status": "passed",
        "model_path": str(model_path),
        "tokenizer_dir": str(tokenizer_dir),
        "compute_unit": args.compute_unit,
        "variant": args.variant,
        "compiled_materialized": compiled_materialized,
        "stateful_runtime": stateful_runtime,
        "state_error": state_error,
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(generated_ids),
        "output_name": output_name,
        "max_query_tokens": max_query_tokens,
        "prefill_seconds": round(prefill_seconds, 3),
        "load_seconds": round(load_elapsed, 3),
        "first_token_latency_seconds": round(step_times[0], 3) if step_times else None,
        "generate_seconds": round(float(sum(step_times)), 3),
        "translation_total_seconds": round(translation_total_seconds, 3),
        "end_to_end_seconds": round(end_to_end_seconds, 3),
        "memory_rss_before_load_bytes": memory_before_load,
        "memory_rss_after_load_bytes": memory_after_load,
        "memory_rss_after_generate_bytes": memory_after_generate,
        "memory_rss_before_load_mb": _bytes_to_mb(memory_before_load),
        "memory_rss_after_load_mb": _bytes_to_mb(memory_after_load),
        "memory_rss_after_generate_mb": _bytes_to_mb(memory_after_generate),
        "memory_rss_delta_load_bytes": memory_delta_load,
        "memory_rss_delta_load_mb": _bytes_to_mb(memory_delta_load),
        "generated_text": text,
    }
    shutil.rmtree(tmpdir, ignore_errors=True)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CoreML translation smoke/benchmark.")
    parser.add_argument("--coreml-dir", type=Path, default=None)
    parser.add_argument("--coreml-root-dir", type=Path, default=COREML_ROOT_DIR)
    parser.add_argument("--artifact-stem", default=COREML_ARTIFACT_STEM)
    parser.add_argument("--variant", default=COREML_VARIANT)
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--target-language", default=TARGET_LANGUAGE)
    parser.add_argument("--source-text", default=SOURCE_TEXT)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--compute-unit", default=COMPUTE_UNIT)
    parser.add_argument(
        "--materialize-compiled-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if loading .mlpackage, compile and store sibling .mlmodelc in model dir before testing",
    )
    parser.add_argument(
        "--print-generated-text",
        action=argparse.BooleanOptionalAction,
        default=PRINT_GENERATED_TEXT,
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="print only summary json (no generated text trailer)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    result = _run(args)

    summary = {
        "status": "passed",
        "model_path": result["model_path"],
        "tokenizer_dir": result["tokenizer_dir"],
        "inference": {
            "variant": result["variant"],
            "compute_unit": result["compute_unit"],
            "compiled_materialized": result["compiled_materialized"],
            "stateful_runtime": result["stateful_runtime"],
            "state_error": result["state_error"],
            "load_seconds": result["load_seconds"],
            "prefill_seconds": result["prefill_seconds"],
            "first_token_latency_seconds": result["first_token_latency_seconds"],
            "generate_seconds": result["generate_seconds"],
            "translation_total_seconds": result["translation_total_seconds"],
            "end_to_end_seconds": result["end_to_end_seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "output_tokens": result["output_tokens"],
            "memory_rss_before_load_mb": result["memory_rss_before_load_mb"],
            "memory_rss_after_load_mb": result["memory_rss_after_load_mb"],
            "memory_rss_after_generate_mb": result["memory_rss_after_generate_mb"],
            "memory_rss_delta_load_mb": result["memory_rss_delta_load_mb"],
            "memory_rss_before_load_bytes": result["memory_rss_before_load_bytes"],
            "memory_rss_after_load_bytes": result["memory_rss_after_load_bytes"],
            "memory_rss_after_generate_bytes": result["memory_rss_after_generate_bytes"],
            "memory_rss_delta_load_bytes": result["memory_rss_delta_load_bytes"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    should_print_text = args.print_generated_text and not args.json_only
    if should_print_text:
        print("\n===== GENERATED TEXT =====")
        print(result["generated_text"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

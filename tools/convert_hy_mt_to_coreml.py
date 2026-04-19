#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from torch.export import Dim
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")
DEFAULT_OUTPUT_DIR = Path("models/translation/converted/coreml-fp16/hy-mt1.5-1.8b-coreml")
DEFAULT_PACKAGED_ZIP = Path("models/translation/packaged/hy-mt1.5-1.8b-coreml-fp16.zip")
DEFAULT_REPORT_PATH = Path("models/translation/reports/hy-mt1.5-1.8b-coreml-fp16-report.json")
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_MAX_OUTPUT_LENGTH = 256
DEFAULT_SUPPORTED_LANGUAGES = ["zho", "eng", "jpn", "kor"]
DEFAULT_PREFERRED_COMPUTE_UNITS = ["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"]
DEFAULT_RUNTIME_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


@dataclass
class ConversionSummary:
    status: str
    model_dir: str
    output_dir: str
    packaged_zip: str
    report_path: str
    context_length: int
    max_output_length: int
    prompt_style: str = "hy_mt_coreml_chat_v1"
    family: str = "coreml_causal_llm"
    elapsed_seconds: float = 0.0
    steps: list[dict[str, Any]] = field(default_factory=list)
    python_smoke: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class SliceUpdateKeyValueCache:
    is_compileable = True

    def __init__(
        self,
        *,
        key_caches: list[torch.Tensor],
        value_caches: list[torch.Tensor],
        max_cache_len: int,
    ) -> None:
        self.key_caches = key_caches
        self.value_caches = value_caches
        self.max_cache_len = max_cache_len
        self.is_sliding = [False] * len(key_caches)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-2], device=key_states.device)

        layer_key_cache = self.key_caches[layer_idx]
        layer_value_cache = self.value_caches[layer_idx]
        one_hot = torch.nn.functional.one_hot(
            cache_position.to(torch.int64),
            num_classes=self.max_cache_len,
        ).to(layer_key_cache.dtype)
        selection = one_hot.sum(dim=0).clamp(0, 1).view(1, 1, self.max_cache_len, 1)

        key_updates = torch.matmul(key_states.to(layer_key_cache.dtype).permute(0, 1, 3, 2), one_hot).permute(
            0, 1, 3, 2
        )
        value_updates = torch.matmul(
            value_states.to(layer_value_cache.dtype).permute(0, 1, 3, 2), one_hot
        ).permute(0, 1, 3, 2)

        layer_key_cache.mul_(1.0 - selection)
        layer_key_cache.add_(key_updates)
        layer_value_cache.mul_(1.0 - selection)
        layer_value_cache.add_(value_updates)

        return layer_key_cache.to(key_states.dtype), layer_value_cache.to(value_states.dtype)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        del cache_position, layer_idx
        return self.max_cache_len, 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.max_cache_len


class StatefulHunYuanForCoreML(torch.nn.Module):
    def __init__(self, model: AutoModelForCausalLM, max_cache_len: int) -> None:
        super().__init__()
        self.model = model
        self.max_cache_len = max_cache_len
        config = model.config
        num_layers = int(config.num_hidden_layers)
        self.num_layers = num_layers
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        layer_cache_shape = (1, num_kv_heads, max_cache_len, head_dim)
        for layer_idx in range(num_layers):
            self.register_buffer(f"key_cache_{layer_idx}", torch.zeros(layer_cache_shape, dtype=torch.float16))
            self.register_buffer(f"value_cache_{layer_idx}", torch.zeros(layer_cache_shape, dtype=torch.float16))
        self.register_buffer("cache_position", torch.zeros((1,), dtype=torch.float16))

    def _layer_key_caches(self) -> list[torch.Tensor]:
        return [getattr(self, f"key_cache_{layer_idx}") for layer_idx in range(self.num_layers)]

    def _layer_value_caches(self) -> list[torch.Tensor]:
        return [getattr(self, f"value_cache_{layer_idx}") for layer_idx in range(self.num_layers)]

    def reset_cache(self) -> None:
        for cache in self._layer_key_caches():
            cache.zero_()
        for cache in self._layer_value_caches():
            cache.zero_()
        self.cache_position.zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(torch.int64)
        start_position = self.cache_position.to(torch.int64)
        local_positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.int64)
        cache_position = local_positions + start_position
        kv_positions = torch.arange(self.max_cache_len, device=input_ids.device, dtype=torch.int64)
        allowed = kv_positions.unsqueeze(0) <= cache_position.unsqueeze(1)
        zero_mask = torch.zeros(allowed.shape, dtype=torch.float16, device=input_ids.device)
        neg_inf_mask = torch.full(allowed.shape, -1.0e4, dtype=torch.float16, device=input_ids.device)
        attention_mask = torch.where(allowed, zero_mask, neg_inf_mask).unsqueeze(0).unsqueeze(0)

        past_key_values = SliceUpdateKeyValueCache(
            key_caches=self._layer_key_caches(),
            value_caches=self._layer_value_caches(),
            max_cache_len=self.max_cache_len,
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,  # type: ignore[arg-type]
            use_cache=True,
            cache_position=cache_position,
            return_dict=True,
        )

        next_position = torch.clamp(cache_position[-1:] + 1, max=self.max_cache_len)
        self.cache_position.copy_(next_position.to(self.cache_position.dtype))
        return outputs.logits.to(torch.float16)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert HY-MT1.5-1.8B to a stateful FP16 CoreML bundle for translator-new."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--packaged-zip", default=str(DEFAULT_PACKAGED_ZIP))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-output-length", type=int, default=DEFAULT_MAX_OUTPUT_LENGTH)
    parser.add_argument(
        "--compute-units",
        default="all,cpuAndNeuralEngine,cpuAndGPU,cpuOnly",
        help="Comma-separated compute unit preference list for manifest runtime section.",
    )
    parser.add_argument(
        "--run-python-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run local Python smoke generation against compiled model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Per-case cap for smoke generation.")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _copy_runtime_files(model_dir: Path, output_dir: Path) -> None:
    missing: list[str] = []
    for file_name in DEFAULT_RUNTIME_FILES:
        src = model_dir / file_name
        dst = output_dir / file_name
        if not src.is_file():
            missing.append(file_name)
            continue
        shutil.copy2(src, dst)
    if missing:
        raise RuntimeError(f"missing required runtime files: {missing}")


def _write_manifest(
    *,
    output_dir: Path,
    context_length: int,
    max_output_length: int,
    preferred_compute_units: list[str],
) -> Path:
    manifest = {
        "family": "coreml_causal_llm",
        "promptStyle": "hy_mt_coreml_chat_v1",
        "supportedLanguages": DEFAULT_SUPPORTED_LANGUAGES,
        "coreml": {
            "packageFile": "causal_lm.mlpackage",
            "stateful": True,
        },
        "runtime": {
            "contextLength": context_length,
            "preferredComputeUnits": preferred_compute_units,
            "keepsRuntimeLoaded": True,
        },
        "generation": {
            "maxInputLength": context_length,
            "maxOutputLength": max_output_length,
            "temperature": 0.0,
            "topP": 1.0,
            "repetitionPenalty": None,
        },
        "tokenizer": {
            "kind": "huggingface_tokenizer_json",
        },
    }
    manifest_path = output_dir / "translation-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _compile_mlpackage_to_output(mlpackage_path: Path, output_mlmodelc_path: Path) -> Path:
    compiled_tmp_path = Path(ct.models.utils.compile_model(str(mlpackage_path)))
    output_mlmodelc_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mlmodelc_path.exists():
        shutil.rmtree(output_mlmodelc_path)
    shutil.copytree(compiled_tmp_path, output_mlmodelc_path)
    return output_mlmodelc_path


def _make_zip_with_parent(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(
        ["/usr/bin/ditto", "-c", "-k", "--keepParent", str(source_dir), str(zip_path)],
        check=True,
    )


def _resolve_compute_units_for_manifest(raw_value: str) -> list[str]:
    units = [unit.strip() for unit in raw_value.split(",") if unit.strip()]
    if not units:
        return DEFAULT_PREFERRED_COMPUTE_UNITS
    return units


def _resolve_coreml_compute_unit(preferred_units: list[str]) -> ct.ComputeUnit:
    for unit in preferred_units:
        if unit == "cpuAndNeuralEngine":
            return ct.ComputeUnit.CPU_AND_NE
        if unit == "all":
            return ct.ComputeUnit.ALL
        if unit == "cpuAndGPU":
            return ct.ComputeUnit.CPU_AND_GPU
        if unit == "cpuOnly":
            return ct.ComputeUnit.CPU_ONLY
    return ct.ComputeUnit.ALL


def _convert_model(
    *,
    model_dir: Path,
    output_dir: Path,
    context_length: int,
    preferred_units: list[str],
) -> tuple[Path, dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config._attn_implementation = "eager"
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.rope_type = "default"
    wrapper = StatefulHunYuanForCoreML(model=model, max_cache_len=context_length)
    wrapper.eval()
    wrapper.reset_cache()

    sample_input_ids = torch.ones((1, 8), dtype=torch.int32)
    dynamic_shapes = {
        "input_ids": {
            1: Dim("query_length", min=1, max=context_length),
        }
    }

    exported_program = torch.export.export(
        wrapper,
        args=(sample_input_ids,),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    exported_program = exported_program.run_decompositions({})

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hy-mt-coreml-export-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        mlpackage_path = tmp_dir_path / "causal_lm.mlpackage"
        default_passes = list(ct.PassPipeline.DEFAULT.passes)
        custom_pass_pipeline = ct.PassPipeline(pass_names=default_passes, pipeline_name="hy_mt_coreml_export")
        custom_pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

        coreml_model = ct.convert(
            exported_program,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            pass_pipeline=custom_pass_pipeline,
            compute_units=_resolve_coreml_compute_unit(preferred_units),
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=context_length, default=1)),
                    dtype=np.int32,
                ),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            states=(
                [
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=tuple(getattr(wrapper, f"key_cache_{layer_idx}").shape),
                            dtype=np.float16,
                        ),
                        name=f"key_cache_{layer_idx}",
                    )
                    for layer_idx in range(wrapper.num_layers)
                ]
                + [
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=tuple(getattr(wrapper, f"value_cache_{layer_idx}").shape),
                            dtype=np.float16,
                        ),
                        name=f"value_cache_{layer_idx}",
                    )
                    for layer_idx in range(wrapper.num_layers)
                ]
                + [
                    ct.StateType(
                        wrapped_type=ct.TensorType(shape=tuple(wrapper.cache_position.shape), dtype=np.float16),
                        name="cache_position",
                    )
                ]
            ),
        )
        coreml_model.save(str(mlpackage_path))

        compiled_output_path = output_dir / "Compiled" / "causal_lm.mlmodelc"
        compiled_path = _compile_mlpackage_to_output(mlpackage_path, compiled_output_path)
        spec = coreml_model.get_spec()

    conversion_meta = {
        "spec_inputs": [desc.name for desc in spec.description.input],
        "spec_outputs": [desc.name for desc in spec.description.output],
        "state_count": len(spec.description.state),
        "state_names": [desc.name for desc in spec.description.state],
        "compiled_model_path": str(compiled_path),
    }
    return compiled_path, conversion_meta


def _load_generation_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_prompt(target_language: str, source_text: str) -> str:
    return (
        f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
        f"{source_text}"
    )


def _run_python_smoke(
    *,
    model_dir: Path,
    compiled_model_path: Path,
    context_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    generation_config = _load_generation_config(model_dir)
    eos_ids = set()
    eos_raw = generation_config.get("eos_token_id")
    if isinstance(eos_raw, int):
        eos_ids.add(eos_raw)
    elif isinstance(eos_raw, list):
        eos_ids.update(int(v) for v in eos_raw if isinstance(v, int))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    compiled_model = ct.models.CompiledMLModel(str(compiled_model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    state = compiled_model.make_state()
    if state is None:
        raise RuntimeError("compiled model failed to create state")

    input_name = "input_ids"
    logits_name = "logits"

    test_cases = [
        ("zh-en", "English", "今天下午 3:30 在 5A 会议室同步 v1.5.8 发布计划。"),
        ("zh-ja", "Japanese", "今天下午 3:30 在 5A 会议室同步 v1.5.8 发布计划。"),
    ]
    case_reports: list[dict[str, Any]] = []

    for route, target_language, source_text in test_cases:
        state = compiled_model.make_state()
        messages = [
            {"role": "system", "content": "You are a translation engine."},
            {"role": "user", "content": _build_prompt(target_language, source_text)},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = prompt_ids[-context_length:]
        current_input = np.asarray([prompt_ids], dtype=np.int32)
        generated_ids: list[int] = []
        step_times: list[float] = []

        for _ in range(max_new_tokens):
            step_start = time.monotonic()
            output = compiled_model.predict({input_name: current_input}, state=state)
            step_times.append(time.monotonic() - step_start)
            if logits_name not in output:
                raise RuntimeError(f"compiled model output missing '{logits_name}', keys={list(output.keys())}")
            logits = np.asarray(output[logits_name])
            if logits.ndim == 3:
                next_token_scores = logits[0, -1, :]
            elif logits.ndim == 2:
                next_token_scores = logits[0, :]
            else:
                raise RuntimeError(f"unsupported logits rank from compiled model: {logits.shape}")

            next_token = int(np.argmax(next_token_scores))
            if next_token in eos_ids:
                break
            generated_ids.append(next_token)
            current_input = np.asarray([[next_token]], dtype=np.int32)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError(f"python smoke generated empty translation for route={route}")
        case_reports.append(
            {
                "route": route,
                "generated_text": text,
                "generated_token_count": len(generated_ids),
                "first_token_latency_seconds": step_times[0] if step_times else None,
                "total_generation_seconds": float(sum(step_times)),
            }
        )

    return {
        "status": "passed",
        "input_name": input_name,
        "logits_name": logits_name,
        "eos_ids": sorted(eos_ids),
        "cases": case_reports,
    }


def run(args: argparse.Namespace) -> ConversionSummary:
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    packaged_zip = Path(args.packaged_zip)
    report_path = Path(args.report_path)
    preferred_units = _resolve_compute_units_for_manifest(args.compute_units)

    summary = ConversionSummary(
        status="running",
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        packaged_zip=str(packaged_zip),
        report_path=str(report_path),
        context_length=int(args.context_length),
        max_output_length=int(args.max_output_length),
    )
    started_at = time.monotonic()

    if not model_dir.is_dir():
        raise RuntimeError(f"model_dir does not exist: {model_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.steps.append({"name": "convert_to_coreml", "status": "running"})
    compiled_model_path, conversion_meta = _convert_model(
        model_dir=model_dir,
        output_dir=output_dir,
        context_length=int(args.context_length),
        preferred_units=preferred_units,
    )
    summary.steps[-1] = {"name": "convert_to_coreml", "status": "completed", "meta": conversion_meta}

    summary.steps.append({"name": "copy_runtime_files", "status": "running"})
    _copy_runtime_files(model_dir=model_dir, output_dir=output_dir)
    summary.steps[-1] = {"name": "copy_runtime_files", "status": "completed"}

    summary.steps.append({"name": "write_manifest", "status": "running"})
    manifest_path = _write_manifest(
        output_dir=output_dir,
        context_length=int(args.context_length),
        max_output_length=int(args.max_output_length),
        preferred_compute_units=preferred_units,
    )
    summary.steps[-1] = {
        "name": "write_manifest",
        "status": "completed",
        "manifest_path": str(manifest_path),
    }

    summary.steps.append({"name": "package_zip", "status": "running"})
    _make_zip_with_parent(source_dir=output_dir, zip_path=packaged_zip)
    summary.steps[-1] = {
        "name": "package_zip",
        "status": "completed",
        "zip_path": str(packaged_zip),
        "zip_size_bytes": packaged_zip.stat().st_size,
    }

    if bool(args.run_python_smoke):
        summary.steps.append({"name": "python_smoke", "status": "running"})
        summary.python_smoke = _run_python_smoke(
            model_dir=model_dir,
            compiled_model_path=compiled_model_path,
            context_length=int(args.context_length),
            max_new_tokens=int(args.max_new_tokens),
        )
        summary.steps[-1] = {"name": "python_smoke", "status": "completed", "result": summary.python_smoke}

    summary.status = "completed"
    summary.elapsed_seconds = time.monotonic() - started_at
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    report_path = Path(args.report_path)
    try:
        summary = run(args)
        payload = {
            "status": summary.status,
            "summary": {
                "model_dir": summary.model_dir,
                "output_dir": summary.output_dir,
                "packaged_zip": summary.packaged_zip,
                "context_length": summary.context_length,
                "max_output_length": summary.max_output_length,
                "elapsed_seconds": round(summary.elapsed_seconds, 4),
            },
            "steps": summary.steps,
            "python_smoke": summary.python_smoke,
        }
        _write_json(report_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nReport written to: {report_path}")
        return 0
    except Exception as exc:
        failure_payload = {
            "status": "failed",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json(report_path, failure_payload)
        print(json.dumps(failure_payload, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nFailure report written to: {report_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

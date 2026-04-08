from __future__ import annotations

from pathlib import Path

from .paths import DEFAULT_CONFIG_PATH, resolve_repo_path
from .schemas import BenchmarkConfig, DecisionPolicy, DecodeConfig, MetricsConfig, QuantizationConfig


REQUIRED_TOP_LEVEL_KEYS = {
    "artifacts_root",
    "results_root",
    "systems",
    "routes",
    "quantization",
    "decode",
    "metrics",
    "decision_policy",
}


def load_config(config_path: Path | None = None) -> BenchmarkConfig:
    import yaml

    resolved_path = config_path or DEFAULT_CONFIG_PATH
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))

    if not isinstance(raw_payload, dict):
        raise ValueError(f"Benchmark config must be a mapping: {resolved_path}")

    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(raw_payload)
    if missing_keys:
        raise ValueError(f"Missing config keys: {sorted(missing_keys)}")

    artifacts_root = resolve_repo_path(raw_payload["artifacts_root"])
    results_root = resolve_repo_path(raw_payload["results_root"])
    systems = tuple(_require_string_list(raw_payload["systems"], field_name="systems"))
    routes = tuple(_require_string_list(raw_payload["routes"], field_name="routes"))

    quantization_payload = _require_mapping(raw_payload["quantization"], field_name="quantization")
    decode_payload = _require_mapping(raw_payload["decode"], field_name="decode")
    metrics_payload = _require_mapping(raw_payload["metrics"], field_name="metrics")
    decision_payload = _require_mapping(raw_payload["decision_policy"], field_name="decision_policy")

    return BenchmarkConfig(
        artifacts_root=artifacts_root,
        results_root=results_root,
        systems=systems,
        routes=routes,
        quantization=QuantizationConfig(
            weight_type=_require_string(quantization_payload["weight_type"], field_name="quantization.weight_type")
        ),
        decode=DecodeConfig(
            batch_size=_require_int(decode_payload["batch_size"], field_name="decode.batch_size"),
            do_sample=_require_bool(decode_payload["do_sample"], field_name="decode.do_sample"),
            num_beams=_require_int(decode_payload["num_beams"], field_name="decode.num_beams"),
            greedy_decode=_require_bool(decode_payload["greedy_decode"], field_name="decode.greedy_decode"),
            max_new_tokens=_require_int(decode_payload["max_new_tokens"], field_name="decode.max_new_tokens"),
            warmup_iterations=_require_int(
                decode_payload["warmup_iterations"],
                field_name="decode.warmup_iterations",
            ),
            benchmark_iterations=_require_int(
                decode_payload["benchmark_iterations"],
                field_name="decode.benchmark_iterations",
            ),
        ),
        metrics=MetricsConfig(
            comet_model=_require_string(metrics_payload["comet_model"], field_name="metrics.comet_model"),
            comet_batch_size=_require_int(
                metrics_payload["comet_batch_size"],
                field_name="metrics.comet_batch_size",
            ),
            compute_bleu=_require_bool(metrics_payload["compute_bleu"], field_name="metrics.compute_bleu"),
            compute_chrf_pp=_require_bool(
                metrics_payload["compute_chrf_pp"],
                field_name="metrics.compute_chrf_pp",
            ),
        ),
        decision_policy=DecisionPolicy(
            must_preserve_threshold=_require_float(
                decision_payload["must_preserve_threshold"],
                field_name="decision_policy.must_preserve_threshold",
            ),
            comet_delta_threshold=_require_float(
                decision_payload["comet_delta_threshold"],
                field_name="decision_policy.comet_delta_threshold",
            ),
            latency_tie_threshold=_require_float(
                decision_payload["latency_tie_threshold"],
                field_name="decision_policy.latency_tie_threshold",
            ),
        ),
    )


def _require_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")
    normalized_values: list[str] = []
    for item in value:
        normalized_values.append(_require_string(item, field_name=field_name))
    return normalized_values


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool.")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int.")
    return value


def _require_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)

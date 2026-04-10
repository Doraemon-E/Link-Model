from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "pipelines.yaml"


@dataclass(frozen=True)
class SharedPathsConfig:
    translation_models_root: Path
    translation_results_root: Path
    translation_catalog_output: Path
    speech_models_root: Path
    speech_catalog_output: Path
    corpus_path: Path


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    model_id: str
    family: str
    source_langs: tuple[str, ...]
    target_langs: tuple[str, ...]
    artifact_format: str = "onnx"
    runtime_backend: str = "onnxruntime"
    quantization_format: str = "onnx_qint8"
    quantization_source: str = "local_prepare"
    package_enabled: bool = False
    package_source: str | None = None
    package_target: str | None = None

    @property
    def package_id(self) -> str:
        return f"{self.artifact_id}-onnx"

    @property
    def archive_file_name(self) -> str:
        return f"{self.artifact_id}-onnx-int8.zip"


@dataclass(frozen=True)
class RoutePlanSpec:
    route_id: str
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    display_name: str
    strategy: str
    route_plans: dict[str, RoutePlanSpec]
    baseline: bool = False
    lane: str = "translation"


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    source_lang: str
    target_lang: str
    reference_key: str


@dataclass(frozen=True)
class QuantizationConfig:
    weight_type: str


@dataclass(frozen=True)
class DecodeConfig:
    batch_size: int
    do_sample: bool
    num_beams: int
    greedy_decode: bool
    max_new_tokens: int
    warmup_iterations: int
    benchmark_iterations: int


@dataclass(frozen=True)
class MetricsConfig:
    comet_model: str
    comet_batch_size: int
    compute_bleu: bool
    compute_chrf_pp: bool


@dataclass(frozen=True)
class DecisionPolicy:
    must_preserve_threshold: float
    comet_delta_threshold: float
    latency_tie_threshold: float


@dataclass(frozen=True)
class TranslationBenchmarkConfig:
    routes: dict[str, RouteSpec]
    selected_systems: tuple[str, ...]
    selected_routes: tuple[str, ...]
    quantization: QuantizationConfig
    decode: DecodeConfig
    metrics: MetricsConfig
    decision_policy: DecisionPolicy


@dataclass(frozen=True)
class TranslationPackageConfig:
    package_version: str
    min_app_version: str
    archive_base_url: str


@dataclass(frozen=True)
class TranslationReportConfig:
    title: str


@dataclass(frozen=True)
class TranslationConfig:
    artifacts: dict[str, ArtifactSpec]
    systems: dict[str, SystemSpec]
    benchmark: TranslationBenchmarkConfig
    package: TranslationPackageConfig
    report: TranslationReportConfig


@dataclass(frozen=True)
class SpeechArtifactSpec:
    package_id: str
    repo_id: str
    source_file_name: str
    local_file_name: str
    family: str = "whisper"

    @property
    def archive_file_name(self) -> str:
        return f"{self.package_id}.zip"


@dataclass(frozen=True)
class SpeechPackageConfig:
    package_version: str
    min_app_version: str
    archive_base_url: str


@dataclass(frozen=True)
class SpeechConfig:
    artifacts: dict[str, SpeechArtifactSpec]
    package: SpeechPackageConfig


@dataclass(frozen=True)
class RootConfig:
    shared_paths: SharedPathsConfig
    translation: TranslationConfig
    speech: SpeechConfig

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonify(self)


def load_config(config_path: Path | None = None) -> RootConfig:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(f"Config must be a mapping: {resolved_path}")

    shared_payload = _require_mapping(raw_payload.get("shared"), field_name="shared")
    shared_paths_payload = _require_mapping(shared_payload.get("paths"), field_name="shared.paths")

    translation_payload = _require_mapping(raw_payload.get("translation"), field_name="translation")
    speech_payload = _require_mapping(raw_payload.get("speech"), field_name="speech")

    config = RootConfig(
        shared_paths=SharedPathsConfig(
            translation_models_root=_resolve_repo_path(
                _require_string(shared_paths_payload.get("translation_models_root"), field_name="shared.paths.translation_models_root")
            ),
            translation_results_root=_resolve_repo_path(
                _require_string(shared_paths_payload.get("translation_results_root"), field_name="shared.paths.translation_results_root")
            ),
            translation_catalog_output=_resolve_repo_path(
                _require_string(
                    shared_paths_payload.get("translation_catalog_output"),
                    field_name="shared.paths.translation_catalog_output",
                )
            ),
            speech_models_root=_resolve_repo_path(
                _require_string(shared_paths_payload.get("speech_models_root"), field_name="shared.paths.speech_models_root")
            ),
            speech_catalog_output=_resolve_repo_path(
                _require_string(
                    shared_paths_payload.get("speech_catalog_output"),
                    field_name="shared.paths.speech_catalog_output",
                )
            ),
            corpus_path=_resolve_repo_path(
                _require_string(shared_paths_payload.get("corpus_path"), field_name="shared.paths.corpus_path")
            ),
        ),
        translation=_load_translation_config(translation_payload),
        speech=_load_speech_config(speech_payload),
    )
    _validate_config(config)
    return config


def _load_translation_config(payload: dict[str, object]) -> TranslationConfig:
    artifacts_payload = _require_mapping(payload.get("artifacts"), field_name="translation.artifacts")
    systems_payload = _require_mapping(payload.get("systems"), field_name="translation.systems")
    benchmark_payload = _require_mapping(payload.get("benchmark"), field_name="translation.benchmark")
    package_payload = _require_mapping(payload.get("package"), field_name="translation.package")
    report_payload = _require_mapping(payload.get("report"), field_name="translation.report")

    artifacts = {
        artifact_id: ArtifactSpec(
            artifact_id=artifact_id,
            model_id=_require_string(spec_payload.get("model_id"), field_name=f"translation.artifacts.{artifact_id}.model_id"),
            family=_require_string(spec_payload.get("family"), field_name=f"translation.artifacts.{artifact_id}.family"),
            source_langs=tuple(_require_string_list(spec_payload.get("source_langs"), field_name=f"translation.artifacts.{artifact_id}.source_langs")),
            target_langs=tuple(_require_string_list(spec_payload.get("target_langs"), field_name=f"translation.artifacts.{artifact_id}.target_langs")),
            artifact_format=_optional_string(spec_payload.get("artifact_format"), default="onnx"),
            runtime_backend=_optional_string(spec_payload.get("runtime_backend"), default="onnxruntime"),
            quantization_format=_optional_string(spec_payload.get("quantization_format"), default="onnx_qint8"),
            quantization_source=_optional_string(spec_payload.get("quantization_source"), default="local_prepare"),
            package_enabled=_optional_bool(spec_payload.get("package_enabled"), default=False),
            package_source=_optional_string(spec_payload.get("package_source"), default=None),
            package_target=_optional_string(spec_payload.get("package_target"), default=None),
        )
        for artifact_id, spec_payload in _iter_mapping_payloads(artifacts_payload, field_name="translation.artifacts")
    }

    routes_payload = _require_mapping(benchmark_payload.get("routes"), field_name="translation.benchmark.routes")
    routes = {
        route_id: RouteSpec(
            route_id=route_id,
            source_lang=_require_string(spec_payload.get("source_lang"), field_name=f"translation.benchmark.routes.{route_id}.source_lang"),
            target_lang=_require_string(spec_payload.get("target_lang"), field_name=f"translation.benchmark.routes.{route_id}.target_lang"),
            reference_key=_require_string(spec_payload.get("reference_key"), field_name=f"translation.benchmark.routes.{route_id}.reference_key"),
        )
        for route_id, spec_payload in _iter_mapping_payloads(routes_payload, field_name="translation.benchmark.routes")
    }

    systems = {
        system_id: SystemSpec(
            system_id=system_id,
            display_name=_require_string(spec_payload.get("display_name"), field_name=f"translation.systems.{system_id}.display_name"),
            strategy=_require_string(spec_payload.get("strategy"), field_name=f"translation.systems.{system_id}.strategy"),
            baseline=_optional_bool(spec_payload.get("baseline"), default=False),
            lane=_optional_string(spec_payload.get("lane"), default="translation") or "translation",
            route_plans={
                route_id: RoutePlanSpec(
                    route_id=route_id,
                    artifact_ids=tuple(
                        _require_string_list(
                            plan_payload.get("artifact_ids"),
                            field_name=f"translation.systems.{system_id}.route_plans.{route_id}.artifact_ids",
                        )
                    ),
                )
                for route_id, plan_payload in _iter_mapping_payloads(
                    _require_mapping(spec_payload.get("route_plans"), field_name=f"translation.systems.{system_id}.route_plans"),
                    field_name=f"translation.systems.{system_id}.route_plans",
                )
            },
        )
        for system_id, spec_payload in _iter_mapping_payloads(systems_payload, field_name="translation.systems")
    }

    quantization_payload = _require_mapping(benchmark_payload.get("quantization"), field_name="translation.benchmark.quantization")
    decode_payload = _require_mapping(benchmark_payload.get("decode"), field_name="translation.benchmark.decode")
    metrics_payload = _require_mapping(benchmark_payload.get("metrics"), field_name="translation.benchmark.metrics")
    decision_payload = _require_mapping(benchmark_payload.get("decision_policy"), field_name="translation.benchmark.decision_policy")

    return TranslationConfig(
        artifacts=artifacts,
        systems=systems,
        benchmark=TranslationBenchmarkConfig(
            routes=routes,
            selected_systems=tuple(
                _require_string_list(benchmark_payload.get("selected_systems"), field_name="translation.benchmark.selected_systems")
            ),
            selected_routes=tuple(
                _require_string_list(benchmark_payload.get("selected_routes"), field_name="translation.benchmark.selected_routes")
            ),
            quantization=QuantizationConfig(
                weight_type=_require_string(quantization_payload.get("weight_type"), field_name="translation.benchmark.quantization.weight_type")
            ),
            decode=DecodeConfig(
                batch_size=_require_int(decode_payload.get("batch_size"), field_name="translation.benchmark.decode.batch_size"),
                do_sample=_require_bool(decode_payload.get("do_sample"), field_name="translation.benchmark.decode.do_sample"),
                num_beams=_require_int(decode_payload.get("num_beams"), field_name="translation.benchmark.decode.num_beams"),
                greedy_decode=_require_bool(decode_payload.get("greedy_decode"), field_name="translation.benchmark.decode.greedy_decode"),
                max_new_tokens=_require_int(decode_payload.get("max_new_tokens"), field_name="translation.benchmark.decode.max_new_tokens"),
                warmup_iterations=_require_int(
                    decode_payload.get("warmup_iterations"),
                    field_name="translation.benchmark.decode.warmup_iterations",
                ),
                benchmark_iterations=_require_int(
                    decode_payload.get("benchmark_iterations"),
                    field_name="translation.benchmark.decode.benchmark_iterations",
                ),
            ),
            metrics=MetricsConfig(
                comet_model=_require_string(metrics_payload.get("comet_model"), field_name="translation.benchmark.metrics.comet_model"),
                comet_batch_size=_require_int(
                    metrics_payload.get("comet_batch_size"),
                    field_name="translation.benchmark.metrics.comet_batch_size",
                ),
                compute_bleu=_require_bool(metrics_payload.get("compute_bleu"), field_name="translation.benchmark.metrics.compute_bleu"),
                compute_chrf_pp=_require_bool(
                    metrics_payload.get("compute_chrf_pp"),
                    field_name="translation.benchmark.metrics.compute_chrf_pp",
                ),
            ),
            decision_policy=DecisionPolicy(
                must_preserve_threshold=_require_float(
                    decision_payload.get("must_preserve_threshold"),
                    field_name="translation.benchmark.decision_policy.must_preserve_threshold",
                ),
                comet_delta_threshold=_require_float(
                    decision_payload.get("comet_delta_threshold"),
                    field_name="translation.benchmark.decision_policy.comet_delta_threshold",
                ),
                latency_tie_threshold=_require_float(
                    decision_payload.get("latency_tie_threshold"),
                    field_name="translation.benchmark.decision_policy.latency_tie_threshold",
                ),
            ),
        ),
        package=TranslationPackageConfig(
            package_version=_require_string(package_payload.get("package_version"), field_name="translation.package.package_version"),
            min_app_version=_require_string(package_payload.get("min_app_version"), field_name="translation.package.min_app_version"),
            archive_base_url=_require_string(package_payload.get("archive_base_url"), field_name="translation.package.archive_base_url"),
        ),
        report=TranslationReportConfig(
            title=_require_string(report_payload.get("title"), field_name="translation.report.title"),
        ),
    )


def _load_speech_config(payload: dict[str, object]) -> SpeechConfig:
    artifacts_payload = _require_mapping(payload.get("artifacts"), field_name="speech.artifacts")
    package_payload = _require_mapping(payload.get("package"), field_name="speech.package")

    artifacts = {
        package_id: SpeechArtifactSpec(
            package_id=package_id,
            repo_id=_require_string(spec_payload.get("repo_id"), field_name=f"speech.artifacts.{package_id}.repo_id"),
            source_file_name=_require_string(
                spec_payload.get("source_file_name"),
                field_name=f"speech.artifacts.{package_id}.source_file_name",
            ),
            local_file_name=_require_string(
                spec_payload.get("local_file_name"),
                field_name=f"speech.artifacts.{package_id}.local_file_name",
            ),
            family=_optional_string(spec_payload.get("family"), default="whisper") or "whisper",
        )
        for package_id, spec_payload in _iter_mapping_payloads(artifacts_payload, field_name="speech.artifacts")
    }

    return SpeechConfig(
        artifacts=artifacts,
        package=SpeechPackageConfig(
            package_version=_require_string(package_payload.get("package_version"), field_name="speech.package.package_version"),
            min_app_version=_require_string(package_payload.get("min_app_version"), field_name="speech.package.min_app_version"),
            archive_base_url=_require_string(package_payload.get("archive_base_url"), field_name="speech.package.archive_base_url"),
        ),
    )


def _validate_config(config: RootConfig) -> None:
    translation = config.translation

    unknown_selected_systems = sorted(set(translation.benchmark.selected_systems) - set(translation.systems))
    if unknown_selected_systems:
        raise ValueError(f"Unknown selected translation systems: {unknown_selected_systems}")

    unknown_selected_routes = sorted(set(translation.benchmark.selected_routes) - set(translation.benchmark.routes))
    if unknown_selected_routes:
        raise ValueError(f"Unknown selected translation routes: {unknown_selected_routes}")

    if not any(system.baseline for system in translation.systems.values()):
        raise ValueError("Translation config must define at least one baseline system.")

    for system in translation.systems.values():
        if system.strategy not in {"direct", "pivot_via_en"}:
            raise ValueError(f"Unsupported strategy for {system.system_id}: {system.strategy}")
        for route_id, route_plan in system.route_plans.items():
            if route_id not in translation.benchmark.routes:
                raise ValueError(f"Unknown route {route_id} in system {system.system_id}")
            unknown_artifacts = sorted(set(route_plan.artifact_ids) - set(translation.artifacts))
            if unknown_artifacts:
                raise ValueError(f"Unknown artifacts for {system.system_id}/{route_id}: {unknown_artifacts}")
            if system.strategy == "pivot_via_en" and len(route_plan.artifact_ids) != 2:
                raise ValueError(f"Pivot route {system.system_id}/{route_id} must reference exactly two artifacts.")
            if system.strategy == "direct" and len(route_plan.artifact_ids) != 1:
                raise ValueError(f"Direct route {system.system_id}/{route_id} must reference exactly one artifact.")


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _require_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _iter_mapping_payloads(payload: dict[str, object], *, field_name: str):
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} must use non-empty string keys.")
        yield key, _require_mapping(value, field_name=f"{field_name}.{key}")


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _optional_string(value: object, *, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Optional string values must be non-empty strings when provided.")
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


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("Optional bool values must be bools when provided.")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int.")
    return value


def _require_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value

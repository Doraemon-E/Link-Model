from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    model_id: str
    family: str
    source_langs: tuple[str, ...]
    target_langs: tuple[str, ...]


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    source_lang: str
    target_lang: str
    reference_key: str


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    lane: str
    artifact_ids: tuple[str, ...]
    executor_module: str
    baseline: bool = False


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
class BenchmarkConfig:
    artifacts_root: Path
    results_root: Path
    systems: tuple[str, ...]
    routes: tuple[str, ...]
    quantization: QuantizationConfig
    decode: DecodeConfig
    metrics: MetricsConfig
    decision_policy: DecisionPolicy

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts_root"] = self.artifacts_root.as_posix()
        payload["results_root"] = self.results_root.as_posix()
        return payload


@dataclass(frozen=True)
class CorpusExpectedResult:
    reference: str
    must_preserve: tuple[str, ...]
    acceptance_note: str


@dataclass(frozen=True)
class CorpusEntry:
    entry_id: str
    bucket: str
    source_text: str
    char_count: int
    scenario_tag: str
    expected_by_route: dict[str, CorpusExpectedResult]

    def expected_for_route(self, route: RouteSpec) -> CorpusExpectedResult:
        return self.expected_by_route[route.route_id]


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    model_id: str
    revision: str
    license: str
    source_langs: list[str]
    target_langs: list[str]
    fp32_size: int
    int8_size: int
    quantization_ratio: float
    export_success: bool
    quantize_success: bool
    error_message: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HopDetail:
    hop: str
    source_lang: str
    target_lang: str
    input_text: str
    output_text: str
    latency_ms: float
    output_token_count: int
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionRecord:
    system_id: str
    lane: str
    route: str
    iteration: int
    entry_id: str
    bucket: str
    source_text: str
    translated_text: str
    reference_text: str
    must_preserve: list[str]
    sentence_latency_ms: float
    output_token_count: int
    error: str | None
    hop_details: list[HopDetail] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hop_details"] = [hop.to_json_dict() for hop in self.hop_details]
        return payload


@dataclass(frozen=True)
class RuntimeSummary:
    system_id: str
    lane: str
    route: str
    artifact_ids: list[str]
    model_ids: list[str]
    licenses: list[str]
    int8_size: int
    cold_start_ms: float
    p50_ms: float
    p95_ms: float
    total_duration_s: float
    tokens_per_second: float
    peak_rss_mb: float
    empty_output_count: int
    error_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRecord:
    system_id: str
    lane: str
    route: str
    artifact_ids: list[str]
    model_ids: list[str]
    licenses: list[str]
    int8_size: int
    cold_start_ms: float
    p50_ms: float
    p95_ms: float
    total_duration_s: float
    tokens_per_second: float
    peak_rss_mb: float
    empty_output_count: int
    error_count: int
    comet: float | None
    chrf_pp: float
    bleu: float
    must_preserve_rate: float
    eliminated: bool
    elimination_reasons: list[str]
    recommended: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeaderboardRow:
    board_type: str
    lane: str
    route: str
    rank: int
    system_id: str
    comet: float
    p50_ms: float
    peak_rss_mb: float
    int8_size: int
    recommended: bool
    eliminated: bool

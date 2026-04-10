from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def expected_for_route(self, route_id: str) -> CorpusExpectedResult:
        return self.expected_by_route[route_id]


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    model_id: str
    revision: str
    license: str
    source_langs: list[str]
    target_langs: list[str]
    fp32_size: int | None
    quantized_size: int
    quantization_ratio: float | None
    export_success: bool
    quantize_success: bool
    quantization_format: str = "onnx_qint8"
    quantization_source: str = "local_prepare"
    source_file_name: str | None = None
    source_file_sha256: str | None = None
    error_message: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ArtifactManifest":
        normalized = dict(payload)
        if "quantized_size" not in normalized and "int8_size" in normalized:
            normalized["quantized_size"] = normalized.pop("int8_size")
        normalized.setdefault("fp32_size", None)
        normalized.setdefault("quantization_ratio", None)
        normalized.setdefault("quantization_format", "onnx_qint8")
        normalized.setdefault("quantization_source", "local_prepare")
        normalized.setdefault("source_file_name", None)
        normalized.setdefault("source_file_sha256", None)
        return cls(**normalized)


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

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "HopDetail":
        return cls(**payload)


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
    display_name: str = ""
    strategy: str = "direct"
    runtime_backend: str = ""
    hop_details: list[HopDetail] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hop_details"] = [hop.to_json_dict() for hop in self.hop_details]
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "PredictionRecord":
        normalized = dict(payload)
        normalized.setdefault("display_name", "")
        normalized.setdefault("strategy", "direct")
        normalized.setdefault("runtime_backend", "")
        normalized["hop_details"] = [HopDetail.from_json_dict(item) for item in normalized.get("hop_details", [])]
        return cls(**normalized)


@dataclass(frozen=True)
class RuntimeSummary:
    system_id: str
    lane: str
    route: str
    artifact_ids: list[str]
    model_ids: list[str]
    licenses: list[str]
    quantized_size: int
    cold_start_ms: float
    p50_ms: float
    p95_ms: float
    total_duration_s: float
    tokens_per_second: float
    peak_rss_mb: float
    empty_output_count: int
    error_count: int
    display_name: str = ""
    strategy: str = "direct"
    runtime_backend: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "RuntimeSummary":
        normalized = dict(payload)
        if "quantized_size" not in normalized and "int8_size" in normalized:
            normalized["quantized_size"] = normalized.pop("int8_size")
        normalized.setdefault("display_name", "")
        normalized.setdefault("strategy", "direct")
        normalized.setdefault("runtime_backend", "")
        return cls(**normalized)


@dataclass(frozen=True)
class GateCheck:
    value: float | int | None
    threshold: float | int | str | None
    passed: bool | None
    reason: str

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
    quantized_size: int
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
    must_preserve_hits: int
    must_preserve_total: int
    display_name: str = ""
    strategy: str = "direct"
    runtime_backend: str = ""
    eliminated: bool = False
    elimination_reasons: list[str] = field(default_factory=list)
    recommended: bool = False
    gate_checks: dict[str, GateCheck] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_checks"] = {key: gate.to_json_dict() for key, gate in self.gate_checks.items()}
        return payload


@dataclass(frozen=True)
class LeaderboardRow:
    board_type: str
    route: str
    rank: int
    system_id: str
    display_name: str
    strategy: str
    comet: float | None
    bleu: float
    chrf_pp: float
    must_preserve_rate: float
    must_preserve_hits: int
    must_preserve_total: int
    p50_ms: float
    peak_rss_mb: float
    quantized_size: int
    recommended: bool
    eliminated: bool
    status: str

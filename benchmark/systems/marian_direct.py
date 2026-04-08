from __future__ import annotations

import time

from ..paths import artifact_manifest_path, artifact_stage_directory
from ..prepare import load_artifact_manifest
from ..runtime import MemorySampler, TranslationRuntime, percentile
from ..schemas import BenchmarkConfig, PredictionRecord, RouteSpec, RuntimeSummary, SystemSpec


def run_system(
    config: BenchmarkConfig,
    system: SystemSpec,
    route: RouteSpec,
    corpus_entries,
) -> tuple[list[PredictionRecord], RuntimeSummary]:
    artifact_id = "marian-zh-en" if route.route_id == "zh-en" else "marian-zh-ja"
    loaded_runtime = TranslationRuntime.load(
        artifact_stage_directory(config.artifacts_root, "quantized", artifact_id)
    )
    runtime = loaded_runtime.runtime
    cold_start_ms = loaded_runtime.cold_start_ms

    _warmup(config, runtime, corpus_entries[0].source_text)

    predictions: list[PredictionRecord] = []
    latencies_ms: list[float] = []
    total_output_tokens = 0
    total_inference_seconds = 0.0
    empty_output_count = 0
    error_count = 0

    sampler = MemorySampler()
    sampler.start()
    started_at = time.perf_counter()

    for iteration in range(1, config.decode.benchmark_iterations + 1):
        for entry in corpus_entries:
            reference = entry.expected_for_route(route)
            call_started_at = time.perf_counter()
            translated_text = ""
            error_message: str | None = None
            output_token_count = 0

            try:
                result = runtime.translate(
                    entry.source_text,
                    source_lang=None,
                    target_lang=None,
                    batch_size=config.decode.batch_size,
                    do_sample=config.decode.do_sample,
                    num_beams=config.decode.num_beams,
                    max_new_tokens=config.decode.max_new_tokens,
                )
                translated_text = result.text
                output_token_count = result.output_token_count
            except Exception as exc:
                error_message = str(exc)
                error_count += 1

            sentence_latency_ms = (time.perf_counter() - call_started_at) * 1000
            total_inference_seconds += sentence_latency_ms / 1000
            latencies_ms.append(sentence_latency_ms)
            total_output_tokens += output_token_count

            if not translated_text.strip():
                empty_output_count += 1

            predictions.append(
                PredictionRecord(
                    system_id=system.system_id,
                    lane=system.lane,
                    route=route.route_id,
                    iteration=iteration,
                    entry_id=entry.entry_id,
                    bucket=entry.bucket,
                    source_text=entry.source_text,
                    translated_text=translated_text,
                    reference_text=reference.reference,
                    must_preserve=list(reference.must_preserve),
                    sentence_latency_ms=sentence_latency_ms,
                    output_token_count=output_token_count,
                    error=error_message,
                    hop_details=[],
                )
            )

    total_duration_s = time.perf_counter() - started_at
    peak_rss_mb = sampler.stop()
    manifest = load_artifact_manifest(artifact_manifest_path(config.artifacts_root, artifact_id))

    summary = RuntimeSummary(
        system_id=system.system_id,
        lane=system.lane,
        route=route.route_id,
        artifact_ids=[artifact_id],
        model_ids=[manifest.model_id],
        licenses=[manifest.license],
        int8_size=manifest.int8_size,
        cold_start_ms=cold_start_ms,
        p50_ms=percentile(latencies_ms, 0.5),
        p95_ms=percentile(latencies_ms, 0.95),
        total_duration_s=total_duration_s,
        tokens_per_second=(float(total_output_tokens) / total_inference_seconds) if total_inference_seconds else 0.0,
        peak_rss_mb=peak_rss_mb,
        empty_output_count=empty_output_count,
        error_count=error_count,
    )
    return predictions, summary


def _warmup(config: BenchmarkConfig, runtime: TranslationRuntime, text: str) -> None:
    for _ in range(config.decode.warmup_iterations):
        runtime.translate(
            text,
            source_lang=None,
            target_lang=None,
            batch_size=config.decode.batch_size,
            do_sample=config.decode.do_sample,
            num_beams=config.decode.num_beams,
            max_new_tokens=config.decode.max_new_tokens,
        )

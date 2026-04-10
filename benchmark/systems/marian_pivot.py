from __future__ import annotations

import time
from pathlib import Path

from ..paths import artifact_stage_directory, artifact_manifest_path
from ..prepare import load_artifact_manifest
from ..registry import artifact_for_id
from ..runtime import MemorySampler, TranslationRuntime, percentile
from ..schemas import BenchmarkConfig, HopDetail, PredictionRecord, RouteSpec, RuntimeSummary, SystemSpec


def run_system(
    config: BenchmarkConfig,
    system: SystemSpec,
    route: RouteSpec,
    corpus_entries,
) -> tuple[list[PredictionRecord], RuntimeSummary]:
    zh_en_runtime = TranslationRuntime.load(
        artifact_stage_directory(config.artifacts_root, "quantized", "marian-zh-en")
    )
    cold_start_ms = zh_en_runtime.cold_start_ms
    runtimes = {"zh-en": zh_en_runtime.runtime}
    artifact_ids = ["marian-zh-en"]

    if route.route_id == "zh-ja":
        en_ja_runtime = TranslationRuntime.load(
            artifact_stage_directory(config.artifacts_root, "quantized", "marian-en-ja")
        )
        cold_start_ms += en_ja_runtime.cold_start_ms
        runtimes["en-ja"] = en_ja_runtime.runtime
        artifact_ids.append("marian-en-ja")

    _warmup(config, route, runtimes, corpus_entries[0].source_text)

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
            error_message: str | None = None
            translated_text = ""
            hop_details: list[HopDetail] = []
            output_token_count = 0

            try:
                first_hop_started_at = time.perf_counter()
                first_hop_result = runtimes["zh-en"].translate(
                    entry.source_text,
                    source_lang=None,
                    target_lang=None,
                    batch_size=config.decode.batch_size,
                    do_sample=config.decode.do_sample,
                    num_beams=config.decode.num_beams,
                    max_new_tokens=config.decode.max_new_tokens,
                )
                first_hop_latency_ms = (time.perf_counter() - first_hop_started_at) * 1000

                if route.route_id == "zh-en":
                    translated_text = first_hop_result.text
                    output_token_count = first_hop_result.output_token_count
                else:
                    hop_details.append(
                        HopDetail(
                            hop="hop1",
                            source_lang="zh",
                            target_lang="en",
                            input_text=entry.source_text,
                            output_text=first_hop_result.text,
                            latency_ms=first_hop_latency_ms,
                            output_token_count=first_hop_result.output_token_count,
                        )
                    )

                    second_hop_started_at = time.perf_counter()
                    second_hop_result = runtimes["en-ja"].translate(
                        first_hop_result.text,
                        source_lang=None,
                        target_lang=None,
                        batch_size=config.decode.batch_size,
                        do_sample=config.decode.do_sample,
                        num_beams=config.decode.num_beams,
                        max_new_tokens=config.decode.max_new_tokens,
                    )
                    second_hop_latency_ms = (time.perf_counter() - second_hop_started_at) * 1000
                    hop_details.append(
                        HopDetail(
                            hop="hop2",
                            source_lang="en",
                            target_lang="ja",
                            input_text=first_hop_result.text,
                            output_text=second_hop_result.text,
                            latency_ms=second_hop_latency_ms,
                            output_token_count=second_hop_result.output_token_count,
                        )
                    )

                    translated_text = second_hop_result.text
                    output_token_count = (
                        first_hop_result.output_token_count + second_hop_result.output_token_count
                    )
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
                    hop_details=hop_details,
                )
            )

    total_duration_s = time.perf_counter() - started_at
    peak_rss_mb = sampler.stop()
    manifests = [load_artifact_manifest(artifact_manifest_path(config.artifacts_root, artifact_id)) for artifact_id in artifact_ids]

    summary = RuntimeSummary(
        system_id=system.system_id,
        lane=system.lane,
        route=route.route_id,
        artifact_ids=artifact_ids,
        model_ids=[manifest.model_id for manifest in manifests],
        licenses=[manifest.license for manifest in manifests],
        quantized_size=sum(manifest.quantized_size for manifest in manifests),
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


def _warmup(config: BenchmarkConfig, route: RouteSpec, runtimes: dict[str, TranslationRuntime], text: str) -> None:
    for _ in range(config.decode.warmup_iterations):
        first_hop_result = runtimes["zh-en"].translate(
            text,
            source_lang=None,
            target_lang=None,
            batch_size=config.decode.batch_size,
            do_sample=config.decode.do_sample,
            num_beams=config.decode.num_beams,
            max_new_tokens=config.decode.max_new_tokens,
        )
        if route.route_id == "zh-ja":
            runtimes["en-ja"].translate(
                first_hop_result.text,
                source_lang=None,
                target_lang=None,
                batch_size=config.decode.batch_size,
                do_sample=config.decode.do_sample,
                num_beams=config.decode.num_beams,
                max_new_tokens=config.decode.max_new_tokens,
            )

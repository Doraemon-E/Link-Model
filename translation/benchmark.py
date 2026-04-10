from __future__ import annotations

import json
import time
from pathlib import Path

from shared.config import ArtifactSpec, RootConfig, RoutePlanSpec, RouteSpec, SystemSpec
from shared.files import ensure_directory

from .corpus import load_corpus
from .prepare import load_existing_manifest
from .runtime import GGUFTranslationRuntime, LoadedRuntime, MemorySampler, TranslationRuntime, percentile
from .schemas import HopDetail, PredictionRecord, RuntimeSummary
from .storage import (
    artifact_manifest_path,
    has_quantized_payload,
    migrate_legacy_translation_assets,
    translation_stage_directory,
)


def run_benchmark(config: RootConfig, *, timestamp: str | None = None) -> Path:
    migrate_legacy_translation_assets(config)
    corpus_entries = load_corpus(config)
    routes = selected_routes(config)
    systems = selected_systems(config)
    result_dir = new_result_directory(config.shared_paths.translation_results_root, timestamp=timestamp)
    predictions: list[PredictionRecord] = []
    runtime_summaries: list[RuntimeSummary] = []
    smoke_entry = select_smoke_entry(corpus_entries)

    for system in systems:
        supported_routes = [route for route in routes if route.route_id in system.route_plans]
        readiness = system_readiness(config, system, supported_routes)
        if readiness is not None:
            print(f"[translation benchmark] skip system={system.system_id}: {readiness}")
            continue

        smoke_test_system(config, system, supported_routes, smoke_entry)
        for route in supported_routes:
            print(f"[translation benchmark] system={system.system_id} route={route.route_id}")
            route_predictions, runtime_summary = run_system_route(config, system, route, corpus_entries)
            predictions.extend(route_predictions)
            runtime_summaries.append(runtime_summary)

    predictions_path = result_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(prediction.to_json_dict(), ensure_ascii=False) + "\n")

    (result_dir / "runtime-summary.json").write_text(
        json.dumps([summary.to_json_dict() for summary in runtime_summaries], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (result_dir / "config-snapshot.json").write_text(
        json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[translation benchmark] wrote predictions to {predictions_path}")
    return result_dir


def selected_routes(config: RootConfig) -> list[RouteSpec]:
    return [config.translation.benchmark.routes[route_id] for route_id in config.translation.benchmark.selected_routes]


def selected_systems(config: RootConfig) -> list[SystemSpec]:
    return [config.translation.systems[system_id] for system_id in config.translation.benchmark.selected_systems]


def new_result_directory(results_root: Path, *, timestamp: str | None) -> Path:
    from datetime import datetime, timezone

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(results_root / timestamp)


def latest_result_directory(results_root: Path) -> Path:
    ensure_directory(results_root)
    candidates = [path for path in results_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No benchmark results found under {results_root}")
    return sorted(candidates)[-1]


def resolve_result_dir(config: RootConfig, result_dir: Path | None) -> Path:
    if result_dir is not None:
        return result_dir
    return latest_result_directory(config.shared_paths.translation_results_root)


def run_system_route(
    config: RootConfig,
    system: SystemSpec,
    route: RouteSpec,
    corpus_entries,
) -> tuple[list[PredictionRecord], RuntimeSummary]:
    route_plan = system.route_plans[route.route_id]
    if system.strategy == "pivot_via_en":
        return _run_pivot_route(config, system, route, route_plan, corpus_entries)
    return _run_direct_route(config, system, route, route_plan, corpus_entries)


def system_readiness(config: RootConfig, system: SystemSpec, routes: list[RouteSpec]) -> str | None:
    for route in routes:
        for artifact_id in system.route_plans[route.route_id].artifact_ids:
            artifact = config.translation.artifacts[artifact_id]
            manifest_path = artifact_manifest_path(config, artifact_id)
            if not manifest_path.exists():
                return f"missing artifact manifest for {artifact_id}; run translation prepare first"
            manifest = load_existing_manifest(config, artifact_id)
            if not manifest.quantize_success:
                return f"artifact {artifact_id} is not ready ({manifest.error_message or 'prepare failed'})"
            export_dir = translation_stage_directory(config, "exported", artifact_id)
            quantized_dir = translation_stage_directory(config, "quantized", artifact_id)
            if not has_quantized_payload(artifact, export_dir, quantized_dir):
                return f"artifact {artifact_id} is incomplete under {quantized_dir}; rerun translation prepare"
    return None


def smoke_test_system(config: RootConfig, system: SystemSpec, routes: list[RouteSpec], corpus_entry) -> None:
    for route in routes:
        route_plan = system.route_plans[route.route_id]
        if system.strategy == "pivot_via_en":
            _smoke_test_pivot_route(config, route, route_plan, corpus_entry)
        else:
            _smoke_test_direct_route(config, route, route_plan, corpus_entry)


def select_smoke_entry(corpus_entries):
    for entry in corpus_entries:
        if entry.bucket == "short":
            return entry
    if not corpus_entries:
        raise ValueError("Corpus is empty.")
    return corpus_entries[0]


def _run_direct_route(
    config: RootConfig,
    system: SystemSpec,
    route: RouteSpec,
    route_plan: RoutePlanSpec,
    corpus_entries,
) -> tuple[list[PredictionRecord], RuntimeSummary]:
    artifact = config.translation.artifacts[route_plan.artifact_ids[0]]
    loaded_runtime = _load_runtime(config, artifact)
    runtime = loaded_runtime.runtime
    cold_start_ms = loaded_runtime.cold_start_ms
    _warmup_direct(config, runtime, artifact, route, corpus_entries[0].source_text)

    predictions: list[PredictionRecord] = []
    latencies_ms: list[float] = []
    total_output_tokens = 0
    total_inference_seconds = 0.0
    empty_output_count = 0
    error_count = 0

    sampler = MemorySampler()
    sampler.start()
    started_at = time.perf_counter()
    for iteration in range(1, config.translation.benchmark.decode.benchmark_iterations + 1):
        for entry in corpus_entries:
            reference = entry.expected_for_route(route.route_id)
            call_started_at = time.perf_counter()
            translated_text = ""
            error_message: str | None = None
            output_token_count = 0

            try:
                result = _translate_once(config, runtime, artifact, route, entry.source_text)
                translated_text = result.text
                output_token_count = result.output_token_count
            except Exception as exc:
                error_message = str(exc)
                error_count += 1

            sentence_latency_ms = (time.perf_counter() - call_started_at) * 1000
            latencies_ms.append(sentence_latency_ms)
            total_inference_seconds += sentence_latency_ms / 1000
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
                    display_name=system.display_name,
                    strategy=system.strategy,
                    runtime_backend=artifact.runtime_backend,
                    hop_details=[],
                )
            )

    manifest = load_existing_manifest(config, artifact.artifact_id)
    total_duration_s = time.perf_counter() - started_at
    summary = RuntimeSummary(
        system_id=system.system_id,
        lane=system.lane,
        route=route.route_id,
        artifact_ids=[artifact.artifact_id],
        model_ids=[manifest.model_id],
        licenses=[manifest.license],
        quantized_size=manifest.quantized_size,
        cold_start_ms=cold_start_ms,
        p50_ms=percentile(latencies_ms, 0.5),
        p95_ms=percentile(latencies_ms, 0.95),
        total_duration_s=total_duration_s,
        tokens_per_second=(float(total_output_tokens) / total_inference_seconds) if total_inference_seconds else 0.0,
        peak_rss_mb=sampler.stop(),
        empty_output_count=empty_output_count,
        error_count=error_count,
        display_name=system.display_name,
        strategy=system.strategy,
        runtime_backend=artifact.runtime_backend,
    )
    return predictions, summary


def _run_pivot_route(
    config: RootConfig,
    system: SystemSpec,
    route: RouteSpec,
    route_plan: RoutePlanSpec,
    corpus_entries,
) -> tuple[list[PredictionRecord], RuntimeSummary]:
    first_artifact = config.translation.artifacts[route_plan.artifact_ids[0]]
    second_artifact = config.translation.artifacts[route_plan.artifact_ids[1]]
    first_loaded = _load_runtime(config, first_artifact)
    second_loaded = _load_runtime(config, second_artifact)
    _warmup_pivot(config, first_loaded.runtime, second_loaded.runtime, first_artifact, second_artifact, route, corpus_entries[0].source_text)

    predictions: list[PredictionRecord] = []
    latencies_ms: list[float] = []
    total_output_tokens = 0
    total_inference_seconds = 0.0
    empty_output_count = 0
    error_count = 0

    sampler = MemorySampler()
    sampler.start()
    started_at = time.perf_counter()
    for iteration in range(1, config.translation.benchmark.decode.benchmark_iterations + 1):
        for entry in corpus_entries:
            reference = entry.expected_for_route(route.route_id)
            call_started_at = time.perf_counter()
            translated_text = ""
            error_message: str | None = None
            output_token_count = 0
            hop_details: list[HopDetail] = []

            try:
                first_started_at = time.perf_counter()
                first_result = _translate_once(config, first_loaded.runtime, first_artifact, route, entry.source_text)
                first_latency_ms = (time.perf_counter() - first_started_at) * 1000
                hop_details.append(
                    HopDetail(
                        hop="hop1",
                        source_lang=first_artifact.source_langs[0],
                        target_lang=first_artifact.target_langs[0],
                        input_text=entry.source_text,
                        output_text=first_result.text,
                        latency_ms=first_latency_ms,
                        output_token_count=first_result.output_token_count,
                    )
                )

                second_started_at = time.perf_counter()
                second_result = _translate_once(
                    config,
                    second_loaded.runtime,
                    second_artifact,
                    route,
                    first_result.text,
                )
                second_latency_ms = (time.perf_counter() - second_started_at) * 1000
                hop_details.append(
                    HopDetail(
                        hop="hop2",
                        source_lang=second_artifact.source_langs[0],
                        target_lang=second_artifact.target_langs[0],
                        input_text=first_result.text,
                        output_text=second_result.text,
                        latency_ms=second_latency_ms,
                        output_token_count=second_result.output_token_count,
                    )
                )

                translated_text = second_result.text
                output_token_count = first_result.output_token_count + second_result.output_token_count
            except Exception as exc:
                error_message = str(exc)
                error_count += 1

            sentence_latency_ms = (time.perf_counter() - call_started_at) * 1000
            latencies_ms.append(sentence_latency_ms)
            total_inference_seconds += sentence_latency_ms / 1000
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
                    display_name=system.display_name,
                    strategy=system.strategy,
                    runtime_backend=first_artifact.runtime_backend,
                    hop_details=hop_details,
                )
            )

    manifests = [load_existing_manifest(config, artifact_id) for artifact_id in route_plan.artifact_ids]
    total_duration_s = time.perf_counter() - started_at
    summary = RuntimeSummary(
        system_id=system.system_id,
        lane=system.lane,
        route=route.route_id,
        artifact_ids=list(route_plan.artifact_ids),
        model_ids=[manifest.model_id for manifest in manifests],
        licenses=[manifest.license for manifest in manifests],
        quantized_size=sum(manifest.quantized_size for manifest in manifests),
        cold_start_ms=first_loaded.cold_start_ms + second_loaded.cold_start_ms,
        p50_ms=percentile(latencies_ms, 0.5),
        p95_ms=percentile(latencies_ms, 0.95),
        total_duration_s=total_duration_s,
        tokens_per_second=(float(total_output_tokens) / total_inference_seconds) if total_inference_seconds else 0.0,
        peak_rss_mb=sampler.stop(),
        empty_output_count=empty_output_count,
        error_count=error_count,
        display_name=system.display_name,
        strategy=system.strategy,
        runtime_backend=first_artifact.runtime_backend,
    )
    return predictions, summary


def _smoke_test_direct_route(config: RootConfig, route: RouteSpec, route_plan: RoutePlanSpec, corpus_entry) -> None:
    artifact = config.translation.artifacts[route_plan.artifact_ids[0]]
    runtime = _load_runtime(config, artifact).runtime
    result = _translate_once(config, runtime, artifact, route, corpus_entry.source_text, smoke=True)
    if not result.text.strip():
        raise RuntimeError(f"empty smoke output for route={route.route_id}")


def _smoke_test_pivot_route(config: RootConfig, route: RouteSpec, route_plan: RoutePlanSpec, corpus_entry) -> None:
    first_artifact = config.translation.artifacts[route_plan.artifact_ids[0]]
    second_artifact = config.translation.artifacts[route_plan.artifact_ids[1]]
    first_runtime = _load_runtime(config, first_artifact).runtime
    second_runtime = _load_runtime(config, second_artifact).runtime
    first_result = _translate_once(config, first_runtime, first_artifact, route, corpus_entry.source_text, smoke=True)
    second_result = _translate_once(config, second_runtime, second_artifact, route, first_result.text, smoke=True)
    if not second_result.text.strip():
        raise RuntimeError(f"empty smoke output for route={route.route_id}")


def _warmup_direct(config: RootConfig, runtime, artifact: ArtifactSpec, route: RouteSpec, text: str) -> None:
    for _ in range(config.translation.benchmark.decode.warmup_iterations):
        _translate_once(config, runtime, artifact, route, text)


def _warmup_pivot(config: RootConfig, first_runtime, second_runtime, first_artifact: ArtifactSpec, second_artifact: ArtifactSpec, route: RouteSpec, text: str) -> None:
    for _ in range(config.translation.benchmark.decode.warmup_iterations):
        first_result = _translate_once(config, first_runtime, first_artifact, route, text)
        _translate_once(config, second_runtime, second_artifact, route, first_result.text)


def _load_runtime(config: RootConfig, artifact: ArtifactSpec) -> LoadedRuntime:
    quantized_dir = translation_stage_directory(config, "quantized", artifact.artifact_id)
    if artifact.runtime_backend == "llama_cpp":
        return GGUFTranslationRuntime.load(quantized_dir)
    return TranslationRuntime.load(quantized_dir)


def _translate_once(config: RootConfig, runtime, artifact: ArtifactSpec, route: RouteSpec, text: str, *, smoke: bool = False):
    max_new_tokens = min(config.translation.benchmark.decode.max_new_tokens, 64) if smoke else config.translation.benchmark.decode.max_new_tokens
    do_sample = False if smoke else config.translation.benchmark.decode.do_sample
    num_beams = 1 if smoke else config.translation.benchmark.decode.num_beams
    source_lang = route.source_lang if artifact.runtime_backend == "llama_cpp" else None
    target_lang = route.target_lang if artifact.runtime_backend == "llama_cpp" else None
    return runtime.translate(
        text,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=config.translation.benchmark.decode.batch_size,
        do_sample=do_sample,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )

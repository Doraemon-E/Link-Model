from __future__ import annotations

import importlib

from .schemas import ArtifactSpec, BenchmarkConfig, RouteSpec, SystemSpec


ARTIFACTS: dict[str, ArtifactSpec] = {
    "marian-zh-en": ArtifactSpec(
        artifact_id="marian-zh-en",
        model_id="Helsinki-NLP/opus-mt-zh-en",
        family="marian",
        source_langs=("zh",),
        target_langs=("en",),
    ),
    "marian-en-ja": ArtifactSpec(
        artifact_id="marian-en-ja",
        model_id="Helsinki-NLP/opus-mt-en-jap",
        family="marian",
        source_langs=("en",),
        target_langs=("ja",),
    ),
    "m2m100-418m": ArtifactSpec(
        artifact_id="m2m100-418m",
        model_id="facebook/m2m100_418M",
        family="m2m100",
        source_langs=("zh",),
        target_langs=("en", "ja"),
    ),
    "marian-zh-ja": ArtifactSpec(
        artifact_id="marian-zh-ja",
        model_id="Helsinki-NLP/opus-mt-tc-big-zh-ja",
        family="marian",
        source_langs=("zh",),
        target_langs=("ja",),
    ),
    "granite-3-1-2b-instruct": ArtifactSpec(
        artifact_id="granite-3-1-2b-instruct",
        model_id="ibm-granite/granite-3.1-2b-instruct",
        family="causal_llm",
        source_langs=("zh",),
        target_langs=("en", "ja"),
    ),
    "mistral-7b-instruct-v0.3": ArtifactSpec(
        artifact_id="mistral-7b-instruct-v0.3",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        family="causal_llm",
        source_langs=("zh",),
        target_langs=("en", "ja"),
    ),
    "phi-4-mini-instruct": ArtifactSpec(
        artifact_id="phi-4-mini-instruct",
        model_id="microsoft/Phi-4-mini-instruct",
        family="causal_llm",
        source_langs=("zh",),
        target_langs=("en", "ja"),
    ),
    "hy-mt1.5-1.8b-gguf-q4km": ArtifactSpec(
        artifact_id="hy-mt1.5-1.8b-gguf-q4km",
        model_id="tencent/HY-MT1.5-1.8B-GGUF",
        family="gguf_causal_llm",
        source_langs=("zh",),
        target_langs=("en", "ja"),
        artifact_format="gguf",
        runtime_backend="llama_cpp",
        quantization_format="gguf_q4_k_m",
        quantization_source="vendor_prequantized",
    ),
}


ROUTES: dict[str, RouteSpec] = {
    "zh-en": RouteSpec(
        route_id="zh-en",
        source_lang="zh",
        target_lang="en",
        reference_key="zh-en",
    ),
    "zh-ja": RouteSpec(
        route_id="zh-ja",
        source_lang="zh",
        target_lang="ja",
        reference_key="zh-ja",
    ),
}


SYSTEMS: dict[str, SystemSpec] = {
    "marian-pivot": SystemSpec(
        system_id="marian-pivot",
        lane="seq2seq",
        artifact_ids=("marian-zh-en", "marian-en-ja"),
        executor_module="benchmark.systems.marian_pivot",
        baseline=True,
    ),
    "marian-direct": SystemSpec(
        system_id="marian-direct",
        lane="seq2seq",
        artifact_ids=("marian-zh-en", "marian-zh-ja"),
        executor_module="benchmark.systems.marian_direct",
    ),
    "m2m100-418m": SystemSpec(
        system_id="m2m100-418m",
        lane="seq2seq",
        artifact_ids=("m2m100-418m",),
        executor_module="benchmark.systems.m2m100",
    ),
    "granite-3-1-2b-instruct": SystemSpec(
        system_id="granite-3-1-2b-instruct",
        lane="llm",
        artifact_ids=("granite-3-1-2b-instruct",),
        executor_module="benchmark.systems.causal_llm",
    ),
    "mistral-7b-instruct-v0.3": SystemSpec(
        system_id="mistral-7b-instruct-v0.3",
        lane="llm",
        artifact_ids=("mistral-7b-instruct-v0.3",),
        executor_module="benchmark.systems.causal_llm",
    ),
    "phi-4-mini-instruct": SystemSpec(
        system_id="phi-4-mini-instruct",
        lane="llm",
        artifact_ids=("phi-4-mini-instruct",),
        executor_module="benchmark.systems.causal_llm",
    ),
    "hy-mt1.5-1.8b-gguf-q4km": SystemSpec(
        system_id="hy-mt1.5-1.8b-gguf-q4km",
        lane="llm",
        artifact_ids=("hy-mt1.5-1.8b-gguf-q4km",),
        executor_module="benchmark.systems.gguf_causal_llm",
    ),
}


def validate_config_selection(config: BenchmarkConfig) -> None:
    unknown_systems = sorted(set(config.systems) - set(SYSTEMS))
    if unknown_systems:
        raise ValueError(f"Unknown systems in config: {unknown_systems}")

    unknown_routes = sorted(set(config.routes) - set(ROUTES))
    if unknown_routes:
        raise ValueError(f"Unknown routes in config: {unknown_routes}")


def selected_systems(config: BenchmarkConfig) -> list[SystemSpec]:
    return [SYSTEMS[system_id] for system_id in config.systems]


def selected_routes(config: BenchmarkConfig) -> list[RouteSpec]:
    return [ROUTES[route_id] for route_id in config.routes]


def selected_artifacts(config: BenchmarkConfig) -> list[ArtifactSpec]:
    artifact_ids: list[str] = []
    for system in selected_systems(config):
        for artifact_id in system.artifact_ids:
            if artifact_id not in artifact_ids:
                artifact_ids.append(artifact_id)
    return [ARTIFACTS[artifact_id] for artifact_id in artifact_ids]


def artifact_for_id(artifact_id: str) -> ArtifactSpec:
    return ARTIFACTS[artifact_id]


def system_for_id(system_id: str) -> SystemSpec:
    return SYSTEMS[system_id]


def route_for_id(route_id: str) -> RouteSpec:
    return ROUTES[route_id]


def baseline_system_id() -> str:
    for system in SYSTEMS.values():
        if system.baseline:
            return system.system_id
    raise RuntimeError("No baseline system registered.")


def load_system_module(system: SystemSpec):
    return importlib.import_module(system.executor_module)


def load_executor(system: SystemSpec):
    module = load_system_module(system)
    if not hasattr(module, "run_system"):
        raise AttributeError(f"{system.executor_module} does not expose run_system().")
    return module.run_system


def load_smoke_test(system: SystemSpec):
    module = load_system_module(system)
    smoke_test = getattr(module, "smoke_test_system", None)
    if smoke_test is None:
        return None
    return smoke_test

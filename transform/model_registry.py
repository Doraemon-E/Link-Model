from __future__ import annotations

try:
    from .download_manifest import MODEL_SPECS, ModelSpec
except ImportError:
    from download_manifest import MODEL_SPECS, ModelSpec

MODEL_NAMES = {
    spec.language_pair: spec.model_name
    for spec in MODEL_SPECS
}


def model_directory_name(spec: ModelSpec) -> str:
    return spec.local_name

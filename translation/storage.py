from __future__ import annotations

import json
from pathlib import Path

from shared.config import ArtifactSpec, RootConfig
from shared.files import ensure_directory

from .manifests import TRANSLATION_MANIFEST_FILE_NAME
from .schemas import ArtifactManifest


REQUIRED_ONNX_FILE_NAMES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
)
RUNTIME_CONFIG_FILE_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
)
MARIAN_TOKENIZER_FILE_NAMES = (
    "vocab.json",
    "source.spm",
    "target.spm",
)
WEIGHT_FILE_SUFFIXES = (
    ".safetensors",
    ".bin",
)


def ensure_translation_stage_directories(config: RootConfig) -> None:
    for path in (
        translation_models_root(config),
        translation_stage_root(config, "downloaded"),
        translation_stage_root(config, "exported"),
        translation_stage_root(config, "quantized"),
        translation_stage_root(config, "packaged"),
    ):
        ensure_directory(path)


def translation_models_root(config: RootConfig) -> Path:
    return config.shared_paths.translation_models_root


def translation_stage_root(config: RootConfig, stage: str) -> Path:
    return translation_models_root(config) / stage


def translation_stage_directory(config: RootConfig, stage: str, artifact_id: str) -> Path:
    return translation_stage_root(config, stage) / artifact_id


def translation_archive_path(config: RootConfig, artifact: ArtifactSpec) -> Path:
    return translation_stage_root(config, "packaged") / artifact.archive_file_name


def artifact_manifest_path(config: RootConfig, artifact_id: str) -> Path:
    return translation_stage_directory(config, "quantized", artifact_id) / "artifact-manifest.json"


def load_artifact_manifest(manifest_path: Path) -> ArtifactManifest:
    return ArtifactManifest.from_json_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def write_artifact_manifest(manifest_path: Path, manifest: ArtifactManifest) -> None:
    ensure_directory(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def has_required_files(directory: Path, file_names: tuple[str, ...]) -> bool:
    return directory.exists() and all((directory / file_name).exists() for file_name in file_names)


def has_weight_file(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(path.is_file() and path.suffix in WEIGHT_FILE_SUFFIXES for path in directory.iterdir())


def has_download_payload(directory: Path, artifact: ArtifactSpec) -> bool:
    if artifact.artifact_format == "gguf":
        return has_gguf_payload(directory)
    return has_required_files(directory, RUNTIME_CONFIG_FILE_NAMES) and has_required_files(directory, MARIAN_TOKENIZER_FILE_NAMES) and has_weight_file(directory)


def has_onnx_payload(directory: Path) -> bool:
    return directory.exists() and any(path.is_file() and path.suffix == ".onnx" for path in directory.rglob("*"))


def has_gguf_payload(directory: Path) -> bool:
    return directory.exists() and any(path.is_file() and path.suffix == ".gguf" for path in directory.rglob("*"))


def has_quantized_payload(artifact: ArtifactSpec, export_dir: Path, quantized_dir: Path) -> bool:
    if artifact.artifact_format == "gguf":
        return has_gguf_payload(quantized_dir)
    if not quantized_dir.exists():
        return False
    if not has_required_files(quantized_dir, REQUIRED_ONNX_FILE_NAMES):
        return False
    return (quantized_dir / TRANSLATION_MANIFEST_FILE_NAME).exists()


def resolve_single_gguf_payload(directory: Path) -> Path:
    matches = sorted(path for path in directory.rglob("*.gguf") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one GGUF payload under {directory}, found {matches or 'none'}")
    return matches[0]

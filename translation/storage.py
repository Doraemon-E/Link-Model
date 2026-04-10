from __future__ import annotations

import json
import shutil
from pathlib import Path

from shared.config import ArtifactSpec, RootConfig, REPO_ROOT
from shared.files import copy_regular_files, ensure_directory, merge_move_path

from .manifests import TRANSLATION_MANIFEST_FILE_NAME, write_translation_manifest
from .schemas import ArtifactManifest


RAW_ONNX_FILE_NAMES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
)
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
LEGACY_MODELS_DIR = REPO_ROOT / "models"
LEGACY_BENCHMARK_TRANSLATION_DIR = LEGACY_MODELS_DIR / "benchmark_translation"


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


def migrate_legacy_translation_assets(config: RootConfig) -> None:
    ensure_translation_stage_directories(config)
    print("检查旧版 translation 资产并迁移到 models/translation/ ...")
    for artifact in config.translation.artifacts.values():
        _migrate_artifact(config, artifact)


def _migrate_artifact(config: RootConfig, artifact: ArtifactSpec) -> None:
    artifact_id = artifact.artifact_id

    merge_move_path(LEGACY_MODELS_DIR / artifact_id, translation_stage_directory(config, "downloaded", artifact_id))
    merge_move_path(LEGACY_MODELS_DIR / f"{artifact_id}-onnx", translation_stage_directory(config, "exported", artifact_id))
    merge_move_path(LEGACY_MODELS_DIR / f"{artifact_id}-onnx-int8", translation_stage_directory(config, "quantized", artifact_id))
    merge_move_path(LEGACY_MODELS_DIR / f"{artifact_id}-onnx-int8.zip", translation_archive_path(config, artifact))

    for stage in ("downloaded", "exported", "quantized"):
        merge_move_path(
            LEGACY_BENCHMARK_TRANSLATION_DIR / stage / artifact_id,
            translation_stage_directory(config, stage, artifact_id),
        )

    if artifact.family == "marian":
        _migrate_legacy_quantized_files(config, artifact)
        quantized_dir = translation_stage_directory(config, "quantized", artifact_id)
        if has_required_files(quantized_dir, REQUIRED_ONNX_FILE_NAMES) and not (quantized_dir / TRANSLATION_MANIFEST_FILE_NAME).exists():
            write_translation_manifest(artifact, quantized_dir)


def _migrate_legacy_quantized_files(config: RootConfig, artifact: ArtifactSpec) -> None:
    source_dir = translation_stage_directory(config, "exported", artifact.artifact_id)
    target_dir = translation_stage_directory(config, "quantized", artifact.artifact_id)
    ensure_directory(target_dir)

    copy_regular_files(
        source_dir,
        target_dir,
        exclude_suffixes={".onnx"},
        exclude_names={TRANSLATION_MANIFEST_FILE_NAME},
        overwrite=False,
    )

    copied = False
    for file_name in RAW_ONNX_FILE_NAMES:
        legacy_file = source_dir / f"{Path(file_name).stem}_int8.onnx"
        target_file = target_dir / file_name
        if legacy_file.exists() and not target_file.exists():
            shutil.copy2(legacy_file, target_file)
            copied = True

    if copied and not (target_dir / TRANSLATION_MANIFEST_FILE_NAME).exists():
        write_translation_manifest(artifact, target_dir)

    for file_name in RAW_ONNX_FILE_NAMES:
        legacy_quantized_file = source_dir / f"{Path(file_name).stem}_int8.onnx"
        if legacy_quantized_file.exists():
            legacy_quantized_file.unlink()

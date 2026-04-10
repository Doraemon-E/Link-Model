from __future__ import annotations

import shutil
from pathlib import Path

from shared.config import REPO_ROOT, RootConfig, SpeechArtifactSpec
from shared.files import ensure_directory


LEGACY_MODELS_DIR = REPO_ROOT / "models"


def ensure_speech_stage_directories(config: RootConfig) -> None:
    for path in (
        speech_models_root(config),
        speech_download_root(config),
        speech_package_root(config),
    ):
        ensure_directory(path)


def speech_models_root(config: RootConfig) -> Path:
    return config.shared_paths.speech_models_root


def speech_download_root(config: RootConfig) -> Path:
    return speech_models_root(config) / "downloaded"


def speech_package_root(config: RootConfig) -> Path:
    return speech_models_root(config) / "packaged"


def speech_download_dir(config: RootConfig, artifact: SpeechArtifactSpec) -> Path:
    return speech_download_root(config) / artifact.package_id


def speech_archive_path(config: RootConfig, artifact: SpeechArtifactSpec) -> Path:
    return speech_package_root(config) / artifact.archive_file_name


def migrate_legacy_speech_assets(config: RootConfig) -> None:
    ensure_speech_stage_directories(config)
    print("检查旧版 speech 资产并迁移到 models/speech/ ...")
    for artifact in config.speech.artifacts.values():
        download_dir = speech_download_dir(config, artifact)
        ensure_directory(download_dir)

        legacy_model_path = LEGACY_MODELS_DIR / artifact.local_file_name
        target_model_path = download_dir / artifact.local_file_name
        if legacy_model_path.exists():
            if target_model_path.exists():
                legacy_model_path.unlink()
            else:
                shutil.move(legacy_model_path.as_posix(), target_model_path.as_posix())

        legacy_archive_path = LEGACY_MODELS_DIR / artifact.archive_file_name
        target_archive_path = speech_archive_path(config, artifact)
        if legacy_archive_path.exists():
            ensure_directory(target_archive_path.parent)
            if target_archive_path.exists():
                legacy_archive_path.unlink()
            else:
                shutil.move(legacy_archive_path.as_posix(), target_archive_path.as_posix())

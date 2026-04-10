from __future__ import annotations

from pathlib import Path

from shared.config import RootConfig, SpeechArtifactSpec
from shared.files import ensure_directory


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

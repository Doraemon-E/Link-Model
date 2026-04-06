from __future__ import annotations

import shutil

try:
    from .paths import (
        downloaded_speech_model_dir,
        ensure_speech_stage_directories,
        legacy_speech_archive_path,
        legacy_speech_model_file,
        packaged_speech_archive_path,
    )
    from .speech_manifest import SPEECH_MODEL_SPECS, SpeechModelSpec
except ImportError:
    from paths import (
        downloaded_speech_model_dir,
        ensure_speech_stage_directories,
        legacy_speech_archive_path,
        legacy_speech_model_file,
        packaged_speech_archive_path,
    )
    from speech_manifest import SPEECH_MODEL_SPECS, SpeechModelSpec


def migrate_model(spec: SpeechModelSpec) -> None:
    downloaded_dir = downloaded_speech_model_dir(spec.package_id)
    downloaded_dir.mkdir(parents=True, exist_ok=True)

    legacy_model_path = legacy_speech_model_file(spec.local_file_name)
    target_model_path = downloaded_dir / spec.local_file_name
    if legacy_model_path.exists():
        if target_model_path.exists():
            legacy_model_path.unlink()
        else:
            shutil.move(legacy_model_path.as_posix(), target_model_path.as_posix())

    legacy_archive_path = legacy_speech_archive_path(spec.package_id)
    target_archive_path = packaged_speech_archive_path(spec.package_id)
    if legacy_archive_path.exists():
        target_archive_path.parent.mkdir(parents=True, exist_ok=True)
        if target_archive_path.exists():
            legacy_archive_path.unlink()
        else:
            shutil.move(legacy_archive_path.as_posix(), target_archive_path.as_posix())


def migrate_legacy_speech_layout() -> None:
    ensure_speech_stage_directories()
    print("检查旧版 speech 资产并迁移到 models/speech/ ...")
    for spec in SPEECH_MODEL_SPECS:
        migrate_model(spec)

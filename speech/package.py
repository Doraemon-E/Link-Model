from __future__ import annotations

import shutil

from shared.config import RootConfig
from shared.files import create_temporary_directory, is_archive_complete

from .storage import ensure_speech_stage_directories, migrate_legacy_speech_assets, speech_archive_path, speech_download_dir, speech_package_root


def package_speech_artifacts(config: RootConfig) -> list[str]:
    migrate_legacy_speech_assets(config)
    ensure_speech_stage_directories(config)

    archives: list[str] = []
    for artifact in config.speech.artifacts.values():
        source_dir = speech_download_dir(config, artifact)
        source_file = source_dir / artifact.local_file_name
        archive_path = speech_archive_path(config, artifact)
        if is_archive_complete(archive_path):
            print(f"[speech package] skip existing archive: {archive_path}")
            archives.append(archive_path.as_posix())
            continue
        if not source_file.exists():
            raise FileNotFoundError(f"Missing speech model file: {source_file}")

        temp_root = create_temporary_directory(speech_package_root(config), f"tmp-{artifact.package_id}")
        try:
            payload_dir = temp_root / artifact.package_id
            payload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, payload_dir / artifact.local_file_name)
            archive_base = temp_root / artifact.package_id
            temp_archive_path = archive_base.with_suffix(".zip")
            shutil.make_archive(
                archive_base.as_posix(),
                "zip",
                root_dir=temp_root,
                base_dir=payload_dir.name,
            )
            if archive_path.exists():
                archive_path.unlink()
            shutil.move(temp_archive_path.as_posix(), archive_path.as_posix())
            print(f"[speech package] wrote {archive_path}")
            archives.append(archive_path.as_posix())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
    return archives

from __future__ import annotations

import shutil

from shared.config import ArtifactSpec, RootConfig
from shared.files import create_temporary_directory, is_archive_complete

from .manifests import TRANSLATION_MANIFEST_FILE_NAME, write_translation_manifest
from .storage import ensure_translation_stage_directories, translation_archive_path, translation_stage_directory


def package_translation_artifacts(config: RootConfig) -> list[str]:
    ensure_translation_stage_directories(config)

    archives: list[str] = []
    for artifact in package_enabled_artifacts(config):
        archive_path = translation_archive_path(config, artifact)
        source_dir = translation_stage_directory(config, "quantized", artifact.artifact_id)
        if is_archive_complete(archive_path):
            print(f"[translation package] skip existing archive: {archive_path}")
            archives.append(archive_path.as_posix())
            continue
        if artifact.family == "marian" and not (source_dir / TRANSLATION_MANIFEST_FILE_NAME).exists():
            write_translation_manifest(artifact, source_dir)
        temp_root = create_temporary_directory(archive_path.parent, f"tmp-{artifact.artifact_id}")
        try:
            payload_dir = temp_root / archive_path.stem
            shutil.copytree(source_dir, payload_dir)
            archive_base = temp_root / archive_path.stem
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
            print(f"[translation package] wrote {archive_path}")
            archives.append(archive_path.as_posix())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
    return archives


def package_enabled_artifacts(config: RootConfig) -> list[ArtifactSpec]:
    return [artifact for artifact in config.translation.artifacts.values() if artifact.package_enabled]

from __future__ import annotations

import shutil

from shared.config import RootConfig
from shared.files import create_temporary_directory, replace_directory

from .storage import ensure_speech_stage_directories, speech_download_dir, speech_download_root


def prepare_speech(config: RootConfig, *, force: bool = False) -> list[str]:
    ensure_speech_stage_directories(config)

    downloaded: list[str] = []
    for artifact in config.speech.artifacts.values():
        target_dir = speech_download_dir(config, artifact)
        target_file = target_dir / artifact.local_file_name
        if force and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[speech prepare] skip downloaded model: {target_file}")
            downloaded.append(target_dir.as_posix())
            continue

        temp_dir = create_temporary_directory(speech_download_root(config), f"tmp-{artifact.package_id}")
        try:
            from huggingface_hub import hf_hub_download

            cached_path = hf_hub_download(repo_id=artifact.repo_id, filename=artifact.source_file_name)
            shutil.copy2(cached_path, temp_dir / artifact.local_file_name)
            replace_directory(temp_dir, target_dir)
            print(f"[speech prepare] downloaded {artifact.package_id}")
            downloaded.append(target_dir.as_posix())
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    return downloaded

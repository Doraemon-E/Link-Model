from __future__ import annotations

import shutil

from huggingface_hub import hf_hub_download

try:
    from .paths import (
        DOWNLOADED_SPEECH_MODELS_DIR,
        downloaded_speech_model_dir,
        ensure_speech_stage_directories,
    )
    from .speech_manifest import SPEECH_MODEL_SPECS, SpeechModelSpec
    from .stage_helpers import create_temporary_directory, replace_directory
except ImportError:
    from paths import (
        DOWNLOADED_SPEECH_MODELS_DIR,
        downloaded_speech_model_dir,
        ensure_speech_stage_directories,
    )
    from speech_manifest import SPEECH_MODEL_SPECS, SpeechModelSpec
    from stage_helpers import create_temporary_directory, replace_directory


def is_download_complete(spec: SpeechModelSpec) -> bool:
    target_file = downloaded_speech_model_dir(spec.package_id) / spec.local_file_name
    return target_file.exists() and target_file.stat().st_size > 0


def download_model(spec: SpeechModelSpec) -> bool:
    target_dir = downloaded_speech_model_dir(spec.package_id)
    target_file = target_dir / spec.local_file_name

    if is_download_complete(spec):
        print(f"跳过已下载语音模型: {target_file}")
        return False

    temp_dir = create_temporary_directory(
        DOWNLOADED_SPEECH_MODELS_DIR,
        f"tmp-{spec.package_id}",
    )

    try:
        cached_path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.source_file_name,
        )
        shutil.copy2(cached_path, temp_dir / spec.local_file_name)
        replace_directory(temp_dir, target_dir)
        return True
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> None:
    ensure_speech_stage_directories()

    for spec in SPEECH_MODEL_SPECS:
        print(f"开始下载语音模型 {spec.package_id}: {spec.repo_id}/{spec.source_file_name}")
        downloaded = download_model(spec)
        if downloaded:
            print(f"下载完成，已保存到: {downloaded_speech_model_dir(spec.package_id)}")


if __name__ == "__main__":
    main()

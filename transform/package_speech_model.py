from __future__ import annotations

import shutil

try:
    from .paths import (
        PACKAGED_SPEECH_MODELS_DIR,
        downloaded_speech_model_dir,
        ensure_speech_stage_directories,
        packaged_speech_archive_path,
    )
    from .speech_manifest import SPEECH_MODEL_SPECS, SpeechModelSpec
    from .stage_helpers import create_temporary_directory, is_archive_complete
except ImportError:
    from paths import (
        PACKAGED_SPEECH_MODELS_DIR,
        downloaded_speech_model_dir,
        ensure_speech_stage_directories,
        packaged_speech_archive_path,
    )
    from speech_manifest import SPEECH_MODEL_SPECS, SpeechModelSpec
    from stage_helpers import create_temporary_directory, is_archive_complete


def is_download_complete(spec: SpeechModelSpec) -> bool:
    target_file = downloaded_speech_model_dir(spec.package_id) / spec.local_file_name
    return target_file.exists() and target_file.stat().st_size > 0


def package_model(spec: SpeechModelSpec) -> bool:
    source_dir = downloaded_speech_model_dir(spec.package_id)
    source_file = source_dir / spec.local_file_name
    archive_path = packaged_speech_archive_path(spec.package_id)

    if is_archive_complete(archive_path):
        print(f"跳过已存在语音压缩包: {archive_path}")
        return False

    if not is_download_complete(spec):
        raise FileNotFoundError(f"未找到可打包的语音模型文件: {source_file}")

    temp_root = create_temporary_directory(
        PACKAGED_SPEECH_MODELS_DIR,
        f"tmp-{spec.package_id}",
    )

    try:
        payload_dir = temp_root / spec.package_id
        payload_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, payload_dir / spec.local_file_name)

        archive_base = temp_root / spec.package_id
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
        return True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    ensure_speech_stage_directories()

    for spec in SPEECH_MODEL_SPECS:
        print(f"开始打包语音模型: {downloaded_speech_model_dir(spec.package_id)}")
        packaged = package_model(spec)
        if packaged:
            print(f"已生成语音压缩包: {packaged_speech_archive_path(spec.package_id)}")


if __name__ == "__main__":
    main()

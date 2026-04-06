from __future__ import annotations

import shutil

try:
    from .download_manifest import MODEL_SPECS, ModelSpec
    from .paths import PACKAGED_MODELS_DIR, ensure_stage_directories, packaged_archive_path, quantized_model_dir
    from .stage_helpers import (
        cleanup_packaging_temporary_directories,
        create_temporary_directory,
        is_archive_complete,
        is_quantized_complete,
    )
except ImportError:
    from download_manifest import MODEL_SPECS, ModelSpec
    from paths import PACKAGED_MODELS_DIR, ensure_stage_directories, packaged_archive_path, quantized_model_dir
    from stage_helpers import (
        cleanup_packaging_temporary_directories,
        create_temporary_directory,
        is_archive_complete,
        is_quantized_complete,
    )


def package_model(spec: ModelSpec) -> bool:
    source_dir = quantized_model_dir(spec.local_name)
    archive_path = packaged_archive_path(spec.local_name)

    if is_archive_complete(archive_path):
        print(f"跳过已存在压缩包: {archive_path}")
        return False

    if not is_quantized_complete(source_dir):
        raise FileNotFoundError(f"未找到可打包的量化目录: {source_dir}")

    temp_root = create_temporary_directory(PACKAGED_MODELS_DIR, f"tmp-{spec.local_name}")
    payload_dir = temp_root / archive_path.stem

    try:
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
        return True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    ensure_stage_directories()
    cleanup_packaging_temporary_directories()

    for spec in MODEL_SPECS:
        source_dir = quantized_model_dir(spec.local_name)
        archive_path = packaged_archive_path(spec.local_name)
        print(f"开始打包量化模型: {source_dir}")
        packaged = package_model(spec)
        if packaged:
            print(f"已生成压缩包: {archive_path}")


if __name__ == "__main__":
    main()

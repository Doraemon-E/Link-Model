from __future__ import annotations

import shutil
from pathlib import Path

try:
    from .download_manifest import MODEL_SPECS, ModelSpec
    from .paths import (
        RAW_ONNX_FILE_NAMES,
        TRANSLATION_MANIFEST_FILE_NAME,
        downloaded_model_dir,
        ensure_stage_directories,
        exported_model_dir,
        legacy_downloaded_model_dir,
        legacy_exported_model_dir,
        legacy_packaged_archive_path,
        legacy_quantized_model_dir,
        packaged_archive_path,
        quantized_model_dir,
    )
    from .stage_helpers import (
        copy_regular_files,
        is_quantized_complete,
    )
    from .translation_manifest import write_translation_manifest
except ImportError:
    from download_manifest import MODEL_SPECS, ModelSpec
    from paths import (
        RAW_ONNX_FILE_NAMES,
        TRANSLATION_MANIFEST_FILE_NAME,
        downloaded_model_dir,
        ensure_stage_directories,
        exported_model_dir,
        legacy_downloaded_model_dir,
        legacy_exported_model_dir,
        legacy_packaged_archive_path,
        legacy_quantized_model_dir,
        packaged_archive_path,
        quantized_model_dir,
    )
    from stage_helpers import (
        copy_regular_files,
        is_quantized_complete,
    )
    from translation_manifest import write_translation_manifest


def merge_move_path(source_path: Path, target_path: Path) -> None:
    if not source_path.exists():
        return

    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source_path.as_posix(), target_path.as_posix())
        return

    if source_path.is_dir() and target_path.is_dir():
        for child in sorted(source_path.iterdir()):
            merge_move_path(child, target_path / child.name)
        if source_path.exists():
            source_path.rmdir()
        return

    if source_path.is_file():
        source_path.unlink()


def migrate_quantized_from_legacy_export(spec: ModelSpec) -> bool:
    source_dir = exported_model_dir(spec.local_name)
    target_dir = quantized_model_dir(spec.local_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    copy_regular_files(
        source_dir,
        target_dir,
        exclude_suffixes={".onnx"},
        exclude_names={"translation-manifest.json"},
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
        write_translation_manifest(spec, target_dir)

    return copied


def cleanup_exported_directory(local_name: str) -> None:
    directory = exported_model_dir(local_name)
    if not directory.exists():
        return

    for file_name in RAW_ONNX_FILE_NAMES:
        legacy_quantized_file = directory / f"{Path(file_name).stem}_int8.onnx"
        if legacy_quantized_file.exists():
            legacy_quantized_file.unlink()


def migrate_model(spec: ModelSpec) -> None:
    merge_move_path(
        legacy_downloaded_model_dir(spec.local_name),
        downloaded_model_dir(spec.local_name),
    )
    merge_move_path(
        legacy_exported_model_dir(spec.local_name),
        exported_model_dir(spec.local_name),
    )
    merge_move_path(
        legacy_quantized_model_dir(spec.local_name),
        quantized_model_dir(spec.local_name),
    )
    merge_move_path(
        legacy_packaged_archive_path(spec.local_name),
        packaged_archive_path(spec.local_name),
    )

    if not is_quantized_complete(quantized_model_dir(spec.local_name)):
        migrate_quantized_from_legacy_export(spec)

    if quantized_model_dir(spec.local_name).exists() and not (
        quantized_model_dir(spec.local_name) / TRANSLATION_MANIFEST_FILE_NAME
    ).exists():
        write_translation_manifest(spec, quantized_model_dir(spec.local_name))

    cleanup_exported_directory(spec.local_name)


def migrate_legacy_layout() -> None:
    ensure_stage_directories()
    print("检查旧版 models/ 布局并迁移到 models/translation/ ...")
    for spec in MODEL_SPECS:
        migrate_model(spec)

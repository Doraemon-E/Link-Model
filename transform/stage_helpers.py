from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable

try:
    from .paths import (
        MARIAN_TOKENIZER_FILE_NAMES,
        PACKAGED_MODELS_DIR,
        REQUIRED_ONNX_FILE_NAMES,
        RUNTIME_CONFIG_FILE_NAMES,
        TRANSLATION_MANIFEST_FILE_NAME,
        WEIGHT_FILE_SUFFIXES,
    )
except ImportError:
    from paths import (
        MARIAN_TOKENIZER_FILE_NAMES,
        PACKAGED_MODELS_DIR,
        REQUIRED_ONNX_FILE_NAMES,
        RUNTIME_CONFIG_FILE_NAMES,
        TRANSLATION_MANIFEST_FILE_NAME,
        WEIGHT_FILE_SUFFIXES,
    )


def create_temporary_directory(parent_dir: Path, prefix: str) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=parent_dir))


def replace_directory(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(source_dir.as_posix(), target_dir.as_posix())


def copy_regular_files(
    source_dir: Path,
    target_dir: Path,
    *,
    exclude_names: Iterable[str] = (),
    exclude_suffixes: Iterable[str] = (),
    overwrite: bool = True,
) -> None:
    excluded_names = set(exclude_names)
    excluded_suffixes = set(exclude_suffixes)
    target_dir.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.name in excluded_names:
            continue
        if source_path.suffix in excluded_suffixes:
            continue

        target_path = target_dir / source_path.name
        if target_path.exists() and not overwrite:
            continue
        shutil.copy2(source_path, target_path)


def has_required_files(directory: Path, file_names: Iterable[str]) -> bool:
    return directory.exists() and all((directory / file_name).exists() for file_name in file_names)


def has_weight_file(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(
        path.is_file() and path.suffix in WEIGHT_FILE_SUFFIXES
        for path in directory.iterdir()
    )


def is_download_complete(directory: Path) -> bool:
    return (
        has_required_files(directory, RUNTIME_CONFIG_FILE_NAMES)
        and has_required_files(directory, MARIAN_TOKENIZER_FILE_NAMES)
        and has_weight_file(directory)
    )


def is_export_complete(directory: Path) -> bool:
    return (
        has_required_files(directory, RUNTIME_CONFIG_FILE_NAMES)
        and has_required_files(directory, MARIAN_TOKENIZER_FILE_NAMES)
        and has_required_files(directory, REQUIRED_ONNX_FILE_NAMES)
    )


def is_quantized_complete(directory: Path) -> bool:
    return (
        has_required_files(directory, RUNTIME_CONFIG_FILE_NAMES)
        and has_required_files(directory, MARIAN_TOKENIZER_FILE_NAMES)
        and has_required_files(directory, REQUIRED_ONNX_FILE_NAMES)
        and (directory / TRANSLATION_MANIFEST_FILE_NAME).exists()
    )


def is_archive_complete(archive_path: Path) -> bool:
    return archive_path.exists() and archive_path.is_file() and archive_path.stat().st_size > 0


def cleanup_packaging_temporary_directories() -> None:
    if not PACKAGED_MODELS_DIR.exists():
        return

    for path in PACKAGED_MODELS_DIR.iterdir():
        if not path.name.startswith("tmp-"):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


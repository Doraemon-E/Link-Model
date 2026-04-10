from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_temporary_directory(parent_dir: Path, prefix: str) -> Path:
    ensure_directory(parent_dir)
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
    ensure_directory(target_dir)

    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.name in excluded_names:
            continue
        if source_path.suffix in excluded_suffixes:
            continue

        destination_path = target_dir / source_path.name
        if destination_path.exists() and not overwrite:
            continue
        shutil.copy2(source_path, destination_path)


def directory_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def sha256_for_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1_048_576)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def installed_size_for_archive(file_path: Path) -> int:
    with zipfile.ZipFile(file_path) as archive:
        return sum(info.file_size for info in archive.infolist())


def resolve_catalog_version(output_path: Path, requested_version: int | None) -> int:
    if requested_version is not None:
        return requested_version

    if not output_path.exists():
        return 1

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return int(payload.get("version", 1))
    except Exception:
        return 1


def is_archive_complete(archive_path: Path) -> bool:
    return archive_path.exists() and archive_path.is_file() and archive_path.stat().st_size > 0

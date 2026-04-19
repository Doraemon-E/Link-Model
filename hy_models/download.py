from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ORIGINAL_REPO_ID = "tencent/HY-MT1.5-1.8B"
MLX_8BIT_REPO_ID = "mlx-community/HY-MT1.5-1.8B-8bit"

DEFAULT_ORIGINAL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")
DEFAULT_MLX_8BIT_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b-mlx-8bit")
DEFAULT_REPORT_PATH = Path("models/translation/reports/hy-mt-download-compare.json")

IGNORE_PATTERNS = ("*.gguf", "*.onnx", "*.h5", "*.msgpack", "*.ot")
_TOKENIZER_GLOBS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_MLX_WEIGHT_ANCHOR_NAMES = ("model.safetensors", "model.safetensors.index.json")
_MAX_DIFF_SAMPLES = 20


def resolve_default_original_dir(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / DEFAULT_ORIGINAL_DIR


def resolve_default_mlx_8bit_dir(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / DEFAULT_MLX_8BIT_DIR


def _snapshot_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in integration only.
        raise RuntimeError(
            "huggingface_hub is required. Install it with "
            "`uv add huggingface_hub`."
        ) from exc
    return snapshot_download(**kwargs)


def _all_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and not _is_internal_cache_file(path, directory)
        ),
        key=lambda path: _to_rel_path(path, directory),
    )


def _to_rel_path(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def _is_internal_cache_file(path: Path, base_dir: Path) -> bool:
    rel_parts = path.relative_to(base_dir).parts
    return bool(rel_parts) and rel_parts[0] == ".cache"


def _find_by_globs(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path.is_file():
                found[_to_rel_path(path, directory)] = path
    return sorted(found.values(), key=lambda path: _to_rel_path(path, directory))


def _find_tokenizer_files(directory: Path) -> list[Path]:
    found = {
        _to_rel_path(path, directory): path for path in _find_by_globs(directory, _TOKENIZER_GLOBS)
    }
    for path in _all_files(directory):
        if path.name.startswith("tokenizer"):
            found[_to_rel_path(path, directory)] = path
    return sorted(found.values(), key=lambda path: _to_rel_path(path, directory))


def _find_weight_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    found: dict[str, Path] = {}
    for path in _all_files(directory):
        if path.suffix in _WEIGHT_SUFFIXES:
            found[_to_rel_path(path, directory)] = path
    return sorted(found.values(), key=lambda path: _to_rel_path(path, directory))


def _find_anchor_files(directory: Path, names: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    by_name = set(names)
    found = {
        _to_rel_path(path, directory): path
        for path in _all_files(directory)
        if path.name in by_name
    }
    return sorted(found.values(), key=lambda path: _to_rel_path(path, directory))


def _find_named_file(directory: Path, file_name: str) -> Path | None:
    direct = directory / file_name
    if direct.is_file():
        return direct
    matches = _find_by_globs(directory, (file_name,))
    return matches[0] if matches else None


def _total_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in _all_files(directory))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _common_file_comparison(
    original_dir: Path,
    mlx_dir: Path,
    rel_paths: list[str],
) -> list[str]:
    mismatches: list[str] = []
    for rel_path in rel_paths:
        if _sha256(original_dir / rel_path) != _sha256(mlx_dir / rel_path):
            mismatches.append(rel_path)
    return sorted(mismatches)


def _inspect_model_dir(model_dir: Path) -> dict[str, Any]:
    exists = model_dir.exists()
    tokenizer_files = _find_tokenizer_files(model_dir)
    weight_files = _find_weight_files(model_dir)
    anchor_files = _find_anchor_files(model_dir, _MLX_WEIGHT_ANCHOR_NAMES)
    config_file = _find_named_file(model_dir, "config.json")
    generation_config_file = _find_named_file(model_dir, "generation_config.json")
    chat_template_file = _find_named_file(model_dir, "chat_template.jinja")
    all_files = _all_files(model_dir)
    return {
        "exists": exists,
        "total_size_bytes": _total_size_bytes(model_dir),
        "total_files": len(all_files),
        "all_files": [_to_rel_path(path, model_dir) for path in all_files],
        "key_files": {
            "config_json": _to_rel_path(config_file, model_dir) if config_file else None,
            "generation_config_json": (
                _to_rel_path(generation_config_file, model_dir) if generation_config_file else None
            ),
            "chat_template_jinja": (
                _to_rel_path(chat_template_file, model_dir) if chat_template_file else None
            ),
            "tokenizer_files": [_to_rel_path(path, model_dir) for path in tokenizer_files],
            "weight_files": [_to_rel_path(path, model_dir) for path in weight_files],
            "mlx_weight_anchor_files": [_to_rel_path(path, model_dir) for path in anchor_files],
        },
    }


def check_original_integrity(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir)
    inspected = _inspect_model_dir(directory)
    checks = {
        "has_config_json": bool(inspected["key_files"]["config_json"]),
        "has_tokenizer_files": bool(inspected["key_files"]["tokenizer_files"]),
        "has_weight_files": bool(inspected["key_files"]["weight_files"]),
    }
    missing_requirements = [
        name for name, passed in checks.items() if not passed
    ]
    return {
        "model_type": "original",
        "directory": str(directory),
        "is_complete": all(checks.values()),
        "checks": checks,
        "missing_requirements": missing_requirements,
        "key_files": inspected["key_files"],
    }


def check_mlx_integrity(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir)
    inspected = _inspect_model_dir(directory)
    checks = {
        "has_config_json": bool(inspected["key_files"]["config_json"]),
        "has_tokenizer_files": bool(inspected["key_files"]["tokenizer_files"]),
        "has_chat_template_jinja": bool(inspected["key_files"]["chat_template_jinja"]),
        "has_model_safetensors_or_index": bool(inspected["key_files"]["mlx_weight_anchor_files"]),
    }
    missing_requirements = [
        name for name, passed in checks.items() if not passed
    ]
    return {
        "model_type": "mlx_8bit",
        "directory": str(directory),
        "is_complete": all(checks.values()),
        "checks": checks,
        "missing_requirements": missing_requirements,
        "key_files": inspected["key_files"],
    }


def _integrity_for_model(model_type: str):
    if model_type == "original":
        return check_original_integrity
    if model_type == "mlx_8bit":
        return check_mlx_integrity
    raise ValueError(f"Unsupported model_type: {model_type}")


def ensure_model_download(
    repo_id: str,
    model_type: str,
    local_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    directory = Path(local_dir)
    directory.mkdir(parents=True, exist_ok=True)
    integrity_check = _integrity_for_model(model_type)
    before = integrity_check(directory)

    if before["is_complete"] and not force:
        return {
            "repo_id": repo_id,
            "model_type": model_type,
            "directory": str(directory),
            "force": force,
            "download_triggered": False,
            "snapshot_path": None,
            "status": "skipped_already_complete",
            "before_integrity": before,
            "after_integrity": before,
        }

    snapshot_path = _snapshot_download(
        repo_id=repo_id,
        local_dir=str(directory),
        ignore_patterns=list(IGNORE_PATTERNS),
        force_download=force,
    )
    after = integrity_check(directory)
    if not after["is_complete"]:
        raise RuntimeError(
            f"{model_type} download incomplete for {repo_id}: "
            f"missing {after['missing_requirements']}"
        )

    return {
        "repo_id": repo_id,
        "model_type": model_type,
        "directory": str(directory),
        "force": force,
        "download_triggered": True,
        "snapshot_path": str(snapshot_path),
        "status": "downloaded",
        "before_integrity": before,
        "after_integrity": after,
    }


def _compare_named_file(
    original_dir: Path, mlx_dir: Path, file_name: str
) -> dict[str, Any]:
    original_path = _find_named_file(original_dir, file_name)
    mlx_path = _find_named_file(mlx_dir, file_name)
    same_content: bool | None
    if original_path and mlx_path:
        same_content = _sha256(original_path) == _sha256(mlx_path)
    else:
        same_content = None
    return {
        "file_name": file_name,
        "exists_in_original": bool(original_path),
        "exists_in_mlx_8bit": bool(mlx_path),
        "same_content": same_content,
    }


def _model_summary(directory: Path) -> dict[str, Any]:
    inspected = _inspect_model_dir(directory)
    return {
        "exists": inspected["exists"],
        "total_size_bytes": inspected["total_size_bytes"],
        "total_files": inspected["total_files"],
        "key_files": inspected["key_files"],
    }


def generate_static_compare_report(
    *,
    original_dir: str | Path,
    mlx_dir: str | Path,
    original_repo_id: str = ORIGINAL_REPO_ID,
    mlx_repo_id: str = MLX_8BIT_REPO_ID,
    original_download: dict[str, Any] | None = None,
    mlx_download: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_directory = Path(original_dir)
    mlx_directory = Path(mlx_dir)

    original_summary = _model_summary(original_directory)
    mlx_summary = _model_summary(mlx_directory)
    original_integrity = check_original_integrity(original_directory)
    mlx_integrity = check_mlx_integrity(mlx_directory)

    original_files = original_summary["key_files"]["weight_files"]
    mlx_files = mlx_summary["key_files"]["weight_files"]
    common_weights = sorted(set(original_files) & set(mlx_files))

    original_tokenizer_files = original_summary["key_files"]["tokenizer_files"]
    mlx_tokenizer_files = mlx_summary["key_files"]["tokenizer_files"]
    common_tokenizers = sorted(
        set(original_tokenizer_files) & set(mlx_tokenizer_files)
    )
    tokenizer_mismatches = _common_file_comparison(
        original_directory, mlx_directory, common_tokenizers
    )

    original_all_files = original_summary["key_files"]["weight_files"]
    mlx_all_files = mlx_summary["key_files"]["weight_files"]

    original_file_set = set(_inspect_model_dir(original_directory)["all_files"])
    mlx_file_set = set(_inspect_model_dir(mlx_directory)["all_files"])
    only_in_original = sorted(original_file_set - mlx_file_set)
    only_in_mlx = sorted(mlx_file_set - original_file_set)

    original_size = original_summary["total_size_bytes"]
    mlx_size = mlx_summary["total_size_bytes"]

    report: dict[str, Any] = {
        "schema_version": 1,
        "original": {
            "repo_id": original_repo_id,
            "directory": str(original_directory),
            "integrity": original_integrity,
            "summary": original_summary,
        },
        "mlx_8bit": {
            "repo_id": mlx_repo_id,
            "directory": str(mlx_directory),
            "integrity": mlx_integrity,
            "summary": mlx_summary,
        },
        "comparison": {
            "size": {
                "original_total_size_bytes": original_size,
                "mlx_8bit_total_size_bytes": mlx_size,
                "mlx_minus_original_bytes": mlx_size - original_size,
                "mlx_to_original_ratio": (
                    round(mlx_size / original_size, 6) if original_size else None
                ),
            },
            "weights": {
                "original_weight_files": original_files,
                "mlx_8bit_weight_files": mlx_files,
                "original_weight_count": len(original_files),
                "mlx_8bit_weight_count": len(mlx_files),
                "common_weight_files": common_weights,
            },
            "tokenizer": {
                "original_tokenizer_files": original_tokenizer_files,
                "mlx_8bit_tokenizer_files": mlx_tokenizer_files,
                "only_in_original": sorted(
                    set(original_tokenizer_files) - set(mlx_tokenizer_files)
                ),
                "only_in_mlx_8bit": sorted(
                    set(mlx_tokenizer_files) - set(original_tokenizer_files)
                ),
                "common_files": common_tokenizers,
                "common_file_content_mismatches": tokenizer_mismatches,
            },
            "named_files": {
                "config_json": _compare_named_file(
                    original_directory, mlx_directory, "config.json"
                ),
                "generation_config_json": _compare_named_file(
                    original_directory, mlx_directory, "generation_config.json"
                ),
                "chat_template_jinja": _compare_named_file(
                    original_directory, mlx_directory, "chat_template.jinja"
                ),
            },
            "file_set_diff": {
                "original_file_count": len(original_file_set),
                "mlx_8bit_file_count": len(mlx_file_set),
                "common_file_count": len(original_file_set & mlx_file_set),
                "only_in_original_count": len(only_in_original),
                "only_in_mlx_8bit_count": len(only_in_mlx),
                "only_in_original_sample": only_in_original[:_MAX_DIFF_SAMPLES],
                "only_in_mlx_8bit_sample": only_in_mlx[:_MAX_DIFF_SAMPLES],
            },
            "file_diff_summary": {
                "original_weight_files": original_all_files,
                "mlx_8bit_weight_files": mlx_all_files,
                "weight_file_count_delta": len(mlx_files) - len(original_files),
            },
        },
    }

    if original_download is not None:
        report["original"]["download"] = original_download
    if mlx_download is not None:
        report["mlx_8bit"]["download"] = mlx_download
    return report


def write_compare_report(report: dict[str, Any], report_path: str | Path) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def download_and_compare(
    *,
    download_original: bool = True,
    download_mlx_8bit: bool = True,
    original_dir: str | Path = DEFAULT_ORIGINAL_DIR,
    mlx_dir: str | Path = DEFAULT_MLX_8BIT_DIR,
    force: bool = False,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    original_directory = Path(original_dir)
    mlx_directory = Path(mlx_dir)

    original_download = None
    if download_original:
        original_download = ensure_model_download(
            repo_id=ORIGINAL_REPO_ID,
            model_type="original",
            local_dir=original_directory,
            force=force,
        )

    mlx_download = None
    if download_mlx_8bit:
        mlx_download = ensure_model_download(
            repo_id=MLX_8BIT_REPO_ID,
            model_type="mlx_8bit",
            local_dir=mlx_directory,
            force=force,
        )

    report = generate_static_compare_report(
        original_dir=original_directory,
        mlx_dir=mlx_directory,
        original_repo_id=ORIGINAL_REPO_ID,
        mlx_repo_id=MLX_8BIT_REPO_ID,
        original_download=original_download,
        mlx_download=mlx_download,
    )
    write_compare_report(report, report_path)
    return report

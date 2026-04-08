from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
DEFAULT_CONFIG_PATH = BENCHMARK_ROOT / "configs" / "translation.yaml"
CORPUS_PATH = REPO_ROOT.parent / "Link" / "linkTests" / "Fixtures" / "translation_performance_corpus.json"


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_root_directory(artifacts_root: Path, stage: str) -> Path:
    return ensure_directory(artifacts_root / stage)


def artifact_stage_directory(artifacts_root: Path, stage: str, artifact_id: str) -> Path:
    return artifact_root_directory(artifacts_root, stage) / artifact_id


def artifact_manifest_path(artifacts_root: Path, artifact_id: str) -> Path:
    return artifact_stage_directory(artifacts_root, "quantized", artifact_id) / "artifact-manifest.json"


def new_result_directory(results_root: Path, timestamp: str | None = None) -> Path:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(results_root / timestamp)


def latest_result_directory(results_root: Path) -> Path:
    ensure_directory(results_root)
    candidates = [path for path in results_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No benchmark results found under {results_root}")
    return sorted(candidates)[-1]


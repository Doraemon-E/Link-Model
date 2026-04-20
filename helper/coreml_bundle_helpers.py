from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PREFERRED_COMPUTE_UNITS = ["cpuAndNeuralEngine", "cpuOnly", "all"]

REQUIRED_RUNTIME_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

OPTIONAL_RUNTIME_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def build_timestamped_coreml_paths(
    timestamp: str,
    *,
    output_root: Path,
    packaged_root: Path,
    artifact_stem: str,
) -> tuple[Path, Path]:
    output_dir = output_root / f"{artifact_stem}-{timestamp}"
    packaged_zip = packaged_root / f"{artifact_stem}-{timestamp}.zip"
    return output_dir, packaged_zip


def make_zip_with_parent(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(
        [
            "/usr/bin/ditto",
            "-c",
            "-k",
            "--norsrc",
            "--keepParent",
            str(source_dir),
            str(zip_path),
        ],
        check=True,
    )


def _stage_coreml_distribution(
    source_dir: Path,
    *,
    model_file_name: str,
    staging_root: Path,
) -> Path:
    staged_bundle_dir = staging_root / source_dir.name
    staged_bundle_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [
        source_dir / "translation-manifest.json",
        source_dir / model_file_name,
    ]
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"missing required packaging artifact: {path}")

    # Always include the exact model referenced by manifest.
    model_source = source_dir / model_file_name
    model_target = staged_bundle_dir / model_file_name
    if model_source.is_dir():
        shutil.copytree(model_source, model_target)
    else:
        shutil.copy2(model_source, model_target)

    shutil.copy2(
        source_dir / "translation-manifest.json",
        staged_bundle_dir / "translation-manifest.json",
    )

    for file_name in REQUIRED_RUNTIME_FILES:
        source_path = source_dir / file_name
        if not source_path.is_file():
            raise RuntimeError(f"missing required runtime file for packaging: {source_path}")
        shutil.copy2(source_path, staged_bundle_dir / file_name)

    for file_name in OPTIONAL_RUNTIME_FILES:
        source_path = source_dir / file_name
        if source_path.is_file():
            shutil.copy2(source_path, staged_bundle_dir / file_name)

    return staged_bundle_dir


def package_coreml_bundle(
    source_dir: Path,
    *,
    model_file_name: str,
    zip_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hy-mt-coreml-package-") as tmpdir:
        staged_dir = _stage_coreml_distribution(
            source_dir=source_dir,
            model_file_name=model_file_name,
            staging_root=Path(tmpdir),
        )
        make_zip_with_parent(staged_dir, zip_path)


def copy_runtime_files(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_required: list[str] = []

    for file_name in REQUIRED_RUNTIME_FILES:
        source = model_dir / file_name
        if not source.is_file():
            missing_required.append(file_name)
            continue
        shutil.copy2(source, output_dir / file_name)

    if missing_required:
        raise RuntimeError(
            f"missing required runtime files in model_dir={model_dir}: {missing_required}"
        )

    for file_name in OPTIONAL_RUNTIME_FILES:
        source = model_dir / file_name
        if source.is_file():
            shutil.copy2(source, output_dir / file_name)


def write_translation_manifest(
    output_dir: Path,
    *,
    model_file_name: str,
    context_length: int,
    preferred_compute_units: list[str] | None = None,
) -> Path:
    units = preferred_compute_units or DEFAULT_PREFERRED_COMPUTE_UNITS
    manifest = {
        "version": 1,
        "family": "coreml_causal_llm",
        "promptStyle": "hy_mt_coreml_chat_v1",
        "contextLength": context_length,
        "preferredComputeUnits": units,
        "coreml": {
            "kind": "mlpackage",
            "path": model_file_name,
            "inputName": "input_ids",
            "outputName": "logits",
        },
        "tokenizer": {
            "kind": "huggingface_tokenizer_json",
            "tokenizerJson": "tokenizer.json",
            "tokenizerConfig": "tokenizer_config.json",
            "generationConfig": "generation_config.json",
            "chatTemplate": "chat_template.jinja",
        },
    }
    manifest_path = output_dir / "translation-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path

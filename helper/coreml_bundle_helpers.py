from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PREFERRED_COMPUTE_UNITS = ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"]

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


def build_named_coreml_paths(
    *,
    output_root: Path,
    packaged_root: Path,
    artifact_name: str,
) -> tuple[Path, Path]:
    output_dir = output_root / artifact_name
    packaged_zip = packaged_root / f"{artifact_name}.zip"
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
    extra_model_file_names: tuple[str, ...] = (),
    staging_root: Path,
) -> Path:
    staged_bundle_dir = staging_root / source_dir.name
    staged_bundle_dir.mkdir(parents=True, exist_ok=True)

    model_artifact_names: list[str] = [model_file_name]
    for extra_name in extra_model_file_names:
        if extra_name and extra_name not in model_artifact_names:
            model_artifact_names.append(extra_name)

    required_paths = [source_dir / "translation-manifest.json"]
    required_paths.extend(source_dir / name for name in model_artifact_names)
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"missing required packaging artifact: {path}")

    for artifact_name in model_artifact_names:
        source_path = source_dir / artifact_name
        target_path = staged_bundle_dir / artifact_name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

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
    extra_model_file_names: tuple[str, ...] = (),
    zip_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hy-mt-coreml-package-") as tmpdir:
        staged_dir = _stage_coreml_distribution(
            source_dir=source_dir,
            model_file_name=model_file_name,
            extra_model_file_names=extra_model_file_names,
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
    context_length: int,
    output_name: str = "logits",
    preferred_compute_units: list[str] | None = None,
    model_file_name: str | None = None,
    coreml_kind: str | None = None,
    coreml_path: str | None = None,
) -> Path:
    if coreml_path is None:
        if model_file_name is None:
            raise ValueError("one of model_file_name or coreml_path must be provided")
        coreml_path = model_file_name

    if coreml_kind is None:
        coreml_kind = "mlmodelc" if coreml_path.endswith(".mlmodelc") else "mlpackage"
    if Path(coreml_path).is_absolute():
        raise ValueError(f"coreml_path must be relative to output_dir, got: {coreml_path}")
    if coreml_kind not in {"mlpackage", "mlmodelc"}:
        raise ValueError(f"unsupported coreml_kind={coreml_kind}")

    units = preferred_compute_units or DEFAULT_PREFERRED_COMPUTE_UNITS
    manifest = {
        "version": 1,
        "family": "coreml_causal_llm",
        "promptStyle": "hy_mt_coreml_chat_v1",
        "contextLength": context_length,
        "preferredComputeUnits": units,
        "coreml": {
            "kind": coreml_kind,
            "path": coreml_path,
            "inputName": "input_ids",
            "outputName": output_name,
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

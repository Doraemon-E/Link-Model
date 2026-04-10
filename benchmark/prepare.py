from __future__ import annotations

import json
from pathlib import Path

from .config import load_config
from .exporters import download_model, export_model, fetch_model_metadata, stage_vendor_gguf_model
from .paths import artifact_manifest_path, artifact_stage_directory, ensure_directory
from .quantize import has_quantized_onnx_payload, quantize_exported_model
from .registry import selected_artifacts, validate_config_selection
from .schemas import ArtifactManifest, ArtifactSpec


def prepare_benchmark(config_path: Path | None = None, *, force: bool = False) -> list[ArtifactManifest]:
    config = load_config(config_path)
    validate_config_selection(config)

    manifests: list[ArtifactManifest] = []
    ensure_directory(config.artifacts_root)

    for artifact in selected_artifacts(config):
        download_dir = artifact_stage_directory(config.artifacts_root, "downloaded", artifact.artifact_id)
        export_dir = artifact_stage_directory(config.artifacts_root, "exported", artifact.artifact_id)
        quantized_dir = artifact_stage_directory(config.artifacts_root, "quantized", artifact.artifact_id)
        manifest_path = artifact_manifest_path(config.artifacts_root, artifact.artifact_id)

        if force:
            for directory in (download_dir, export_dir, quantized_dir):
                if directory.exists():
                    _remove_directory(directory)

        metadata = _safe_fetch_metadata(artifact.model_id)
        prepare_error: Exception | None = None
        source_file_name: str | None = None
        source_file_sha256: str | None = None

        try:
            if not has_download_payload(download_dir):
                print(f"[prepare] downloading {artifact.artifact_id} <- {artifact.model_id}")
                download_model(artifact, download_dir)

            if artifact.artifact_format == "gguf":
                if not has_gguf_payload(quantized_dir):
                    print(f"[prepare] staging official GGUF {artifact.artifact_id}")
                    source_file_name, source_file_sha256 = stage_vendor_gguf_model(download_dir, quantized_dir)
            else:
                if not has_onnx_payload(export_dir):
                    print(f"[prepare] exporting ONNX {artifact.artifact_id}")
                    export_model(artifact, download_dir, export_dir)

                if not has_quantized_onnx_payload(export_dir, quantized_dir):
                    print(f"[prepare] quantizing INT8 {artifact.artifact_id}")
                    quantize_exported_model(
                        export_dir,
                        quantized_dir,
                        weight_type=config.quantization.weight_type,
                    )
        except Exception as exc:
            prepare_error = exc
            print(f"[prepare] failed artifact={artifact.artifact_id}: {exc}")

        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact.artifact_format == "gguf":
            export_success = has_download_payload(download_dir)
            quantize_success = has_gguf_payload(quantized_dir)
            fp32_size: int | None = None
            quantized_size = directory_size_bytes(quantized_dir) if quantize_success else 0
            quantization_ratio: float | None = None
            if quantize_success and (source_file_name is None or source_file_sha256 is None):
                staged_gguf = resolve_single_gguf_payload(quantized_dir)
                source_file_name = source_file_name or staged_gguf.name
                source_file_sha256 = source_file_sha256 or sha256_for_file(staged_gguf)
        else:
            export_success = has_onnx_payload(export_dir)
            quantize_success = has_quantized_onnx_payload(export_dir, quantized_dir)
            fp32_size = directory_size_bytes(export_dir) if export_success else 0
            quantized_size = directory_size_bytes(quantized_dir) if quantize_success else 0
            quantization_ratio = (float(quantized_size) / float(fp32_size)) if fp32_size else 0.0

        manifest = ArtifactManifest(
            artifact_id=artifact.artifact_id,
            model_id=artifact.model_id,
            revision=metadata["revision"],
            license=metadata["license"],
            source_langs=list(artifact.source_langs),
            target_langs=list(artifact.target_langs),
            fp32_size=fp32_size,
            quantized_size=quantized_size,
            quantization_ratio=quantization_ratio,
            export_success=export_success,
            quantize_success=quantize_success,
            quantization_format=artifact.quantization_format,
            quantization_source=artifact.quantization_source,
            source_file_name=source_file_name,
            source_file_sha256=source_file_sha256,
            error_message=str(prepare_error) if prepare_error is not None else None,
        )
        manifest_path.write_text(
            json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)

    return manifests


def directory_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def load_artifact_manifest(manifest_path: Path) -> ArtifactManifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ArtifactManifest.from_json_dict(payload)


def has_download_payload(directory: Path) -> bool:
    return _has_any_files(directory)


def has_onnx_payload(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(path.is_file() and path.suffix == ".onnx" for path in directory.rglob("*"))


def has_gguf_payload(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(path.is_file() and path.suffix == ".gguf" for path in directory.rglob("*"))


def has_quantized_payload(artifact: ArtifactSpec, export_dir: Path, quantized_dir: Path) -> bool:
    if artifact.artifact_format == "gguf":
        return has_gguf_payload(quantized_dir)
    return has_quantized_onnx_payload(export_dir, quantized_dir)


def _safe_fetch_metadata(model_id: str) -> dict[str, str]:
    try:
        return fetch_model_metadata(model_id)
    except Exception:
        return {
            "revision": "unknown",
            "license": "unknown",
        }


def _remove_directory(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def _has_any_files(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(path.is_file() for path in directory.rglob("*"))


def resolve_single_gguf_payload(directory: Path) -> Path:
    matches = sorted(path for path in directory.rglob("*.gguf") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one GGUF payload under {directory}, found {matches or 'none'}")
    return matches[0]


def sha256_for_file(file_path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1_048_576)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from shared.config import ArtifactSpec, RootConfig
from shared.files import copy_regular_files, create_temporary_directory, directory_size_bytes, ensure_directory, replace_directory, sha256_for_file

from .manifests import TRANSLATION_MANIFEST_FILE_NAME, write_translation_manifest
from .schemas import ArtifactManifest
from .storage import (
    RAW_ONNX_FILE_NAMES,
    artifact_manifest_path,
    ensure_translation_stage_directories,
    has_download_payload,
    has_gguf_payload,
    has_onnx_payload,
    has_quantized_payload,
    load_artifact_manifest,
    migrate_legacy_translation_assets,
    resolve_single_gguf_payload,
    translation_stage_directory,
    write_artifact_manifest,
)


def prepare_translation(config: RootConfig, *, force: bool = False) -> list[ArtifactManifest]:
    migrate_legacy_translation_assets(config)
    ensure_translation_stage_directories(config)

    manifests: list[ArtifactManifest] = []
    for artifact in required_translation_artifacts(config):
        manifests.append(_prepare_artifact(config, artifact, force=force))
    return manifests


def required_translation_artifacts(config: RootConfig) -> list[ArtifactSpec]:
    artifact_ids: list[str] = []
    for system_id in config.translation.benchmark.selected_systems:
        system = config.translation.systems[system_id]
        for route_id in config.translation.benchmark.selected_routes:
            route_plan = system.route_plans.get(route_id)
            if route_plan is None:
                continue
            for artifact_id in route_plan.artifact_ids:
                if artifact_id not in artifact_ids:
                    artifact_ids.append(artifact_id)

    for artifact in config.translation.artifacts.values():
        if artifact.package_enabled and artifact.artifact_id not in artifact_ids:
            artifact_ids.append(artifact.artifact_id)

    return [config.translation.artifacts[artifact_id] for artifact_id in artifact_ids]


def _prepare_artifact(config: RootConfig, artifact: ArtifactSpec, *, force: bool) -> ArtifactManifest:
    download_dir = translation_stage_directory(config, "downloaded", artifact.artifact_id)
    export_dir = translation_stage_directory(config, "exported", artifact.artifact_id)
    quantized_dir = translation_stage_directory(config, "quantized", artifact.artifact_id)
    manifest_path = artifact_manifest_path(config, artifact.artifact_id)

    if force:
        for directory in (download_dir, export_dir, quantized_dir):
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)

    metadata = _safe_fetch_metadata(artifact.model_id)
    prepare_error: Exception | None = None
    source_file_name: str | None = None
    source_file_sha256: str | None = None

    try:
        if not has_download_payload(download_dir, artifact):
            print(f"[translation prepare] downloading {artifact.artifact_id} <- {artifact.model_id}")
            download_model(artifact, download_dir)

        if artifact.artifact_format == "gguf":
            if not has_gguf_payload(quantized_dir):
                print(f"[translation prepare] staging official GGUF {artifact.artifact_id}")
                source_file_name, source_file_sha256 = stage_vendor_gguf_model(download_dir, quantized_dir)
        else:
            if not has_onnx_payload(export_dir):
                print(f"[translation prepare] exporting ONNX {artifact.artifact_id}")
                export_model(artifact, download_dir, export_dir)

            if not has_quantized_payload(artifact, export_dir, quantized_dir):
                print(f"[translation prepare] quantizing {artifact.artifact_id}")
                quantize_exported_model(
                    export_dir,
                    quantized_dir,
                    weight_type=config.translation.benchmark.quantization.weight_type,
                )
                if artifact.family == "marian":
                    write_translation_manifest(artifact, quantized_dir)
    except Exception as exc:
        prepare_error = exc
        print(f"[translation prepare] failed artifact={artifact.artifact_id}: {exc}")

    if artifact.artifact_format == "gguf":
        export_success = has_download_payload(download_dir, artifact)
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
        quantize_success = has_quantized_payload(artifact, export_dir, quantized_dir)
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
    write_artifact_manifest(manifest_path, manifest)
    return manifest


def download_model(artifact: ArtifactSpec, output_dir: Path) -> None:
    if artifact.artifact_format == "gguf" and artifact.quantization_source == "vendor_prequantized":
        _replace_directory_from_temporary_parent(
            output_dir.parent,
            output_dir.name,
            lambda temp_dir: _download_vendor_gguf_to_directory(artifact, temp_dir),
        )
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _replace_directory_from_temporary_parent(
        output_dir.parent,
        output_dir.name,
        lambda temp_dir: _download_to_directory(
            artifact,
            temp_dir,
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        ),
    )


def export_model(artifact: ArtifactSpec, download_dir: Path, output_dir: Path) -> None:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    def action(temp_dir: Path) -> None:
        tokenizer = AutoTokenizer.from_pretrained(download_dir.as_posix())
        model = ORTModelForSeq2SeqLM.from_pretrained(
            download_dir.as_posix(),
            export=True,
            provider="CPUExecutionProvider",
        )
        tokenizer.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)

    _replace_directory_from_temporary_parent(output_dir.parent, output_dir.name, action)


def stage_vendor_gguf_model(download_dir: Path, output_dir: Path) -> tuple[str, str]:
    source_path = resolve_single_gguf_payload(download_dir)

    def action(temp_dir: Path) -> None:
        ensure_directory(temp_dir)
        shutil.copy2(source_path, temp_dir / "model.gguf")

    _replace_directory_from_temporary_parent(output_dir.parent, output_dir.name, action)
    return source_path.name, sha256_for_file(source_path)


def fetch_model_metadata(model_id: str) -> dict[str, str]:
    from huggingface_hub import model_info

    info = model_info(model_id)
    card_data = getattr(info, "cardData", None)
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    if card_data is None or not isinstance(card_data, dict):
        card_data = {}

    revision = getattr(info, "sha", None) or "unknown"
    license_name = str(card_data.get("license") or getattr(info, "license", None) or "unknown")
    return {"revision": revision, "license": license_name}


def load_existing_manifest(config: RootConfig, artifact_id: str) -> ArtifactManifest:
    return load_artifact_manifest(artifact_manifest_path(config, artifact_id))


def quantize_exported_model(export_dir: Path, output_dir: Path, *, weight_type: str = "qint8") -> None:
    from onnxruntime.quantization import quantize_dynamic

    resolved_weight_type = _resolve_weight_type(weight_type)
    ensure_directory(output_dir.parent)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tmp-{output_dir.name}-", dir=output_dir.parent))
    onnx_sources = [path for path in sorted(export_dir.iterdir()) if path.is_file() and path.suffix == ".onnx"]
    source_external_data_files = {
        external_data_file
        for onnx_source in onnx_sources
        for external_data_file in _external_data_filenames(onnx_source)
    }

    try:
        for source_path in sorted(export_dir.iterdir()):
            if not source_path.is_file():
                continue
            if source_path.name in source_external_data_files:
                continue

            destination_path = temp_dir / source_path.name
            if source_path.suffix == ".onnx":
                quantize_dynamic(
                    model_input=source_path.as_posix(),
                    model_output=destination_path.as_posix(),
                    weight_type=resolved_weight_type,
                    use_external_data_format=bool(_external_data_filenames(source_path)),
                )
            else:
                shutil.copy2(source_path, destination_path)

        if not (temp_dir / TRANSLATION_MANIFEST_FILE_NAME).exists():
            copy_regular_files(export_dir, temp_dir, exclude_suffixes={".onnx"}, overwrite=False)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(temp_dir.as_posix(), output_dir.as_posix())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _safe_fetch_metadata(model_id: str) -> dict[str, str]:
    try:
        return fetch_model_metadata(model_id)
    except Exception:
        return {"revision": "unknown", "license": "unknown"}


def _download_to_directory(artifact: ArtifactSpec, temp_dir: Path, auto_tokenizer, auto_model) -> None:
    tokenizer = auto_tokenizer.from_pretrained(artifact.model_id)
    model = auto_model.from_pretrained(artifact.model_id)
    tokenizer.save_pretrained(temp_dir)
    model.save_pretrained(temp_dir)


def _download_vendor_gguf_to_directory(artifact: ArtifactSpec, temp_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    filename = _resolve_vendor_gguf_filename(artifact)
    hf_hub_download(
        repo_id=artifact.model_id,
        filename=filename,
        repo_type="model",
        local_dir=temp_dir.as_posix(),
    )


def _resolve_vendor_gguf_filename(artifact: ArtifactSpec) -> str:
    from huggingface_hub import model_info

    info = model_info(artifact.model_id)
    candidates = sorted(
        sibling.rfilename
        for sibling in getattr(info, "siblings", []) or []
        if getattr(sibling, "rfilename", "").lower().endswith(".gguf")
        and "q4_k_m" in Path(getattr(sibling, "rfilename", "")).name.lower()
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one Q4_K_M GGUF file for {artifact.model_id}, found {candidates or 'none'}")
    return candidates[0]


def _replace_directory_from_temporary_parent(parent_dir: Path, final_name: str, action) -> None:
    temp_dir = create_temporary_directory(parent_dir, f"tmp-{final_name}")
    try:
        action(temp_dir)
        replace_directory(temp_dir, parent_dir / final_name)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _resolve_weight_type(weight_type: str):
    from onnxruntime.quantization import QuantType

    normalized = weight_type.strip().lower()
    if normalized == "qint8":
        return QuantType.QInt8
    if normalized == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"Unsupported quantization weight type: {weight_type}")


def _external_data_filenames(model_path: Path) -> set[str]:
    import onnx

    model = onnx.load_model(model_path.as_posix(), load_external_data=False)
    external_data_files: set[str] = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location" and entry.value:
                external_data_files.add(Path(entry.value).name)
    return external_data_files

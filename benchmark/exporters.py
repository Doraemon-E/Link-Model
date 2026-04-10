from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from .schemas import ArtifactSpec


def download_model(artifact: ArtifactSpec, output_dir: Path) -> None:
    if artifact.artifact_format == "gguf" and artifact.quantization_source == "vendor_prequantized":
        _replace_directory_from_temporary_parent(
            output_dir.parent,
            output_dir.name,
            lambda temp_dir: _download_vendor_gguf_to_directory(artifact, temp_dir),
        )
        return

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    _replace_directory_from_temporary_parent(
        output_dir.parent,
        output_dir.name,
        lambda temp_dir: _download_to_directory(
            artifact,
            temp_dir,
            AutoTokenizer,
            AutoModelForSeq2SeqLM if artifact.family != "causal_llm" else AutoModelForCausalLM,
        ),
    )


def export_model(artifact: ArtifactSpec, download_dir: Path, output_dir: Path) -> None:
    from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    def action(temp_dir: Path) -> None:
        tokenizer = AutoTokenizer.from_pretrained(download_dir)
        model_cls = ORTModelForSeq2SeqLM if artifact.family != "causal_llm" else ORTModelForCausalLM
        model = model_cls.from_pretrained(
            download_dir.as_posix(),
            export=True,
            provider="CPUExecutionProvider",
        )
        tokenizer.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)

    _replace_directory_from_temporary_parent(output_dir.parent, output_dir.name, action)


def stage_vendor_gguf_model(download_dir: Path, output_dir: Path) -> tuple[str, str]:
    source_path = resolve_single_vendor_gguf_file(download_dir)

    def action(temp_dir: Path) -> None:
        _copy_or_link_file(source_path, temp_dir / "model.gguf")

    _replace_directory_from_temporary_parent(output_dir.parent, output_dir.name, action)
    return source_path.name, sha256_for_file(source_path)


def fetch_model_metadata(model_id: str) -> dict[str, str]:
    from huggingface_hub import model_info

    info = model_info(model_id)
    card_data = getattr(info, "cardData", None)
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    if card_data is None:
        card_data = {}
    if not isinstance(card_data, dict):
        card_data = {}

    revision = getattr(info, "sha", None) or "unknown"
    license_name = str(card_data.get("license") or getattr(info, "license", None) or "unknown")
    return {
        "revision": revision,
        "license": license_name,
    }


def _download_to_directory(
    artifact: ArtifactSpec,
    temp_dir: Path,
    auto_tokenizer,
    auto_model,
) -> None:
    tokenizer = auto_tokenizer.from_pretrained(artifact.model_id)
    model = auto_model.from_pretrained(artifact.model_id)
    tokenizer.save_pretrained(temp_dir)
    model.save_pretrained(temp_dir)


def resolve_vendor_gguf_filename(artifact: ArtifactSpec) -> str:
    from huggingface_hub import model_info

    info = model_info(artifact.model_id)
    candidates = sorted(
        sibling.rfilename
        for sibling in getattr(info, "siblings", []) or []
        if getattr(sibling, "rfilename", "").lower().endswith(".gguf")
        and "q4_k_m" in Path(getattr(sibling, "rfilename", "")).name.lower()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one Q4_K_M GGUF file for {artifact.model_id}, found {candidates or 'none'}"
        )
    return candidates[0]


def resolve_single_vendor_gguf_file(download_dir: Path) -> Path:
    candidates = sorted(path for path in download_dir.rglob("*.gguf") if path.is_file())
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one GGUF file under {download_dir}, found {[path.name for path in candidates] or 'none'}"
        )
    return candidates[0]


def sha256_for_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1_048_576)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_vendor_gguf_to_directory(artifact: ArtifactSpec, temp_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    filename = resolve_vendor_gguf_filename(artifact)
    hf_hub_download(
        repo_id=artifact.model_id,
        filename=filename,
        repo_type="model",
        local_dir=temp_dir.as_posix(),
    )


def _copy_or_link_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _replace_directory_from_temporary_parent(parent_dir: Path, final_name: str, action) -> None:
    parent_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tmp-{final_name}-", dir=parent_dir))
    try:
        action(temp_dir)
        destination = parent_dir / final_name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(temp_dir.as_posix(), destination.as_posix())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

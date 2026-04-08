from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .schemas import ArtifactSpec


def download_model(artifact: ArtifactSpec, output_dir: Path) -> None:
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

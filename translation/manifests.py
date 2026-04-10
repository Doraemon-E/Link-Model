from __future__ import annotations

import json
from pathlib import Path

from shared.config import ArtifactSpec


TRANSLATION_MANIFEST_FILE_NAME = "translation-manifest.json"


def build_translation_manifest(artifact: ArtifactSpec, model_dir: Path) -> dict[str, object]:
    if artifact.family != "marian":
        raise ValueError(f"Unsupported translation family for packaging: {artifact.family}")

    config = _load_json(model_dir / "config.json")
    generation_config = _load_json(model_dir / "generation_config.json")
    tokenizer_config = _load_json(model_dir / "tokenizer_config.json")

    max_length = _first_defined(
        generation_config.get("max_length"),
        tokenizer_config.get("model_max_length"),
        config.get("max_position_embeddings"),
        default=512,
    )
    suppressed_token_ids = [
        token_ids[0]
        for token_ids in generation_config.get("bad_words_ids", [])
        if isinstance(token_ids, list) and len(token_ids) == 1
    ]

    decoder_with_past_path = model_dir / "decoder_with_past_model.onnx"

    return {
        "family": artifact.family,
        "tokenizer": {
            "kind": "marian_sentencepiece_vocabulary",
            "vocabularyFile": "vocab.json",
            "sourceSentencePieceFile": "source.spm",
            "targetSentencePieceFile": "target.spm",
        },
        "onnxFiles": {
            "encoder": "encoder_model.onnx",
            "decoder": "decoder_model.onnx",
            "decoderWithPast": "decoder_with_past_model.onnx" if decoder_with_past_path.exists() else None,
        },
        "generation": {
            "maxInputLength": max_length,
            "maxOutputLength": max_length,
            "bosTokenId": _first_defined(generation_config.get("bos_token_id"), config.get("bos_token_id"), default=0),
            "eosTokenId": _first_defined(generation_config.get("eos_token_id"), config.get("eos_token_id"), default=0),
            "padTokenId": _first_defined(generation_config.get("pad_token_id"), config.get("pad_token_id"), default=65000),
            "decoderStartTokenId": _first_defined(
                generation_config.get("decoder_start_token_id"),
                config.get("decoder_start_token_id"),
                generation_config.get("pad_token_id"),
                config.get("pad_token_id"),
                default=65000,
            ),
            "suppressedTokenIds": suppressed_token_ids or None,
        },
        "tensorNames": {
            "encoderInputIDs": "input_ids",
            "encoderAttentionMask": "attention_mask",
            "encoderOutput": "last_hidden_state",
            "decoderInputIDs": "input_ids",
            "decoderEncoderAttentionMask": "encoder_attention_mask",
            "decoderEncoderHiddenStates": "encoder_hidden_states",
            "decoderOutputLogits": "logits",
        },
        "supportedLanguagePairs": [
            {
                "source": artifact.package_source or artifact.source_langs[0],
                "target": artifact.package_target or artifact.target_langs[0],
            }
        ],
    }


def write_translation_manifest(artifact: ArtifactSpec, model_dir: Path) -> Path:
    manifest_path = model_dir / TRANSLATION_MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(build_translation_manifest(artifact, model_dir), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _load_json(file_path: Path) -> dict[str, object]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _first_defined(*values, default):
    for value in values:
        if value is not None:
            return value
    return default

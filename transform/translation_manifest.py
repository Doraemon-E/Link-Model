from __future__ import annotations

import json
from pathlib import Path

try:
    from .download_manifest import ModelSpec
    from .paths import TRANSLATION_MANIFEST_FILE_NAME
except ImportError:
    from download_manifest import ModelSpec
    from paths import TRANSLATION_MANIFEST_FILE_NAME


def first_defined(*values, default):
    for value in values:
        if value is not None:
            return value
    return default


def load_json(file_path: Path) -> dict[str, object]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def build_translation_manifest(spec: ModelSpec, model_dir: Path) -> dict[str, object]:
    if spec.family != "marian":
        raise ValueError(f"Unsupported translation family: {spec.family}")

    config = load_json(model_dir / "config.json")
    generation_config = load_json(model_dir / "generation_config.json")
    tokenizer_config = load_json(model_dir / "tokenizer_config.json")

    max_length = first_defined(
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
        "family": spec.family,
        "tokenizer": {
            "kind": "marian_sentencepiece_vocabulary",
            "vocabularyFile": "vocab.json",
            "sourceSentencePieceFile": "source.spm",
            "targetSentencePieceFile": "target.spm",
        },
        "onnxFiles": {
            "encoder": "encoder_model.onnx",
            "decoder": "decoder_model.onnx",
            "decoderWithPast": (
                "decoder_with_past_model.onnx"
                if decoder_with_past_path.exists()
                else None
            ),
        },
        "generation": {
            "maxInputLength": max_length,
            "maxOutputLength": max_length,
            "bosTokenId": first_defined(
                generation_config.get("bos_token_id"),
                config.get("bos_token_id"),
                default=0,
            ),
            "eosTokenId": first_defined(
                generation_config.get("eos_token_id"),
                config.get("eos_token_id"),
                default=0,
            ),
            "padTokenId": first_defined(
                generation_config.get("pad_token_id"),
                config.get("pad_token_id"),
                default=65000,
            ),
            "decoderStartTokenId": first_defined(
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
                "source": spec.source,
                "target": spec.target,
            }
        ],
    }


def write_translation_manifest(spec: ModelSpec, model_dir: Path) -> Path:
    manifest_path = model_dir / TRANSLATION_MANIFEST_FILE_NAME
    manifest_payload = build_translation_manifest(spec, model_dir)
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path

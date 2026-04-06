from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODELS_DIR = REPO_ROOT / "models"
TRANSLATION_MODELS_DIR = LEGACY_MODELS_DIR / "translation"
DOWNLOADED_MODELS_DIR = TRANSLATION_MODELS_DIR / "downloaded"
EXPORTED_MODELS_DIR = TRANSLATION_MODELS_DIR / "exported"
QUANTIZED_MODELS_DIR = TRANSLATION_MODELS_DIR / "quantized"
PACKAGED_MODELS_DIR = TRANSLATION_MODELS_DIR / "packaged"
SPEECH_MODELS_DIR = LEGACY_MODELS_DIR / "speech"
DOWNLOADED_SPEECH_MODELS_DIR = SPEECH_MODELS_DIR / "downloaded"
PACKAGED_SPEECH_MODELS_DIR = SPEECH_MODELS_DIR / "packaged"

TRANSLATION_MANIFEST_FILE_NAME = "translation-manifest.json"

RAW_ONNX_FILE_NAMES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
)
REQUIRED_ONNX_FILE_NAMES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
)
RUNTIME_CONFIG_FILE_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
)
MARIAN_TOKENIZER_FILE_NAMES = (
    "vocab.json",
    "source.spm",
    "target.spm",
)
OPTIONAL_RUNTIME_FILE_NAMES = (
    "special_tokens_map.json",
)
WEIGHT_FILE_SUFFIXES = (
    ".safetensors",
    ".bin",
)


def ensure_stage_directories() -> None:
    for path in (
        TRANSLATION_MODELS_DIR,
        DOWNLOADED_MODELS_DIR,
        EXPORTED_MODELS_DIR,
        QUANTIZED_MODELS_DIR,
        PACKAGED_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_speech_stage_directories() -> None:
    for path in (
        SPEECH_MODELS_DIR,
        DOWNLOADED_SPEECH_MODELS_DIR,
        PACKAGED_SPEECH_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def downloaded_model_dir(local_name: str) -> Path:
    return DOWNLOADED_MODELS_DIR / local_name


def exported_model_dir(local_name: str) -> Path:
    return EXPORTED_MODELS_DIR / local_name


def quantized_model_dir(local_name: str) -> Path:
    return QUANTIZED_MODELS_DIR / local_name


def packaged_archive_path(local_name: str) -> Path:
    return PACKAGED_MODELS_DIR / f"{local_name}-onnx-int8.zip"


def legacy_downloaded_model_dir(local_name: str) -> Path:
    return LEGACY_MODELS_DIR / local_name


def legacy_exported_model_dir(local_name: str) -> Path:
    return LEGACY_MODELS_DIR / f"{local_name}-onnx"


def legacy_quantized_model_dir(local_name: str) -> Path:
    return LEGACY_MODELS_DIR / f"{local_name}-onnx-int8"


def legacy_packaged_archive_path(local_name: str) -> Path:
    return LEGACY_MODELS_DIR / f"{local_name}-onnx-int8.zip"


def downloaded_speech_model_dir(package_id: str) -> Path:
    return DOWNLOADED_SPEECH_MODELS_DIR / package_id


def packaged_speech_archive_path(package_id: str) -> Path:
    return PACKAGED_SPEECH_MODELS_DIR / f"{package_id}.zip"


def legacy_speech_model_file(file_name: str) -> Path:
    return LEGACY_MODELS_DIR / file_name


def legacy_speech_archive_path(package_id: str) -> Path:
    return LEGACY_MODELS_DIR / f"{package_id}.zip"

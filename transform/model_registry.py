from __future__ import annotations

from typing import TypedDict


class ModelSpec(TypedDict):
    language_pair: str
    model_name: str
    local_name: str
    family: str


MODEL_SPECS: list[ModelSpec] = [
    {
        "language_pair": "中文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-zh-en",
        "local_name": "opus-mt-zh-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 中文",
        "model_name": "Helsinki-NLP/opus-mt-en-zh",
        "local_name": "opus-mt-en-zh",
        "family": "marian",
    },
    {
        "language_pair": "日文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "local_name": "opus-mt-ja-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 日文",
        "model_name": "Helsinki-NLP/opus-mt-en-jap",
        "local_name": "opus-mt-en-jap",
        "family": "marian",
    },
    {
        "language_pair": "韩文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-ko-en",
        "local_name": "opus-mt-ko-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 韩文",
        "model_name": "Helsinki-NLP/opus-mt-tc-big-en-ko",
        "local_name": "opus-mt-tc-big-en-ko",
        "family": "marian",
    },
    {
        "language_pair": "法文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-fr-en",
        "local_name": "opus-mt-fr-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 法文",
        "model_name": "Helsinki-NLP/opus-mt-en-fr",
        "local_name": "opus-mt-en-fr",
        "family": "marian",
    },
    {
        "language_pair": "德文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-de-en",
        "local_name": "opus-mt-de-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 德文",
        "model_name": "Helsinki-NLP/opus-mt-en-de",
        "local_name": "opus-mt-en-de",
        "family": "marian",
    },
    {
        "language_pair": "俄文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-ru-en",
        "local_name": "opus-mt-ru-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 俄文",
        "model_name": "Helsinki-NLP/opus-mt-en-ru",
        "local_name": "opus-mt-en-ru",
        "family": "marian",
    },
    {
        "language_pair": "西班牙文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-es-en",
        "local_name": "opus-mt-es-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 西班牙文",
        "model_name": "Helsinki-NLP/opus-mt-en-es",
        "local_name": "opus-mt-en-es",
        "family": "marian",
    },
    {
        "language_pair": "意大利文 -> 英文",
        "model_name": "Helsinki-NLP/opus-mt-it-en",
        "local_name": "opus-mt-it-en",
        "family": "marian",
    },
    {
        "language_pair": "英文 -> 意大利文",
        "model_name": "Helsinki-NLP/opus-mt-en-it",
        "local_name": "opus-mt-en-it",
        "family": "marian",
    },
]

MODEL_NAMES = {
    spec["language_pair"]: spec["model_name"]
    for spec in MODEL_SPECS
}


def model_directory_name(spec: ModelSpec) -> str:
    return spec["local_name"]

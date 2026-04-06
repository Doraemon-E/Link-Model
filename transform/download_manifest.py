from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    model_name: str
    family: str
    source: str
    target: str

    @property
    def package_id(self) -> str:
        return f"{self.local_name}-onnx"

    @property
    def archive_file_name(self) -> str:
        return f"{self.local_name}-onnx-int8.zip"

    @property
    def language_pair(self) -> str:
        return f"{self.source} -> {self.target}"


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        local_name="opus-mt-zh-en",
        model_name="Helsinki-NLP/opus-mt-zh-en",
        family="marian",
        source="zho",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-zh",
        model_name="Helsinki-NLP/opus-mt-en-zh",
        family="marian",
        source="eng",
        target="zho",
    ),
    ModelSpec(
        local_name="opus-mt-ja-en",
        model_name="Helsinki-NLP/opus-mt-ja-en",
        family="marian",
        source="jpn",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-jap",
        model_name="Helsinki-NLP/opus-mt-en-jap",
        family="marian",
        source="eng",
        target="jpn",
    ),
    ModelSpec(
        local_name="opus-mt-ko-en",
        model_name="Helsinki-NLP/opus-mt-ko-en",
        family="marian",
        source="kor",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-tc-big-en-ko",
        model_name="Helsinki-NLP/opus-mt-tc-big-en-ko",
        family="marian",
        source="eng",
        target="kor",
    ),
    ModelSpec(
        local_name="opus-mt-fr-en",
        model_name="Helsinki-NLP/opus-mt-fr-en",
        family="marian",
        source="fra",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-fr",
        model_name="Helsinki-NLP/opus-mt-en-fr",
        family="marian",
        source="eng",
        target="fra",
    ),
    ModelSpec(
        local_name="opus-mt-de-en",
        model_name="Helsinki-NLP/opus-mt-de-en",
        family="marian",
        source="deu",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-de",
        model_name="Helsinki-NLP/opus-mt-en-de",
        family="marian",
        source="eng",
        target="deu",
    ),
    ModelSpec(
        local_name="opus-mt-ru-en",
        model_name="Helsinki-NLP/opus-mt-ru-en",
        family="marian",
        source="rus",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-ru",
        model_name="Helsinki-NLP/opus-mt-en-ru",
        family="marian",
        source="eng",
        target="rus",
    ),
    ModelSpec(
        local_name="opus-mt-es-en",
        model_name="Helsinki-NLP/opus-mt-es-en",
        family="marian",
        source="spa",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-es",
        model_name="Helsinki-NLP/opus-mt-en-es",
        family="marian",
        source="eng",
        target="spa",
    ),
    ModelSpec(
        local_name="opus-mt-it-en",
        model_name="Helsinki-NLP/opus-mt-it-en",
        family="marian",
        source="ita",
        target="eng",
    ),
    ModelSpec(
        local_name="opus-mt-en-it",
        model_name="Helsinki-NLP/opus-mt-en-it",
        family="marian",
        source="eng",
        target="ita",
    ),
]

MODEL_SPECS_BY_LOCAL_NAME = {
    spec.local_name: spec
    for spec in MODEL_SPECS
}

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechModelSpec:
    package_id: str
    repo_id: str
    source_file_name: str
    local_file_name: str
    family: str = "whisper"

    @property
    def archive_file_name(self) -> str:
        return f"{self.package_id}.zip"


SPEECH_MODEL_SPECS: list[SpeechModelSpec] = [
    SpeechModelSpec(
        package_id="whisper-base-q5_1",
        repo_id="osllmai-community/whisper.cpp",
        source_file_name="ggml-base-q5_1.bin",
        local_file_name="ggml-base-q5_1.bin",
    ),
]

SPEECH_MODEL_SPECS_BY_ID = {
    spec.package_id: spec
    for spec in SPEECH_MODEL_SPECS
}

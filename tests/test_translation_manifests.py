from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shared.config import ArtifactSpec
from translation.catalog import _supported_languages_for_artifact
from translation.manifests import build_translation_manifest


class TranslationManifestTests(unittest.TestCase):
    def test_gguf_package_naming_is_family_aware(self) -> None:
        artifact = ArtifactSpec(
            artifact_id="hy-mt1.5-1.8b-gguf-q4km",
            model_id="tencent/HY-MT1.5-1.8B-GGUF",
            family="gguf_causal_llm",
            source_langs=("zh",),
            target_langs=("en", "ja"),
            artifact_format="gguf",
            runtime_backend="llama_cpp",
            quantization_format="gguf_q4_k_m",
            quantization_source="vendor_prequantized",
            package_enabled=True,
        )

        self.assertEqual(artifact.package_id, "hy-mt1.5-1.8b-gguf-q4km")
        self.assertEqual(artifact.archive_file_name, "hy-mt1.5-1.8b-gguf-q4km.zip")

    def test_build_gguf_manifest_contains_runtime_and_prompt_metadata(self) -> None:
        artifact = ArtifactSpec(
            artifact_id="hy-mt1.5-1.8b-gguf-q4km",
            model_id="tencent/HY-MT1.5-1.8B-GGUF",
            family="gguf_causal_llm",
            source_langs=("zh",),
            target_langs=("en", "ja"),
            artifact_format="gguf",
            runtime_backend="llama_cpp",
            quantization_format="gguf_q4_k_m",
            quantization_source="vendor_prequantized",
            package_enabled=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.gguf").write_bytes(b"gguf")

            manifest = build_translation_manifest(artifact, model_dir)

        self.assertEqual(manifest["family"], "gguf_causal_llm")
        self.assertEqual(manifest["promptStyle"], "hy_mt_translation_v1")
        self.assertEqual(manifest["gguf"]["modelFile"], "model.gguf")
        self.assertEqual(manifest["runtime"]["contextLength"], 4096)
        self.assertEqual(manifest["generation"]["topK"], 20)
        self.assertEqual(manifest["generation"]["topP"], 0.6)
        self.assertEqual(manifest["generation"]["repetitionPenalty"], 1.05)
        self.assertEqual(
            manifest["supportedLanguages"],
            ["zho", "eng", "jpn", "kor", "fra", "deu", "rus", "spa", "ita"],
        )

    def test_catalog_supported_languages_covers_full_hy_mt_bundle(self) -> None:
        artifact = ArtifactSpec(
            artifact_id="hy-mt1.5-1.8b-gguf-q4km",
            model_id="tencent/HY-MT1.5-1.8B-GGUF",
            family="gguf_causal_llm",
            source_langs=("zh",),
            target_langs=("en", "ja"),
            artifact_format="gguf",
            runtime_backend="llama_cpp",
            quantization_format="gguf_q4_k_m",
            quantization_source="vendor_prequantized",
            package_enabled=True,
        )

        self.assertEqual(
            _supported_languages_for_artifact(artifact),
            ["zho", "eng", "jpn", "kor", "fra", "deu", "rus", "spa", "ita"],
        )


if __name__ == "__main__":
    unittest.main()

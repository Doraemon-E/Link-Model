from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import build_parser
from shared.config import load_config
from translation.report import generate_report
from translation.schemas import PredictionRecord, RuntimeSummary


class UnifiedPipelineTests(unittest.TestCase):
    def test_load_config_splits_translation_and_speech_sections(self) -> None:
        config = load_config()

        self.assertIn("opus-mt-direct", config.translation.systems)
        self.assertIn("hy-mt1.5-1.8b-gguf-q4km", config.translation.artifacts)
        self.assertIn("whisper-base-q5_1", config.speech.artifacts)
        self.assertEqual(
            set(config.translation.systems["opus-mt-pivot-via-en"].route_plans.keys()),
            {"zh-ja"},
        )
        self.assertTrue(config.translation.artifacts["opus-mt-zh-en"].package_enabled)
        self.assertFalse(config.translation.artifacts["hy-mt1.5-1.8b-gguf-q4km"].package_enabled)

    def test_root_cli_exposes_translation_and_speech_groups(self) -> None:
        parser = build_parser()

        translation_args = parser.parse_args(["translation", "prepare"])
        speech_args = parser.parse_args(["speech", "catalog"])

        self.assertEqual(translation_args.group, "translation")
        self.assertEqual(translation_args.translation_command, "prepare")
        self.assertTrue(callable(translation_args.func))
        self.assertEqual(speech_args.group, "speech")
        self.assertEqual(speech_args.speech_command, "catalog")
        self.assertTrue(callable(speech_args.func))

    def test_report_writes_exact_scores_and_gate_checks(self) -> None:
        config = load_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            predictions = self._sample_predictions()
            runtime_summaries = self._sample_runtime_summaries()

            (result_dir / "predictions.jsonl").write_text(
                "".join(json.dumps(item.to_json_dict(), ensure_ascii=False) + "\n" for item in predictions),
                encoding="utf-8",
            )
            (result_dir / "runtime-summary.json").write_text(
                json.dumps([item.to_json_dict() for item in runtime_summaries], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with patch(
                "translation.metrics._compute_comet_scores",
                return_value=(
                    {
                        ("opus-mt-direct", "zh-en"): 0.90,
                        ("hy-mt1.5-1.8b-gguf-q4km", "zh-en"): 0.80,
                        ("opus-mt-direct", "zh-ja"): 0.85,
                        ("opus-mt-pivot-via-en", "zh-ja"): 0.84,
                        ("hy-mt1.5-1.8b-gguf-q4km", "zh-ja"): 0.70,
                    },
                    True,
                ),
            ), patch("translation.metrics._compute_bleu", return_value=12.34), patch(
                "translation.metrics._compute_chrf_pp",
                return_value=56.78,
            ):
                payload = generate_report(config, result_dir=result_dir)

            self.assertEqual(len(payload["evaluations"]), 5)

            metrics_payload = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
            hy_zh_en = next(
                item
                for item in metrics_payload["evaluations"]
                if item["system_id"] == "hy-mt1.5-1.8b-gguf-q4km" and item["route"] == "zh-en"
            )
            self.assertEqual(hy_zh_en["must_preserve_hits"], 1)
            self.assertEqual(hy_zh_en["must_preserve_total"], 2)
            self.assertAlmostEqual(hy_zh_en["must_preserve_rate"], 0.5)
            self.assertIn("gate_checks", hy_zh_en)
            self.assertEqual(hy_zh_en["gate_checks"]["must_preserve_rate"]["passed"], False)
            self.assertIn("50.00%", hy_zh_en["gate_checks"]["must_preserve_rate"]["reason"])

            report_text = (result_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("mustPreserve hits/total (%)", report_text)
            self.assertIn("1/2 (50.00%)", report_text)
            self.assertIn("must_preserve_rate 50.00% < 85.00%", report_text)

    def _sample_predictions(self) -> list[PredictionRecord]:
        return [
            PredictionRecord(
                system_id="opus-mt-direct",
                lane="translation",
                route="zh-en",
                iteration=1,
                entry_id="zh-en-1",
                bucket="short",
                source_text="原文1",
                translated_text="hello ticket",
                reference_text="hello ticket",
                must_preserve=["hello", "ticket"],
                sentence_latency_ms=10.0,
                output_token_count=4,
                error=None,
                display_name="Opus MT Direct",
                strategy="direct",
                runtime_backend="onnxruntime",
            ),
            PredictionRecord(
                system_id="hy-mt1.5-1.8b-gguf-q4km",
                lane="translation",
                route="zh-en",
                iteration=1,
                entry_id="zh-en-2",
                bucket="short",
                source_text="原文2",
                translated_text="hello world",
                reference_text="hello ticket",
                must_preserve=["hello", "ticket"],
                sentence_latency_ms=12.0,
                output_token_count=4,
                error=None,
                display_name="HY MT 1.5 1.8B GGUF Q4_K_M",
                strategy="direct",
                runtime_backend="llama_cpp",
            ),
            PredictionRecord(
                system_id="opus-mt-direct",
                lane="translation",
                route="zh-ja",
                iteration=1,
                entry_id="zh-ja-1",
                bucket="short",
                source_text="原文3",
                translated_text="駅の入口",
                reference_text="駅の入口",
                must_preserve=["駅の入口"],
                sentence_latency_ms=20.0,
                output_token_count=4,
                error=None,
                display_name="Opus MT Direct",
                strategy="direct",
                runtime_backend="onnxruntime",
            ),
            PredictionRecord(
                system_id="opus-mt-pivot-via-en",
                lane="translation",
                route="zh-ja",
                iteration=1,
                entry_id="zh-ja-2",
                bucket="short",
                source_text="原文4",
                translated_text="駅の入口",
                reference_text="駅の入口",
                must_preserve=["駅の入口"],
                sentence_latency_ms=30.0,
                output_token_count=6,
                error=None,
                display_name="Opus MT Pivot via EN",
                strategy="pivot_via_en",
                runtime_backend="onnxruntime",
            ),
            PredictionRecord(
                system_id="hy-mt1.5-1.8b-gguf-q4km",
                lane="translation",
                route="zh-ja",
                iteration=1,
                entry_id="zh-ja-3",
                bucket="short",
                source_text="原文5",
                translated_text="こんにちは",
                reference_text="駅の入口",
                must_preserve=["駅の入口"],
                sentence_latency_ms=40.0,
                output_token_count=5,
                error=None,
                display_name="HY MT 1.5 1.8B GGUF Q4_K_M",
                strategy="direct",
                runtime_backend="llama_cpp",
            ),
        ]

    def _sample_runtime_summaries(self) -> list[RuntimeSummary]:
        return [
            RuntimeSummary(
                system_id="opus-mt-direct",
                lane="translation",
                route="zh-en",
                artifact_ids=["opus-mt-zh-en"],
                model_ids=["Helsinki-NLP/opus-mt-zh-en"],
                licenses=["apache-2.0"],
                quantized_size=100,
                cold_start_ms=1.0,
                p50_ms=10.0,
                p95_ms=12.0,
                total_duration_s=1.0,
                tokens_per_second=20.0,
                peak_rss_mb=100.0,
                empty_output_count=0,
                error_count=0,
                display_name="Opus MT Direct",
                strategy="direct",
                runtime_backend="onnxruntime",
            ),
            RuntimeSummary(
                system_id="hy-mt1.5-1.8b-gguf-q4km",
                lane="translation",
                route="zh-en",
                artifact_ids=["hy-mt1.5-1.8b-gguf-q4km"],
                model_ids=["tencent/HY-MT1.5-1.8B-GGUF"],
                licenses=["unknown"],
                quantized_size=200,
                cold_start_ms=2.0,
                p50_ms=11.0,
                p95_ms=15.0,
                total_duration_s=1.2,
                tokens_per_second=18.0,
                peak_rss_mb=200.0,
                empty_output_count=0,
                error_count=0,
                display_name="HY MT 1.5 1.8B GGUF Q4_K_M",
                strategy="direct",
                runtime_backend="llama_cpp",
            ),
            RuntimeSummary(
                system_id="opus-mt-direct",
                lane="translation",
                route="zh-ja",
                artifact_ids=["opus-mt-zh-ja"],
                model_ids=["Helsinki-NLP/opus-mt-tc-big-zh-ja"],
                licenses=["apache-2.0"],
                quantized_size=120,
                cold_start_ms=3.0,
                p50_ms=20.0,
                p95_ms=24.0,
                total_duration_s=1.4,
                tokens_per_second=16.0,
                peak_rss_mb=120.0,
                empty_output_count=0,
                error_count=0,
                display_name="Opus MT Direct",
                strategy="direct",
                runtime_backend="onnxruntime",
            ),
            RuntimeSummary(
                system_id="opus-mt-pivot-via-en",
                lane="translation",
                route="zh-ja",
                artifact_ids=["opus-mt-zh-en", "opus-mt-en-ja"],
                model_ids=["Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-en-jap"],
                licenses=["apache-2.0", "apache-2.0"],
                quantized_size=240,
                cold_start_ms=4.0,
                p50_ms=25.0,
                p95_ms=28.0,
                total_duration_s=1.8,
                tokens_per_second=14.0,
                peak_rss_mb=150.0,
                empty_output_count=0,
                error_count=0,
                display_name="Opus MT Pivot via EN",
                strategy="pivot_via_en",
                runtime_backend="onnxruntime",
            ),
            RuntimeSummary(
                system_id="hy-mt1.5-1.8b-gguf-q4km",
                lane="translation",
                route="zh-ja",
                artifact_ids=["hy-mt1.5-1.8b-gguf-q4km"],
                model_ids=["tencent/HY-MT1.5-1.8B-GGUF"],
                licenses=["unknown"],
                quantized_size=200,
                cold_start_ms=5.0,
                p50_ms=30.0,
                p95_ms=35.0,
                total_duration_s=2.0,
                tokens_per_second=12.0,
                peak_rss_mb=220.0,
                empty_output_count=0,
                error_count=0,
                display_name="HY MT 1.5 1.8B GGUF Q4_K_M",
                strategy="direct",
                runtime_backend="llama_cpp",
            ),
        ]


if __name__ == "__main__":
    unittest.main()

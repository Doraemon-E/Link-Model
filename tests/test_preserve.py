from __future__ import annotations

import unittest

from translation.preserve import compute_must_preserve_rate, term_is_preserved
from translation.schemas import PredictionRecord


class PreserveMatcherTests(unittest.TestCase):
    def test_matches_english_reordered_phrase(self) -> None:
        self.assertTrue(
            term_is_preserved(
                "subway station entrance",
                "I'm already at the entrance of the subway station.",
            )
        )

    def test_matches_english_negation_variant(self) -> None:
        self.assertTrue(
            term_is_preserved(
                "not going",
                "I won't be going to the company today. Instead, I'll work online.",
            )
        )

    def test_matches_japanese_numeric_normalization(self) -> None:
        self.assertTrue(
            term_is_preserved(
                "毎朝七時",
                "計画では、毎朝7時に起きて、20分間散歩した後、戻って朝食を食べます。",
            )
        )

    def test_matches_japanese_orthographic_variant(self) -> None:
        self.assertTrue(
            term_is_preserved(
                "駅の入口",
                "私はもう地下鉄駅の入り口に到着しました。",
            )
        )

    def test_keeps_semantic_overreach_out(self) -> None:
        self.assertFalse(term_is_preserved("free", "Hello, do you have time this afternoon?"))

    def test_compute_rate_uses_soft_matcher(self) -> None:
        records = [
            PredictionRecord(
                system_id="system",
                lane="llm",
                route="zh-en",
                iteration=1,
                entry_id="entry-1",
                bucket="short",
                source_text="source",
                translated_text="I'm already at the entrance of the subway station.",
                reference_text="reference",
                must_preserve=["subway station entrance", "already"],
                sentence_latency_ms=1.0,
                output_token_count=1,
                error=None,
                hop_details=[],
            ),
            PredictionRecord(
                system_id="system",
                lane="llm",
                route="zh-en",
                iteration=1,
                entry_id="entry-2",
                bucket="short",
                source_text="source",
                translated_text="Hello, do you have time this afternoon?",
                reference_text="reference",
                must_preserve=["free"],
                sentence_latency_ms=1.0,
                output_token_count=1,
                error=None,
                hop_details=[],
            ),
        ]

        self.assertAlmostEqual(compute_must_preserve_rate(records), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()

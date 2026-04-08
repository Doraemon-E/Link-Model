from __future__ import annotations

import json

from .paths import CORPUS_PATH
from .registry import ROUTES
from .schemas import CorpusEntry, CorpusExpectedResult


def load_corpus() -> list[CorpusEntry]:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Corpus must be a list: {CORPUS_PATH}")

    entries: list[CorpusEntry] = []
    for raw_entry in payload:
        expected = raw_entry["expected"]
        entries.append(
            CorpusEntry(
                entry_id=raw_entry["id"],
                bucket=raw_entry["bucket"],
                source_text=raw_entry["sourceText"],
                char_count=raw_entry["charCount"],
                scenario_tag=raw_entry["scenarioTag"],
                expected_by_route={
                    "zh-en": CorpusExpectedResult(
                        reference=expected["en"]["reference"],
                        must_preserve=tuple(expected["en"]["mustPreserve"]),
                        acceptance_note=expected["en"]["acceptanceNote"],
                    ),
                    "zh-ja": CorpusExpectedResult(
                        reference=expected["ja"]["reference"],
                        must_preserve=tuple(expected["ja"]["mustPreserve"]),
                        acceptance_note=expected["ja"]["acceptanceNote"],
                    ),
                },
            )
        )

    validate_corpus(entries)
    return entries


def validate_corpus(entries: list[CorpusEntry]) -> None:
    if len(entries) != 15:
        raise ValueError(f"Expected 15 corpus entries, found {len(entries)}.")

    bucket_counts = {bucket: 0 for bucket in ("short", "medium", "long")}
    for entry in entries:
        if entry.bucket not in bucket_counts:
            raise ValueError(f"Unknown corpus bucket: {entry.bucket}")
        bucket_counts[entry.bucket] += 1
        if entry.char_count != len(entry.source_text):
            raise ValueError(f"charCount mismatch for {entry.entry_id}")
        for route in ROUTES.values():
            expected = entry.expected_for_route(route)
            if not expected.reference:
                raise ValueError(f"Missing reference for {entry.entry_id} {route.route_id}")

    for bucket, count in bucket_counts.items():
        if count != 5:
            raise ValueError(f"Expected 5 {bucket} entries, found {count}.")

from __future__ import annotations

import json

from shared.config import RootConfig

from .schemas import CorpusEntry, CorpusExpectedResult


def load_corpus(config: RootConfig) -> list[CorpusEntry]:
    payload = json.loads(config.shared_paths.corpus_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Corpus must be a list: {config.shared_paths.corpus_path}")

    routes = config.translation.benchmark.routes
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
                    route_id: CorpusExpectedResult(
                        reference=expected[route.reference_key]["reference"],
                        must_preserve=tuple(expected[route.reference_key]["mustPreserve"]),
                        acceptance_note=expected[route.reference_key]["acceptanceNote"],
                    )
                    for route_id, route in routes.items()
                },
            )
        )

    validate_corpus(config, entries)
    return entries


def validate_corpus(config: RootConfig, entries: list[CorpusEntry]) -> None:
    if len(entries) != 15:
        raise ValueError(f"Expected 15 corpus entries, found {len(entries)}.")

    bucket_counts = {bucket: 0 for bucket in ("short", "medium", "long")}
    for entry in entries:
        if entry.bucket not in bucket_counts:
            raise ValueError(f"Unknown corpus bucket: {entry.bucket}")
        bucket_counts[entry.bucket] += 1
        if entry.char_count != len(entry.source_text):
            raise ValueError(f"charCount mismatch for {entry.entry_id}")
        for route_id in config.translation.benchmark.routes:
            expected = entry.expected_for_route(route_id)
            if not expected.reference:
                raise ValueError(f"Missing reference for {entry.entry_id} {route_id}")

    for bucket, count in bucket_counts.items():
        if count != 5:
            raise ValueError(f"Expected 5 {bucket} entries, found {count}.")

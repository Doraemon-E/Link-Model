from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .config import load_config
from .judge import build_leaderboards, judge_evaluations
from .metrics import compute_evaluations
from .paths import latest_result_directory
from .schemas import EvaluationRecord, LeaderboardRow, PredictionRecord, RuntimeSummary


def generate_report(result_dir: Path, config_path: Path | None = None) -> dict[str, object]:
    config = load_config(config_path)

    predictions = _load_predictions(result_dir / "predictions.jsonl")
    runtime_summaries = _load_runtime_summaries(result_dir / "runtime-summary.json")
    evaluations = compute_evaluations(predictions, runtime_summaries, config.metrics)
    judged_evaluations, recommendations = judge_evaluations(evaluations, config.decision_policy)
    leaderboard_rows = build_leaderboards(judged_evaluations)

    metrics_payload = {
        "evaluations": [evaluation.to_json_dict() for evaluation in judged_evaluations],
        "recommended_by_lane_route": recommendations,
        "comet_available": any(evaluation.comet is not None for evaluation in judged_evaluations),
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_leaderboard_csv(result_dir / "leaderboard.csv", leaderboard_rows)
    _write_report_markdown(result_dir / "report.md", judged_evaluations, leaderboard_rows, recommendations)
    return metrics_payload


def resolve_result_dir(results_root: Path, result_dir: Path | None) -> Path:
    if result_dir is not None:
        return result_dir
    return latest_result_directory(results_root)


def _load_predictions(predictions_path: Path) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        predictions.append(
            PredictionRecord(
                system_id=payload["system_id"],
                lane=payload.get("lane", "seq2seq"),
                route=payload["route"],
                iteration=payload["iteration"],
                entry_id=payload["entry_id"],
                bucket=payload["bucket"],
                source_text=payload["source_text"],
                translated_text=payload["translated_text"],
                reference_text=payload["reference_text"],
                must_preserve=payload["must_preserve"],
                sentence_latency_ms=payload["sentence_latency_ms"],
                output_token_count=payload["output_token_count"],
                error=payload["error"],
                hop_details=[],
            )
        )
    return predictions


def _load_runtime_summaries(runtime_summary_path: Path) -> list[RuntimeSummary]:
    payload = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
    return [RuntimeSummary(**item) for item in payload]


def _write_leaderboard_csv(output_path: Path, rows: list[LeaderboardRow]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "board_type",
                "lane",
                "route",
                "rank",
                "system_id",
                "comet",
                "p50_ms",
                "peak_rss_mb",
                "int8_size",
                "recommended",
                "eliminated",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.board_type,
                    row.lane,
                    row.route,
                    row.rank,
                    row.system_id,
                    f"{row.comet:.6f}",
                    f"{row.p50_ms:.6f}",
                    f"{row.peak_rss_mb:.6f}",
                    row.int8_size,
                    str(row.recommended).lower(),
                    str(row.eliminated).lower(),
                ]
            )


def _write_report_markdown(
    output_path: Path,
    evaluations: list[EvaluationRecord],
    leaderboard_rows: list[LeaderboardRow],
    recommendations: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append("# Translation Benchmark Report")
    lines.append("")
    lines.append("## 模型与许可证")
    for evaluation in evaluations:
        model_list = ", ".join(evaluation.model_ids)
        license_list = ", ".join(evaluation.licenses)
        lines.append(f"- `{evaluation.lane}` / `{evaluation.route}` / `{evaluation.system_id}`: models={model_list}; licenses={license_list}")

    lines.append("")
    lines.append("## 指标说明")
    if any(evaluation.comet is None for evaluation in evaluations):
        lines.append("- COMET: unavailable in this run (dependency not installed); report falls back to chrF++ / BLEU / mustPreserve for quality comparison.")
    else:
        lines.append("- COMET: available")

    lines.append("")
    lines.append("## 量化产物")
    for evaluation in evaluations:
        lines.append(f"- `{evaluation.lane}` / `{evaluation.route}` / `{evaluation.system_id}`: int8_size={evaluation.int8_size} bytes")

    lines.append("")
    lines.append("## 质量排行")
    for lane, route in sorted({(row.lane, row.route) for row in leaderboard_rows if row.board_type == 'quality'}):
        lines.append(f"### {lane} / {route}")
        for row in [item for item in leaderboard_rows if item.board_type == "quality" and item.route == route and item.lane == lane]:
            recommended = " (recommended)" if row.recommended else ""
            comet_display = "N/A" if row.comet != row.comet else f"{row.comet:.4f}"
            lines.append(f"- #{row.rank} `{row.system_id}` comet={comet_display}{recommended}")

    lines.append("")
    lines.append("## 性能排行")
    for lane, route in sorted({(row.lane, row.route) for row in leaderboard_rows if row.board_type == 'efficiency'}):
        lines.append(f"### {lane} / {route}")
        for row in [item for item in leaderboard_rows if item.board_type == "efficiency" and item.route == route and item.lane == lane]:
            lines.append(
                f"- #{row.rank} `{row.system_id}` p50_ms={row.p50_ms:.2f}, peak_rss_mb={row.peak_rss_mb:.2f}, int8_size={row.int8_size}"
            )

    lines.append("")
    lines.append("## 淘汰原因")
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[(evaluation.lane, evaluation.route)].append(evaluation)
    for (lane, route), route_evaluations in sorted(grouped.items()):
        lines.append(f"### {lane} / {route}")
        for evaluation in sorted(route_evaluations, key=lambda item: item.system_id):
            if evaluation.eliminated:
                reasons = ", ".join(evaluation.elimination_reasons)
                lines.append(f"- `{evaluation.system_id}` eliminated: {reasons}")
            else:
                suffix = " (recommended)" if recommendations.get(f"{lane}:{route}") == evaluation.system_id else ""
                lines.append(f"- `{evaluation.system_id}` kept{suffix}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

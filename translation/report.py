from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from shared.config import RootConfig

from .benchmark import resolve_result_dir
from .judge import build_leaderboards, judge_evaluations
from .metrics import compute_evaluations
from .schemas import EvaluationRecord, PredictionRecord, RuntimeSummary


def generate_report(config: RootConfig, *, result_dir: Path | None = None) -> dict[str, object]:
    resolved_result_dir = resolve_result_dir(config, result_dir)
    predictions = _load_predictions(resolved_result_dir / "predictions.jsonl")
    runtime_summaries = _load_runtime_summaries(resolved_result_dir / "runtime-summary.json")
    evaluations, comet_available = compute_evaluations(predictions, runtime_summaries, config.translation.benchmark.metrics)
    judged_evaluations, recommendations = judge_evaluations(evaluations, config)
    leaderboard_rows = build_leaderboards(judged_evaluations)

    metrics_payload = {
        "evaluations": [evaluation.to_json_dict() for evaluation in judged_evaluations],
        "recommended_by_route": recommendations,
        "comet_available": comet_available,
    }
    (resolved_result_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_leaderboard_csv(resolved_result_dir / "leaderboard.csv", leaderboard_rows)
    _write_report_markdown(resolved_result_dir / "report.md", config, judged_evaluations, leaderboard_rows, recommendations, predictions, comet_available)
    return metrics_payload


def _load_predictions(predictions_path: Path) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        predictions.append(PredictionRecord.from_json_dict(json.loads(line)))
    return predictions


def _load_runtime_summaries(runtime_summary_path: Path) -> list[RuntimeSummary]:
    payload = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
    return [RuntimeSummary.from_json_dict(item) for item in payload]


def _write_leaderboard_csv(output_path: Path, rows) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "board_type",
                "route",
                "rank",
                "system_id",
                "display_name",
                "strategy",
                "comet",
                "bleu",
                "chrf_pp",
                "must_preserve_rate",
                "must_preserve_hits",
                "must_preserve_total",
                "p50_ms",
                "peak_rss_mb",
                "quantized_size",
                "recommended",
                "eliminated",
                "status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.board_type,
                    row.route,
                    row.rank,
                    row.system_id,
                    row.display_name,
                    row.strategy,
                    "" if row.comet is None else f"{row.comet:.6f}",
                    f"{row.bleu:.6f}",
                    f"{row.chrf_pp:.6f}",
                    f"{row.must_preserve_rate:.6f}",
                    row.must_preserve_hits,
                    row.must_preserve_total,
                    f"{row.p50_ms:.6f}",
                    f"{row.peak_rss_mb:.6f}",
                    row.quantized_size,
                    str(row.recommended).lower(),
                    str(row.eliminated).lower(),
                    row.status,
                ]
            )


def _write_report_markdown(
    output_path: Path,
    config: RootConfig,
    evaluations: list[EvaluationRecord],
    leaderboard_rows,
    recommendations: dict[str, str],
    predictions: list[PredictionRecord],
    comet_available: bool,
) -> None:
    lines: list[str] = []
    lines.append(f"# {config.translation.report.title}")
    lines.append("")
    lines.append("## 模型与许可证")
    for evaluation in evaluations:
        model_list = ", ".join(evaluation.model_ids)
        license_list = ", ".join(evaluation.licenses)
        lines.append(f"- `{evaluation.route}` / `{evaluation.system_id}`: models={model_list}; licenses={license_list}")

    lines.append("")
    lines.append("## 指标说明")
    if comet_available:
        lines.append("- COMET: available")
    else:
        lines.append("- COMET: unavailable in this run; gate is skipped and the report still shows BLEU / chrF++ / mustPreserve.")

    lines.append("")
    lines.append("## Route Summary")
    for route_id in config.translation.benchmark.selected_routes:
        route_evaluations = [evaluation for evaluation in evaluations if evaluation.route == route_id]
        if not route_evaluations:
            continue
        lines.append(f"### {route_id}")
        lines.append("")
        lines.append(
            "| system | strategy | mustPreserve hits/total (%) | BLEU | chrF++ | COMET | cold_start_ms | p50_ms | p95_ms | tokens_per_second | peak_rss_mb | quantized_size | status |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        ordered = _quality_order_for_route(leaderboard_rows, route_id, route_evaluations)
        for evaluation in ordered:
            status = _status_label(evaluation)
            comet_display = "N/A" if evaluation.comet is None else f"{evaluation.comet:.4f}"
            lines.append(
                f"| `{evaluation.system_id}` | `{evaluation.strategy}` | `{evaluation.must_preserve_hits}/{evaluation.must_preserve_total} ({evaluation.must_preserve_rate * 100:.2f}%)` | {evaluation.bleu:.2f} | {evaluation.chrf_pp:.2f} | {comet_display} | {evaluation.cold_start_ms:.2f} | {evaluation.p50_ms:.2f} | {evaluation.p95_ms:.2f} | {evaluation.tokens_per_second:.2f} | {evaluation.peak_rss_mb:.2f} | {evaluation.quantized_size} | {status} |"
            )
        lines.append("")

    lines.append("## Gate Results")
    for route_id in config.translation.benchmark.selected_routes:
        route_evaluations = [evaluation for evaluation in evaluations if evaluation.route == route_id]
        if not route_evaluations:
            continue
        lines.append(f"### {route_id}")
        for evaluation in sorted(route_evaluations, key=lambda item: item.system_id):
            reasons = evaluation.elimination_reasons or [gate.reason for gate in evaluation.gate_checks.values()]
            suffix = " (recommended)" if recommendations.get(route_id) == evaluation.system_id else ""
            lines.append(f"- `{evaluation.system_id}`{suffix}: {'; '.join(reasons)}")
        lines.append("")

    hop_summary = _summarize_hops(predictions)
    if hop_summary:
        lines.append("## Pivot Hop Summary")
        for (route_id, system_id), hop_rows in sorted(hop_summary.items()):
            lines.append(f"### {route_id} / {system_id}")
            for hop_name, latency_ms, token_count in hop_rows:
                lines.append(f"- `{hop_name}` avg_latency_ms={latency_ms:.2f}, avg_output_tokens={token_count:.2f}")
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _quality_order_for_route(leaderboard_rows, route_id: str, route_evaluations: list[EvaluationRecord]) -> list[EvaluationRecord]:
    quality_rows = [row for row in leaderboard_rows if row.board_type == "quality" and row.route == route_id]
    evaluation_by_system = {evaluation.system_id: evaluation for evaluation in route_evaluations}
    ordered = [evaluation_by_system[row.system_id] for row in quality_rows if row.system_id in evaluation_by_system]
    seen = {evaluation.system_id for evaluation in ordered}
    ordered.extend(sorted((evaluation for evaluation in route_evaluations if evaluation.system_id not in seen), key=lambda item: item.system_id))
    return ordered


def _status_label(evaluation: EvaluationRecord) -> str:
    if evaluation.recommended:
        return "recommended"
    if evaluation.eliminated:
        return "eliminated"
    return "kept"


def _summarize_hops(predictions: list[PredictionRecord]) -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for prediction in predictions:
        for hop in prediction.hop_details:
            grouped[(prediction.route, prediction.system_id, hop.hop)].append((hop.latency_ms, hop.output_token_count))

    summary: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for (route_id, system_id, hop_name), values in grouped.items():
        avg_latency = sum(item[0] for item in values) / len(values)
        avg_tokens = sum(item[1] for item in values) / len(values)
        summary[(route_id, system_id)].append((hop_name, avg_latency, avg_tokens))

    for key in summary:
        summary[key] = sorted(summary[key], key=lambda item: item[0])
    return summary

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from shared.config import DecisionPolicy, RootConfig

from .schemas import EvaluationRecord, GateCheck, LeaderboardRow


def judge_evaluations(
    evaluations: list[EvaluationRecord],
    config: RootConfig,
) -> tuple[list[EvaluationRecord], dict[str, str]]:
    policy = config.translation.benchmark.decision_policy
    baseline_by_route = _baseline_comet_by_route(evaluations, config)
    grouped_by_route: dict[str, list[EvaluationRecord]] = defaultdict(list)
    updated_evaluations: list[EvaluationRecord] = []

    for evaluation in evaluations:
        gate_checks = _build_gate_checks(evaluation, baseline_by_route.get(evaluation.route), policy)
        elimination_reasons = [gate.reason for gate in gate_checks.values() if gate.passed is False]
        updated = replace(
            evaluation,
            gate_checks=gate_checks,
            eliminated=bool(elimination_reasons),
            elimination_reasons=elimination_reasons,
            recommended=False,
        )
        updated_evaluations.append(updated)
        grouped_by_route[evaluation.route].append(updated)

    recommendations: dict[str, str] = {}
    final_evaluations: list[EvaluationRecord] = []
    for route, route_evaluations in grouped_by_route.items():
        recommended_system_id = _choose_recommended(route_evaluations, policy)
        if recommended_system_id is not None:
            recommendations[route] = recommended_system_id
        for evaluation in route_evaluations:
            final_evaluations.append(replace(evaluation, recommended=evaluation.system_id == recommended_system_id))

    return sorted(final_evaluations, key=lambda item: (item.route, item.system_id)), recommendations


def build_leaderboards(evaluations: list[EvaluationRecord]) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    grouped_by_route: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for evaluation in evaluations:
        grouped_by_route[evaluation.route].append(evaluation)

    for route, route_evaluations in grouped_by_route.items():
        quality_sorted = sorted(
            route_evaluations,
            key=lambda item: (
                -(item.comet if item.comet is not None else float("-inf")),
                -item.chrf_pp,
                -item.bleu,
                -item.must_preserve_rate,
            ),
        )
        efficiency_sorted = sorted(
            route_evaluations,
            key=lambda item: (item.p50_ms, item.peak_rss_mb, item.quantized_size),
        )
        frontier = _pareto_frontier(route_evaluations)
        rows.extend(_to_leaderboard_rows("quality", route, quality_sorted))
        rows.extend(_to_leaderboard_rows("efficiency", route, efficiency_sorted))
        rows.extend(_to_leaderboard_rows("pareto_frontier", route, frontier))

    return rows


def _baseline_comet_by_route(evaluations: list[EvaluationRecord], config: RootConfig) -> dict[str, float]:
    baseline_by_route: dict[str, float] = {}
    for route_id in config.translation.benchmark.selected_routes:
        baseline_system_id = _baseline_system_for_route(config, route_id)
        if baseline_system_id is None:
            continue
        for evaluation in evaluations:
            if evaluation.route == route_id and evaluation.system_id == baseline_system_id and evaluation.comet is not None:
                baseline_by_route[route_id] = evaluation.comet
                break
    return baseline_by_route


def _baseline_system_for_route(config: RootConfig, route_id: str) -> str | None:
    for system_id in config.translation.benchmark.selected_systems:
        system = config.translation.systems[system_id]
        if system.baseline and route_id in system.route_plans:
            return system.system_id
    return None


def _build_gate_checks(
    evaluation: EvaluationRecord,
    baseline_comet: float | None,
    policy: DecisionPolicy,
) -> dict[str, GateCheck]:
    gate_checks: dict[str, GateCheck] = {}
    gate_checks["error_count"] = GateCheck(
        value=evaluation.error_count,
        threshold=0,
        passed=evaluation.error_count == 0,
        reason="error_count == 0" if evaluation.error_count == 0 else f"error_count {evaluation.error_count} > 0",
    )
    gate_checks["empty_output_count"] = GateCheck(
        value=evaluation.empty_output_count,
        threshold=0,
        passed=evaluation.empty_output_count == 0,
        reason=(
            "empty_output_count == 0"
            if evaluation.empty_output_count == 0
            else f"empty_output_count {evaluation.empty_output_count} > 0"
        ),
    )
    gate_checks["must_preserve_rate"] = GateCheck(
        value=evaluation.must_preserve_rate,
        threshold=policy.must_preserve_threshold,
        passed=evaluation.must_preserve_rate >= policy.must_preserve_threshold,
        reason=(
            f"must_preserve_rate {evaluation.must_preserve_rate * 100:.2f}% >= {policy.must_preserve_threshold * 100:.2f}%"
            if evaluation.must_preserve_rate >= policy.must_preserve_threshold
            else f"must_preserve_rate {evaluation.must_preserve_rate * 100:.2f}% < {policy.must_preserve_threshold * 100:.2f}%"
        ),
    )

    if evaluation.comet is None or baseline_comet is None:
        gate_checks["comet_delta"] = GateCheck(
            value=evaluation.comet,
            threshold=None,
            passed=None,
            reason="COMET unavailable; gate skipped.",
        )
    else:
        threshold = baseline_comet - policy.comet_delta_threshold
        passed = evaluation.comet >= threshold
        gate_checks["comet_delta"] = GateCheck(
            value=evaluation.comet,
            threshold=threshold,
            passed=passed,
            reason=(
                f"comet {evaluation.comet:.4f} >= opus baseline {baseline_comet:.4f} - {policy.comet_delta_threshold:.2f}"
                if passed
                else f"comet {evaluation.comet:.4f} < opus baseline {baseline_comet:.4f} - {policy.comet_delta_threshold:.2f}"
            ),
        )
    return gate_checks


def _choose_recommended(evaluations: list[EvaluationRecord], policy: DecisionPolicy) -> str | None:
    eligible = [evaluation for evaluation in evaluations if not evaluation.eliminated]
    if not eligible:
        return None

    ordered = sorted(eligible, key=lambda item: (item.p50_ms, item.peak_rss_mb, item.quantized_size))
    if len(ordered) == 1:
        return ordered[0].system_id

    first = ordered[0]
    second = ordered[1]
    if second.p50_ms <= (first.p50_ms * (1.0 + policy.latency_tie_threshold)):
        if second.peak_rss_mb < first.peak_rss_mb:
            return second.system_id
        if second.peak_rss_mb == first.peak_rss_mb and second.quantized_size < first.quantized_size:
            return second.system_id
    return first.system_id


def _pareto_frontier(evaluations: list[EvaluationRecord]) -> list[EvaluationRecord]:
    frontier: list[EvaluationRecord] = []
    for candidate in evaluations:
        dominated = False
        for other in evaluations:
            if other.system_id == candidate.system_id:
                continue
            other_comet = other.comet if other.comet is not None else float("-inf")
            candidate_comet = candidate.comet if candidate.comet is not None else float("-inf")
            better_or_equal = (
                other_comet >= candidate_comet
                and other.p50_ms <= candidate.p50_ms
                and other.peak_rss_mb <= candidate.peak_rss_mb
                and other.quantized_size <= candidate.quantized_size
            )
            strictly_better = (
                other_comet > candidate_comet
                or other.p50_ms < candidate.p50_ms
                or other.peak_rss_mb < candidate.peak_rss_mb
                or other.quantized_size < candidate.quantized_size
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    return sorted(
        frontier,
        key=lambda item: (-(item.comet if item.comet is not None else float("-inf")), item.p50_ms, item.peak_rss_mb),
    )


def _to_leaderboard_rows(board_type: str, route: str, evaluations: list[EvaluationRecord]) -> list[LeaderboardRow]:
    return [
        LeaderboardRow(
            board_type=board_type,
            route=route,
            rank=index,
            system_id=evaluation.system_id,
            display_name=evaluation.display_name,
            strategy=evaluation.strategy,
            comet=evaluation.comet,
            bleu=evaluation.bleu,
            chrf_pp=evaluation.chrf_pp,
            must_preserve_rate=evaluation.must_preserve_rate,
            must_preserve_hits=evaluation.must_preserve_hits,
            must_preserve_total=evaluation.must_preserve_total,
            p50_ms=evaluation.p50_ms,
            peak_rss_mb=evaluation.peak_rss_mb,
            quantized_size=evaluation.quantized_size,
            recommended=evaluation.recommended,
            eliminated=evaluation.eliminated,
            status=_status_for_evaluation(evaluation),
        )
        for index, evaluation in enumerate(evaluations, start=1)
    ]


def _status_for_evaluation(evaluation: EvaluationRecord) -> str:
    if evaluation.recommended:
        return "recommended"
    if evaluation.eliminated:
        return "eliminated"
    return "kept"

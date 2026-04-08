from __future__ import annotations

from collections import defaultdict

from .registry import baseline_system_id
from .schemas import DecisionPolicy, EvaluationRecord, LeaderboardRow


def judge_evaluations(
    evaluations: list[EvaluationRecord],
    decision_policy: DecisionPolicy,
) -> tuple[list[EvaluationRecord], dict[str, str]]:
    baseline_id = baseline_system_id()
    baseline_comet_by_route = {
        evaluation.route: evaluation.comet
        for evaluation in evaluations
        if evaluation.system_id == baseline_id and evaluation.comet is not None
    }

    updated_evaluations: list[EvaluationRecord] = []
    grouped_by_lane_route: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)

    for evaluation in evaluations:
        elimination_reasons: list[str] = []
        baseline_comet = baseline_comet_by_route.get(evaluation.route)

        if evaluation.error_count > 0:
            elimination_reasons.append("error_count > 0")
        if evaluation.empty_output_count > 0:
            elimination_reasons.append("empty_output_count > 0")
        if evaluation.must_preserve_rate < decision_policy.must_preserve_threshold:
            elimination_reasons.append(
                f"must_preserve_rate < {decision_policy.must_preserve_threshold:.2f}"
            )
        if (
            evaluation.comet is not None
            and baseline_comet is not None
            and evaluation.comet < (baseline_comet - decision_policy.comet_delta_threshold)
        ):
            elimination_reasons.append(
                f"comet < marian_baseline - {decision_policy.comet_delta_threshold:.2f}"
            )

        updated = EvaluationRecord(
            **{
                **evaluation.to_json_dict(),
                "eliminated": bool(elimination_reasons),
                "elimination_reasons": elimination_reasons,
                "recommended": False,
            }
        )
        updated_evaluations.append(updated)
        grouped_by_lane_route[(updated.lane, updated.route)].append(updated)

    recommendations: dict[str, str] = {}
    final_evaluations: list[EvaluationRecord] = []

    for (lane, route), route_evaluations in grouped_by_lane_route.items():
        recommended_system_id = _choose_recommended(route_evaluations, decision_policy)
        if recommended_system_id is not None:
            recommendations[f"{lane}:{route}"] = recommended_system_id

        for evaluation in route_evaluations:
            final_evaluations.append(
                EvaluationRecord(
                    **{
                        **evaluation.to_json_dict(),
                        "recommended": evaluation.system_id == recommended_system_id,
                    }
                )
            )

    return sorted(final_evaluations, key=lambda item: (item.lane, item.route, item.system_id)), recommendations


def build_leaderboards(evaluations: list[EvaluationRecord]) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    grouped_by_lane_route: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for evaluation in evaluations:
        grouped_by_lane_route[(evaluation.lane, evaluation.route)].append(evaluation)

    for (lane, route), route_evaluations in grouped_by_lane_route.items():
        quality_sorted = sorted(
            route_evaluations,
            key=lambda item: (
                -(item.comet if item.comet is not None else float("-inf")),
                -item.chrf_pp,
                -item.bleu,
            ),
        )
        efficiency_sorted = sorted(
            route_evaluations,
            key=lambda item: (item.p50_ms, item.peak_rss_mb, item.int8_size),
        )
        frontier = _pareto_frontier(route_evaluations)

        rows.extend(_to_leaderboard_rows("quality", lane, route, quality_sorted))
        rows.extend(_to_leaderboard_rows("efficiency", lane, route, efficiency_sorted))
        rows.extend(_to_leaderboard_rows("pareto_frontier", lane, route, frontier))

    return rows


def _choose_recommended(
    evaluations: list[EvaluationRecord],
    decision_policy: DecisionPolicy,
) -> str | None:
    eligible = [evaluation for evaluation in evaluations if not evaluation.eliminated]
    if not eligible:
        return None

    ordered = sorted(eligible, key=lambda item: (item.p50_ms, item.peak_rss_mb, item.int8_size))
    if len(ordered) == 1:
        return ordered[0].system_id

    first = ordered[0]
    second = ordered[1]
    if second.p50_ms <= (first.p50_ms * (1.0 + decision_policy.latency_tie_threshold)):
        if second.peak_rss_mb < first.peak_rss_mb:
            return second.system_id
        if second.peak_rss_mb == first.peak_rss_mb and second.int8_size < first.int8_size:
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
                and other.int8_size <= candidate.int8_size
            )
            strictly_better = (
                other_comet > candidate_comet
                or other.p50_ms < candidate.p50_ms
                or other.peak_rss_mb < candidate.peak_rss_mb
                or other.int8_size < candidate.int8_size
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    return sorted(
        frontier,
        key=lambda item: (
            -(item.comet if item.comet is not None else float("-inf")),
            item.p50_ms,
            item.peak_rss_mb,
        ),
    )


def _to_leaderboard_rows(
    board_type: str,
    lane: str,
    route: str,
    evaluations: list[EvaluationRecord],
) -> list[LeaderboardRow]:
    return [
        LeaderboardRow(
            board_type=board_type,
            lane=lane,
            route=route,
            rank=index,
            system_id=evaluation.system_id,
            comet=evaluation.comet if evaluation.comet is not None else float("nan"),
            p50_ms=evaluation.p50_ms,
            peak_rss_mb=evaluation.peak_rss_mb,
            int8_size=evaluation.int8_size,
            recommended=evaluation.recommended,
            eliminated=evaluation.eliminated,
        )
        for index, evaluation in enumerate(evaluations, start=1)
    ]

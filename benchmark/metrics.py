from __future__ import annotations

from collections import defaultdict

from .schemas import EvaluationRecord, MetricsConfig, PredictionRecord, RuntimeSummary


def compute_evaluations(
    predictions: list[PredictionRecord],
    runtime_summaries: list[RuntimeSummary],
    metrics_config: MetricsConfig,
) -> list[EvaluationRecord]:
    summary_by_key = {
        (summary.system_id, summary.route): summary
        for summary in runtime_summaries
    }
    grouped_predictions: dict[tuple[str, str], list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        grouped_predictions[(prediction.system_id, prediction.route)].append(prediction)

    comet_scores_by_key, comet_available = _compute_comet_scores(grouped_predictions, metrics_config)
    evaluations: list[EvaluationRecord] = []

    for key, records in sorted(grouped_predictions.items()):
        hypotheses = [record.translated_text for record in records]
        references = [record.reference_text for record in records]
        summary = summary_by_key[key]

        bleu_score = _compute_bleu(hypotheses, references) if metrics_config.compute_bleu else 0.0
        chrf_pp_score = _compute_chrf_pp(hypotheses, references) if metrics_config.compute_chrf_pp else 0.0
        must_preserve_rate = _compute_must_preserve_rate(records)
        comet_score = comet_scores_by_key.get(key) if comet_available else None

        evaluations.append(
            EvaluationRecord(
                system_id=summary.system_id,
                lane=summary.lane,
                route=summary.route,
                artifact_ids=summary.artifact_ids,
                model_ids=summary.model_ids,
                licenses=summary.licenses,
                int8_size=summary.int8_size,
                cold_start_ms=summary.cold_start_ms,
                p50_ms=summary.p50_ms,
                p95_ms=summary.p95_ms,
                total_duration_s=summary.total_duration_s,
                tokens_per_second=summary.tokens_per_second,
                peak_rss_mb=summary.peak_rss_mb,
                empty_output_count=summary.empty_output_count,
                error_count=summary.error_count,
                comet=comet_score,
                chrf_pp=chrf_pp_score,
                bleu=bleu_score,
                must_preserve_rate=must_preserve_rate,
                eliminated=False,
                elimination_reasons=[],
            )
        )

    return evaluations


def _compute_comet_scores(
    grouped_predictions: dict[tuple[str, str], list[PredictionRecord]],
    metrics_config: MetricsConfig,
) -> tuple[dict[tuple[str, str], float], bool]:
    try:
        from comet import download_model, load_from_checkpoint
    except ModuleNotFoundError as exc:
        print(
            "[report] COMET is not installed. Continuing without COMET; quality ranking and elimination will use the remaining available metrics."
        )
        return {}, False

    model_path = download_model(metrics_config.comet_model)
    model = load_from_checkpoint(model_path)
    scores_by_key: dict[tuple[str, str], float] = {}

    for key, records in grouped_predictions.items():
        inputs = [
            {
                "src": record.source_text,
                "mt": record.translated_text,
                "ref": record.reference_text,
            }
            for record in records
        ]
        prediction_output = model.predict(
            inputs,
            batch_size=metrics_config.comet_batch_size,
            gpus=0,
            progress_bar=False,
        )
        scores_by_key[key] = _normalize_comet_output(prediction_output)

    return scores_by_key, True


def _normalize_comet_output(prediction_output) -> float:
    if isinstance(prediction_output, tuple):
        if len(prediction_output) >= 2 and isinstance(prediction_output[1], (int, float)):
            return float(prediction_output[1])
        if len(prediction_output) >= 1:
            scores = prediction_output[0]
            return float(sum(scores) / len(scores)) if scores else 0.0

    if hasattr(prediction_output, "system_score"):
        return float(prediction_output.system_score)

    if hasattr(prediction_output, "scores"):
        scores = list(prediction_output.scores)
        return float(sum(scores) / len(scores)) if scores else 0.0

    raise ValueError("Unsupported COMET prediction output format.")


def _compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    from sacrebleu import corpus_bleu

    return float(corpus_bleu(hypotheses, [references]).score)


def _compute_chrf_pp(hypotheses: list[str], references: list[str]) -> float:
    from sacrebleu import corpus_chrf

    return float(corpus_chrf(hypotheses, [references], word_order=2).score)


def _compute_must_preserve_rate(records: list[PredictionRecord]) -> float:
    total_terms = 0
    matched_terms = 0

    for record in records:
        normalized_prediction = record.translated_text.casefold()
        for term in record.must_preserve:
            total_terms += 1
            if term.casefold() in normalized_prediction:
                matched_terms += 1

    return (matched_terms / total_terms) if total_terms else 0.0

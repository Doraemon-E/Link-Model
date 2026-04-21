#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT_STEM = "hy-mt1.5-1.8b-coreml-int8"
DEFAULT_COREML_ROOT = ROOT_DIR / "models/translation/converted/coreml-int8"
DEFAULT_MLX_MODEL_DIR = ROOT_DIR / "models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx"

DEFAULT_RESULTS_DIR = ROOT_DIR / "test/results"
DEFAULT_RESULTS_JSON = "coreml-model-benchmark-results.json"
DEFAULT_SUMMARY_JSON = "coreml-model-benchmark-summary.json"
DEFAULT_SUMMARY_MD = "coreml-model-benchmark-summary.md"

COREML_TEST_SCRIPT = ROOT_DIR / "test/test_coreml_model.py"
MLX_TEST_SCRIPT = ROOT_DIR / "test/test_mlx_model.py"


def _run_command(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _load_manifest(coreml_dir: Path) -> dict[str, object] | None:
    manifest_path = coreml_dir / "translation-manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _parse_variants(raw: str) -> list[str]:
    variants = [token.strip() for token in raw.split(",") if token.strip()]
    if not variants:
        raise RuntimeError("no variants provided")

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped


def _discover_coreml_variants(
    *,
    coreml_root_dir: Path,
    artifact_stem: str,
    include_legacy: bool,
) -> list[str]:
    prefix = f"{artifact_stem}-"
    variants: list[str] = []
    if not coreml_root_dir.is_dir():
        return variants

    for child in sorted(coreml_root_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "translation-manifest.json").is_file():
            continue
        if child.name.startswith(prefix):
            variants.append(child.name[len(prefix) :])
            continue
        if include_legacy and child.name == "hy-mt1.5-1.8b-coreml":
            variants.append("__legacy_fixed__")
    return variants


def _resolve_coreml_dir(
    *,
    variant: str,
    coreml_root_dir: Path,
    artifact_stem: str,
) -> Path:
    if variant == "__legacy_fixed__":
        legacy_dir = coreml_root_dir / "hy-mt1.5-1.8b-coreml"
        if legacy_dir.is_dir():
            return legacy_dir
        raise RuntimeError(f"missing legacy coreml dir: {legacy_dir}")

    coreml_dir = coreml_root_dir / f"{artifact_stem}-{variant}"
    if coreml_dir.is_dir():
        return coreml_dir
    raise RuntimeError(f"missing coreml dir for variant={variant}: {coreml_dir}")


def _resolve_coreml_context_length(
    *,
    coreml_dir: Path,
    requested_context_length: int,
) -> int:
    manifest = _load_manifest(coreml_dir)
    if manifest is None:
        return requested_context_length

    raw = manifest.get("contextLength")
    if isinstance(raw, int) and raw > 0:
        return min(requested_context_length, raw)
    return requested_context_length


def _run_single_coreml_model(
    *,
    variant: str,
    coreml_dir: Path,
    compute_unit: str,
    source_text: str,
    target_language: str,
    max_new_tokens: int,
    context_length: int,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(COREML_TEST_SCRIPT),
        "--coreml-dir",
        str(coreml_dir),
        "--variant",
        variant,
        "--compute-unit",
        compute_unit,
        "--source-text",
        source_text,
        "--target-language",
        target_language,
        "--max-new-tokens",
        str(max_new_tokens),
        "--context-length",
        str(context_length),
        "--materialize-compiled-model",
        "--json-only",
        "--no-print-generated-text",
    ]
    output = _run_command(cmd, ROOT_DIR)
    parsed = json.loads(output)
    inference = parsed.get("inference", {})
    if not isinstance(inference, dict):
        inference = {}

    return {
        "status": "passed",
        "runtime": "coreml",
        "variant": variant,
        "coreml_dir": str(coreml_dir),
        "model_path": parsed.get("model_path"),
        "inference": inference,
    }


def _run_single_mlx_model(
    *,
    model_dir: Path,
    source_text: str,
    target_language: str,
    max_new_tokens: int,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(MLX_TEST_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--source-text",
        source_text,
        "--target-language",
        target_language,
        "--max-new-tokens",
        str(max_new_tokens),
        "--json-only",
        "--no-print-generated-text",
    ]
    output = _run_command(cmd, ROOT_DIR)
    parsed = json.loads(output)
    inference = parsed.get("inference", {})
    if not isinstance(inference, dict):
        inference = {}

    return {
        "status": "passed",
        "runtime": "mlx",
        "variant": model_dir.name,
        "mlx_model_dir": str(model_dir),
        "model_path": str(model_dir),
        "inference": inference,
    }


def _row_label(row: dict[str, object]) -> str:
    runtime = str(row.get("runtime", "unknown"))
    variant = str(row.get("variant", "unknown"))
    return f"{runtime}:{variant}"


def _model_row_for_summary(row: dict[str, object]) -> dict[str, object]:
    inference = row.get("inference", {})
    if not isinstance(inference, dict):
        inference = {}
    return {
        "runtime": row.get("runtime"),
        "variant": row.get("variant"),
        "label": _row_label(row),
        "coreml_dir": row.get("coreml_dir"),
        "mlx_model_dir": row.get("mlx_model_dir"),
        "model_path": row.get("model_path"),
        "stateful_runtime": inference.get("stateful_runtime"),
        "load_seconds": inference.get("load_seconds"),
        "translation_total_seconds": inference.get("translation_total_seconds"),
        "generate_seconds": inference.get("generate_seconds"),
        "memory_rss_before_load_mb": inference.get("memory_rss_before_load_mb"),
        "memory_rss_after_load_mb": inference.get("memory_rss_after_load_mb"),
        "memory_rss_delta_load_mb": inference.get("memory_rss_delta_load_mb"),
        "memory_rss_after_generate_mb": inference.get("memory_rss_after_generate_mb"),
        "prompt_tokens": inference.get("prompt_tokens"),
        "output_tokens": inference.get("output_tokens"),
    }


def _extract_metric(row: dict[str, object], metric: str) -> float | None:
    inference = row.get("inference", {})
    if not isinstance(inference, dict):
        return None
    value = inference.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pick_best(rows: list[dict[str, object]], metric: str) -> dict[str, object] | None:
    candidates: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        value = _extract_metric(row, metric)
        if value is None:
            continue
        candidates.append((value, row))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    best_value, best_row = candidates[0]
    return {
        "runtime": best_row.get("runtime"),
        "variant": best_row.get("variant"),
        "label": _row_label(best_row),
        "value": round(best_value, 6),
    }


def _build_rankings(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    metrics = {
        "by_translation_total_seconds": "translation_total_seconds",
        "by_load_seconds": "load_seconds",
        "by_memory_rss_after_load_mb": "memory_rss_after_load_mb",
    }
    rankings: dict[str, list[dict[str, object]]] = {}

    for rank_name, metric in metrics.items():
        ranked: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            value = _extract_metric(row, metric)
            if value is None:
                continue
            ranked.append((value, row))
        ranked.sort(key=lambda item: item[0])
        rankings[rank_name] = [
            {
                "rank": idx + 1,
                "runtime": item[1].get("runtime"),
                "variant": item[1].get("variant"),
                "label": _row_label(item[1]),
                metric: round(item[0], 6),
            }
            for idx, item in enumerate(ranked)
        ]

    return rankings


def _write_markdown_summary(path: Path, summary: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append("# Runtime Model Benchmark Summary")
    lines.append("")
    lines.append(f"- generated_at: `{summary.get('generated_at')}`")
    lines.append(f"- coreml_compute_unit: `{summary.get('coreml_compute_unit')}`")
    lines.append(f"- include_mlx: `{summary.get('include_mlx')}`")
    lines.append(f"- total_models: `{summary.get('total_models')}`")
    lines.append(f"- success_count: `{summary.get('success_count')}`")
    lines.append(f"- failure_count: `{summary.get('failure_count')}`")
    lines.append("")

    lines.append("## Best")
    lines.append("")
    lines.append(f"- fastest_load: `{summary.get('fastest_load')}`")
    lines.append(f"- fastest_translation: `{summary.get('fastest_translation')}`")
    lines.append(f"- lowest_load_memory: `{summary.get('lowest_load_memory')}`")
    lines.append("")

    lines.append("## Per Model")
    lines.append("")
    lines.append(
        "| runtime | variant | load_seconds | translation_total_seconds | "
        "memory_before_load_mb | memory_after_load_mb | memory_delta_load_mb |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    model_rows = summary.get("model_rows", [])
    if isinstance(model_rows, list):
        for row in model_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {runtime} | {variant} | {load_seconds} | {translation_total_seconds} | "
                "{memory_rss_before_load_mb} | {memory_rss_after_load_mb} | "
                "{memory_rss_delta_load_mb} |".format(
                    runtime=row.get("runtime"),
                    variant=row.get("variant"),
                    load_seconds=row.get("load_seconds"),
                    translation_total_seconds=row.get("translation_total_seconds"),
                    memory_rss_before_load_mb=row.get("memory_rss_before_load_mb"),
                    memory_rss_after_load_mb=row.get("memory_rss_after_load_mb"),
                    memory_rss_delta_load_mb=row.get("memory_rss_delta_load_mb"),
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark CoreML variants and MLX model in one report."
    )
    parser.add_argument("--coreml-root-dir", type=Path, default=DEFAULT_COREML_ROOT)
    parser.add_argument("--artifact-stem", default=DEFAULT_ARTIFACT_STEM)
    parser.add_argument(
        "--variants",
        default="auto",
        help="comma-separated coreml variants (e.g. nocache,cache,cache-opt) or 'auto'",
    )
    parser.add_argument(
        "--include-legacy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include legacy fixed directory hy-mt1.5-1.8b-coreml when variants=auto",
    )
    parser.add_argument(
        "--include-mlx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include MLX model benchmark in the same report",
    )
    parser.add_argument("--mlx-model-dir", type=Path, default=DEFAULT_MLX_MODEL_DIR)
    parser.add_argument("--compute-unit", default="cpuAndNeuralEngine")
    parser.add_argument("--source-text", default="今天下午三点半在5A会议室开会。")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue benchmarking other models when one model fails",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.context_length <= 0:
        raise RuntimeError(f"context-length must be > 0, got {args.context_length}")

    coreml_root_dir = args.coreml_root_dir.expanduser().resolve()
    mlx_model_dir = args.mlx_model_dir.expanduser().resolve()

    if args.variants == "auto":
        variants = _discover_coreml_variants(
            coreml_root_dir=coreml_root_dir,
            artifact_stem=args.artifact_stem,
            include_legacy=args.include_legacy,
        )
    else:
        variants = _parse_variants(args.variants)

    if not variants and not args.include_mlx:
        raise RuntimeError(
            "no benchmark targets found: coreml variants empty and include_mlx is false"
        )

    rows: list[dict[str, object]] = []

    for variant in variants:
        coreml_dir = _resolve_coreml_dir(
            variant=variant,
            coreml_root_dir=coreml_root_dir,
            artifact_stem=args.artifact_stem,
        )
        effective_context = _resolve_coreml_context_length(
            coreml_dir=coreml_dir,
            requested_context_length=args.context_length,
        )
        try:
            row = _run_single_coreml_model(
                variant=variant,
                coreml_dir=coreml_dir,
                compute_unit=args.compute_unit,
                source_text=args.source_text,
                target_language=args.target_language,
                max_new_tokens=args.max_new_tokens,
                context_length=effective_context,
            )
            row["effective_context_length"] = effective_context
            rows.append(row)
        except Exception as exc:
            error_row = {
                "status": "failed",
                "runtime": "coreml",
                "variant": variant,
                "coreml_dir": str(coreml_dir),
                "effective_context_length": effective_context,
                "error": str(exc),
            }
            rows.append(error_row)
            if not args.continue_on_error:
                raise

    if args.include_mlx:
        if not mlx_model_dir.is_dir():
            error_row = {
                "status": "failed",
                "runtime": "mlx",
                "variant": mlx_model_dir.name,
                "mlx_model_dir": str(mlx_model_dir),
                "error": f"mlx model dir does not exist: {mlx_model_dir}",
            }
            rows.append(error_row)
            if not args.continue_on_error:
                raise RuntimeError(error_row["error"])
        else:
            try:
                rows.append(
                    _run_single_mlx_model(
                        model_dir=mlx_model_dir,
                        source_text=args.source_text,
                        target_language=args.target_language,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            except Exception as exc:
                error_row = {
                    "status": "failed",
                    "runtime": "mlx",
                    "variant": mlx_model_dir.name,
                    "mlx_model_dir": str(mlx_model_dir),
                    "error": str(exc),
                }
                rows.append(error_row)
                if not args.continue_on_error:
                    raise

    timestamp = datetime.now(timezone.utc).isoformat()
    success_rows = [row for row in rows if row.get("status") == "passed"]
    failure_rows = [row for row in rows if row.get("status") != "passed"]

    summary = {
        "status": "completed" if not failure_rows else "completed_with_failures",
        "generated_at": timestamp,
        "coreml_compute_unit": args.compute_unit,
        "include_mlx": args.include_mlx,
        "coreml_root_dir": str(coreml_root_dir),
        "mlx_model_dir": str(mlx_model_dir),
        "artifact_stem": args.artifact_stem,
        "coreml_variants": variants,
        "total_models": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "fastest_load": _pick_best(success_rows, "load_seconds"),
        "fastest_translation": _pick_best(success_rows, "translation_total_seconds"),
        "lowest_load_memory": _pick_best(success_rows, "memory_rss_after_load_mb"),
        "model_rows": [_model_row_for_summary(row) for row in success_rows],
        "failures": failure_rows,
        "rankings": _build_rankings(success_rows),
    }

    full_results = {
        "status": summary["status"],
        "generated_at": timestamp,
        "config": {
            "coreml_compute_unit": args.compute_unit,
            "source_text": args.source_text,
            "target_language": args.target_language,
            "max_new_tokens": args.max_new_tokens,
            "requested_context_length": args.context_length,
            "coreml_root_dir": str(coreml_root_dir),
            "artifact_stem": args.artifact_stem,
            "coreml_variants": variants,
            "include_mlx": args.include_mlx,
            "mlx_model_dir": str(mlx_model_dir),
        },
        "rows": rows,
        "summary": summary,
    }

    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = results_dir / args.results_json
    summary_json_path = results_dir / args.summary_json
    summary_md_path = results_dir / args.summary_md

    results_json_path.write_text(
        json.dumps(full_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_summary(summary_md_path, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "results_json": str(results_json_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "total_models": summary["total_models"],
                "success_count": summary["success_count"],
                "failure_count": summary["failure_count"],
                "fastest_load": summary["fastest_load"],
                "fastest_translation": summary["fastest_translation"],
                "lowest_load_memory": summary["lowest_load_memory"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

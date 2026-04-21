#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_STEM = "hy-mt1.5-1.8b-coreml-int8"
DEFAULT_COREML_ROOT = ROOT_DIR / "models/translation/converted/coreml-int8"
TEST_SCRIPT = ROOT_DIR / "test/test_coreml_model.py"
CONVERT_SCRIPT = ROOT_DIR / "covert_to_coreml.py"


def _run_command(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _ensure_profile_built(
    *,
    profile: str,
    context_length: int,
    optimized_context_length: int,
    force_rebuild: bool,
) -> None:
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--profile",
        profile,
        "--context-length",
        str(context_length),
        "--optimized-context-length",
        str(optimized_context_length),
    ]
    if force_rebuild:
        cmd.extend(["--force-rebuild"])
    _run_command(cmd, ROOT_DIR)


def _run_variant_test(
    *,
    variant: str,
    coreml_root_dir: Path,
    artifact_stem: str,
    compute_unit: str,
    source_text: str,
    target_language: str,
    max_new_tokens: int,
    context_length: int,
) -> dict[str, object]:
    coreml_dir = coreml_root_dir / f"{artifact_stem}-{variant}"
    if not coreml_dir.is_dir():
        raise RuntimeError(f"missing variant directory: {coreml_dir}")

    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
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
        "--json-only",
        "--no-print-generated-text",
    ]
    output = _run_command(cmd, ROOT_DIR)
    parsed = json.loads(output)
    inference = parsed.get("inference", {})
    return {
        "variant": variant,
        "model_path": parsed.get("model_path"),
        "inference": inference,
    }


def _compute_comparison(rows: list[dict[str, object]]) -> dict[str, dict[str, float | None]]:
    by_variant = {row["variant"]: row for row in rows}
    baseline = by_variant.get("nocache")
    if baseline is None:
        return {}

    baseline_inf = baseline["inference"]
    baseline_total = baseline_inf.get("translation_total_seconds")
    baseline_load_mem = baseline_inf.get("memory_rss_after_load_mb")
    baseline_generate_mem = baseline_inf.get("memory_rss_after_generate_mb")

    comparisons: dict[str, dict[str, float | None]] = {}
    for row in rows:
        variant = row["variant"]
        inf = row["inference"]
        total = inf.get("translation_total_seconds")
        load_mem = inf.get("memory_rss_after_load_mb")
        generate_mem = inf.get("memory_rss_after_generate_mb")

        speedup = None
        if isinstance(baseline_total, (int, float)) and isinstance(total, (int, float)) and total > 0:
            speedup = round(float(baseline_total) / float(total), 3)

        load_mem_saved_mb = None
        if isinstance(baseline_load_mem, (int, float)) and isinstance(load_mem, (int, float)):
            load_mem_saved_mb = round(float(baseline_load_mem) - float(load_mem), 3)

        generate_mem_saved_mb = None
        if isinstance(baseline_generate_mem, (int, float)) and isinstance(
            generate_mem, (int, float)
        ):
            generate_mem_saved_mb = round(
                float(baseline_generate_mem) - float(generate_mem), 3
            )

        comparisons[variant] = {
            "vs_nocache_translation_speedup": speedup,
            "vs_nocache_load_memory_saved_mb": load_mem_saved_mb,
            "vs_nocache_after_generate_memory_saved_mb": generate_mem_saved_mb,
        }

    return comparisons


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark no-cache/cache/cache-opt CoreML variants.")
    parser.add_argument(
        "--variants",
        default="nocache,cache,cache-opt",
        help="comma-separated variant names",
    )
    parser.add_argument("--coreml-root-dir", type=Path, default=DEFAULT_COREML_ROOT)
    parser.add_argument("--artifact-stem", default=DEFAULT_ARTIFACT_STEM)
    parser.add_argument("--compute-unit", default="cpuAndNeuralEngine")
    parser.add_argument("--source-text", default="今天下午三点半在5A会议室开会。")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--optimized-context-length", type=int, default=128)
    parser.add_argument(
        "--build-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build missing variants before benchmarking",
    )
    parser.add_argument(
        "--force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="force rebuild variants before benchmarking",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise RuntimeError("no variants provided")

    for variant in variants:
        variant_dir = args.coreml_root_dir / f"{args.artifact_stem}-{variant}"
        should_build = args.force_rebuild or (args.build_missing and not variant_dir.is_dir())
        if should_build:
            _ensure_profile_built(
                profile=variant,
                context_length=args.context_length,
                optimized_context_length=args.optimized_context_length,
                force_rebuild=args.force_rebuild,
            )
        elif not variant_dir.is_dir():
            raise RuntimeError(
                f"missing variant artifact: {variant_dir}. "
                "Run covert_to_coreml.py first or pass --build-missing."
            )

    rows = [
        _run_variant_test(
            variant=variant,
            coreml_root_dir=args.coreml_root_dir,
            artifact_stem=args.artifact_stem,
            compute_unit=args.compute_unit,
            source_text=args.source_text,
            target_language=args.target_language,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
        )
        for variant in variants
    ]

    summary = {
        "status": "completed",
        "compute_unit": args.compute_unit,
        "rows": rows,
        "comparisons": _compute_comparison(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

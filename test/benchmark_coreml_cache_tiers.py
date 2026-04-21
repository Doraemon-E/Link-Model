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
DEFAULT_TIER_CONTEXTS = "256,128,96,64"
DEFAULT_TIER_PREFIX = "cache-c"
DEFAULT_COREML_TMP_ROOT = Path("/private/var/folders/lm/r0xbqry5417fppl8xr6pny9c0000gn/T")

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


def _cleanup_temp_coreml_artifacts(tmp_root: Path) -> dict[str, object]:
    removed_patterns = [
        "hy_mt_w8_from_torch_*.mlmodelc",
        "tmp*.mlmodelc",
        "tmp*.mlpackage",
        "translator-new-validation-store-*",
    ]
    removed_paths: list[str] = []
    if not tmp_root.is_dir():
        return {"tmp_root": str(tmp_root), "removed_count": 0, "removed_paths": []}

    for pattern in removed_patterns:
        for path in tmp_root.glob(pattern):
            if path.is_dir():
                subprocess.run(["/bin/rm", "-rf", str(path)], check=True)
            else:
                path.unlink(missing_ok=True)
            removed_paths.append(str(path))

    return {
        "tmp_root": str(tmp_root),
        "removed_count": len(removed_paths),
        "removed_paths": removed_paths,
    }


def _parse_tier_contexts(raw: str) -> list[int]:
    contexts: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        context = int(token)
        if context <= 0:
            raise ValueError(f"context must be > 0, got {context}")
        if context in seen:
            continue
        seen.add(context)
        contexts.append(context)
    if not contexts:
        raise ValueError("tier contexts cannot be empty")
    return contexts


def _variant_name(prefix: str, context_length: int) -> str:
    return f"{prefix}{context_length}"


def _build_missing_tiers(
    *,
    contexts: list[int],
    prefix: str,
    artifact_stem: str,
    coreml_root_dir: Path,
    force_rebuild: bool,
) -> None:
    missing_contexts: list[int] = []
    for context in contexts:
        variant_dir = coreml_root_dir / f"{artifact_stem}-{_variant_name(prefix, context)}"
        if force_rebuild or not variant_dir.is_dir():
            missing_contexts.append(context)

    if not missing_contexts:
        return

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--profile",
        "cache-tiers",
        "--cache-tier-prefix",
        prefix,
        "--cache-tier-contexts",
        ",".join(str(v) for v in missing_contexts),
        "--decode-only",
    ]
    if force_rebuild:
        cmd.append("--force-rebuild")

    _run_command(cmd, ROOT_DIR)


def _run_tier_test(
    *,
    context_length: int,
    prefix: str,
    artifact_stem: str,
    coreml_root_dir: Path,
    compute_unit: str,
    source_text: str,
    target_language: str,
    max_new_tokens: int,
    prompt_context_length: int,
) -> dict[str, object]:
    variant = _variant_name(prefix, context_length)
    coreml_dir = coreml_root_dir / f"{artifact_stem}-{variant}"
    if not coreml_dir.is_dir():
        raise RuntimeError(f"missing tier directory: {coreml_dir}")

    effective_context = min(context_length, prompt_context_length)
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
        str(effective_context),
        "--json-only",
        "--no-print-generated-text",
    ]
    output = _run_command(cmd, ROOT_DIR)
    parsed = json.loads(output)
    inference = parsed.get("inference", {})
    return {
        "context_length": context_length,
        "variant": variant,
        "model_path": parsed.get("model_path"),
        "inference": inference,
    }


def _build_tier_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    by_ctx = {int(row["context_length"]): row for row in rows}
    baseline_ctx = max(by_ctx.keys())
    baseline = by_ctx[baseline_ctx]
    base_inf = baseline["inference"]
    base_total = base_inf.get("translation_total_seconds")
    base_load_mem = base_inf.get("memory_rss_after_load_mb")
    base_gen_mem = base_inf.get("memory_rss_after_generate_mb")

    table: list[dict[str, object]] = []
    for ctx in sorted(by_ctx.keys(), reverse=True):
        row = by_ctx[ctx]
        inf = row["inference"]

        total = inf.get("translation_total_seconds")
        load_mem = inf.get("memory_rss_after_load_mb")
        gen_mem = inf.get("memory_rss_after_generate_mb")

        speedup = None
        if isinstance(base_total, (int, float)) and isinstance(total, (int, float)) and total > 0:
            speedup = round(float(base_total) / float(total), 3)

        load_saved = None
        if isinstance(base_load_mem, (int, float)) and isinstance(load_mem, (int, float)):
            load_saved = round(float(base_load_mem) - float(load_mem), 3)

        gen_saved = None
        if isinstance(base_gen_mem, (int, float)) and isinstance(gen_mem, (int, float)):
            gen_saved = round(float(base_gen_mem) - float(gen_mem), 3)

        table.append(
            {
                "context_length": ctx,
                "variant": row["variant"],
                "stateful_runtime": inf.get("stateful_runtime"),
                "load_seconds": inf.get("load_seconds"),
                "prefill_seconds": inf.get("prefill_seconds"),
                "first_token_latency_seconds": inf.get("first_token_latency_seconds"),
                "generate_seconds": inf.get("generate_seconds"),
                "translation_total_seconds": total,
                "memory_rss_after_load_mb": load_mem,
                "memory_rss_after_generate_mb": gen_mem,
                "speedup_vs_max_context": speedup,
                "load_memory_saved_vs_max_context_mb": load_saved,
                "after_generate_memory_saved_vs_max_context_mb": gen_saved,
            }
        )

    return table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark aggressive memory-saving cache tiers by context length."
    )
    parser.add_argument("--tier-contexts", default=DEFAULT_TIER_CONTEXTS)
    parser.add_argument("--tier-prefix", default=DEFAULT_TIER_PREFIX)
    parser.add_argument("--artifact-stem", default=DEFAULT_ARTIFACT_STEM)
    parser.add_argument("--coreml-root-dir", type=Path, default=DEFAULT_COREML_ROOT)
    parser.add_argument("--compute-unit", default="cpuAndNeuralEngine")
    parser.add_argument("--source-text", default="今天下午三点半在5A会议室开会。")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-context-length",
        type=int,
        default=256,
        help="prompt truncation length used by test runner",
    )
    parser.add_argument(
        "--build-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cleanup-temp-coremlc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="delete stale temporary CoreML compile directories under /private/var/folders/*/T",
    )
    parser.add_argument("--coreml-tmp-root", type=Path, default=DEFAULT_COREML_TMP_ROOT)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    contexts = _parse_tier_contexts(args.tier_contexts)

    cleanup_summary = None
    if args.cleanup_temp_coremlc:
        cleanup_summary = _cleanup_temp_coreml_artifacts(args.coreml_tmp_root)

    if args.build_missing or args.force_rebuild:
        _build_missing_tiers(
            contexts=contexts,
            prefix=args.tier_prefix,
            artifact_stem=args.artifact_stem,
            coreml_root_dir=args.coreml_root_dir,
            force_rebuild=args.force_rebuild,
        )

    rows = [
        _run_tier_test(
            context_length=context,
            prefix=args.tier_prefix,
            artifact_stem=args.artifact_stem,
            coreml_root_dir=args.coreml_root_dir,
            compute_unit=args.compute_unit,
            source_text=args.source_text,
            target_language=args.target_language,
            max_new_tokens=args.max_new_tokens,
            prompt_context_length=args.prompt_context_length,
        )
        for context in contexts
    ]

    summary = {
        "status": "completed",
        "compute_unit": args.compute_unit,
        "cleanup_summary": cleanup_summary,
        "tier_contexts": contexts,
        "baseline_context": max(contexts),
        "rows": rows,
        "tier_table": _build_tier_table(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler


def _apply_mlx_compat_patch() -> None:
    """
    mlx_lm 内部仍在调用已弃用的 mx.metal.device_info。
    这里在运行时替换为 mx.device_info，避免告警并兼容未来版本。
    """
    metal = getattr(mx, "metal", None)
    modern_device_info = getattr(mx, "device_info", None)
    if metal is None or modern_device_info is None:
        return
    try:
        setattr(metal, "device_info", modern_device_info)
    except Exception:
        pass


MODEL_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
TARGET_LANGUAGE = "English"
SOURCE_TEXT = "今天下午三点半在5A会议室开会。"
MAX_NEW_TOKENS = 64
PRINT_GENERATED_TEXT = True
TOP_K = 20
TOP_P = 0.6
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.05


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024.0 * 1024.0), 3)


def _current_rss_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    try:
        # ps rss is in KB
        return int(output) * 1024
    except ValueError:
        return None


def _build_prompt(target_language: str, source_text: str) -> str:
    return (
        f"将以下文本翻译为{target_language}，注意字词、语法、语义语境，"
        "并只输出翻译结果：\n\n"
        f"{source_text}"
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    _apply_mlx_compat_patch()
    model_dir = args.model_dir.expanduser().resolve()
    prompt = _build_prompt(args.target_language, args.source_text)

    translation_start = time.perf_counter()
    memory_before_load = _current_rss_bytes()
    load_start = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    load_elapsed = time.perf_counter() - load_start
    memory_after_load = _current_rss_bytes()

    if getattr(tokenizer, "chat_template", None) is not None and hasattr(
        tokenizer, "apply_chat_template"
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt_for_generate = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    else:
        prompt_for_generate = prompt

    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    logits_processors = make_logits_processors(repetition_penalty=args.repetition_penalty)

    gen_start = time.perf_counter()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_for_generate,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    gen_elapsed = time.perf_counter() - gen_start
    translation_total_elapsed = time.perf_counter() - translation_start
    memory_after_generate = _current_rss_bytes()

    text = output if isinstance(output, str) else str(output)
    if isinstance(prompt_for_generate, list):
        prompt_tokens = len(prompt_for_generate)
    elif isinstance(prompt_for_generate, str) and hasattr(tokenizer, "encode"):
        prompt_tokens = len(tokenizer.encode(prompt_for_generate))
    else:
        prompt_tokens = None
    output_tokens = (
        len(tokenizer.encode(text)) if hasattr(tokenizer, "encode") else None
    )

    memory_delta_load = None
    if memory_before_load is not None and memory_after_load is not None:
        memory_delta_load = memory_after_load - memory_before_load

    return {
        "status": "passed" if text.strip() else "failed",
        "load_seconds": round(load_elapsed, 3),
        "generate_seconds": round(gen_elapsed, 3),
        "translation_total_seconds": round(translation_total_elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "memory_rss_before_load_bytes": memory_before_load,
        "memory_rss_after_load_bytes": memory_after_load,
        "memory_rss_after_generate_bytes": memory_after_generate,
        "memory_rss_delta_load_bytes": memory_delta_load,
        "memory_rss_before_load_mb": _bytes_to_mb(memory_before_load),
        "memory_rss_after_load_mb": _bytes_to_mb(memory_after_load),
        "memory_rss_after_generate_mb": _bytes_to_mb(memory_after_generate),
        "memory_rss_delta_load_mb": _bytes_to_mb(memory_delta_load),
        "output_text": text,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MLX translation smoke/benchmark.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--target-language", default=TARGET_LANGUAGE)
    parser.add_argument("--source-text", default=SOURCE_TEXT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument(
        "--print-generated-text",
        action=argparse.BooleanOptionalAction,
        default=PRINT_GENERATED_TEXT,
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="print only summary json (no generated text trailer)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    inference = _run(args)
    summary = {
        "status": "passed" if inference["status"] == "passed" else "failed",
        "runtime": "mlx",
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "inference": {
            key: value for key, value in inference.items() if key != "output_text"
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    should_print_text = args.print_generated_text and not args.json_only
    if should_print_text:
        print("\n===== GENERATED TEXT =====")
        print(inference["output_text"])

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

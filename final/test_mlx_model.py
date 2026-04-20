#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from mlx_lm import generate, load


MODEL_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
TARGET_LANGUAGE = "English"
SOURCE_TEXT = "今天下午三点半在5A会议室开会。"
PROMPT = (
    f"将以下文本翻译为{TARGET_LANGUAGE}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
    f"{SOURCE_TEXT}"
)
MAX_TOKENS = 64
PRINT_GENERATED_TEXT = True


def _load_mlx_and_generate(
    model_dir: Path,
    prompt: str,
    max_tokens: int,
) -> dict[str, object]:

    load_start = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    load_elapsed = time.perf_counter() - load_start

    gen_start = time.perf_counter()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    gen_elapsed = time.perf_counter() - gen_start

    text = output if isinstance(output, str) else str(output)
    prompt_tokens = (
        len(tokenizer.encode(prompt)) if hasattr(tokenizer, "encode") else None
    )
    output_tokens = (
        len(tokenizer.encode(text)) if hasattr(tokenizer, "encode") else None
    )

    return {
        "status": "passed" if text.strip() else "failed",
        "load_seconds": round(load_elapsed, 3),
        "generate_seconds": round(gen_elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_text": text,
    }


def main() -> int:
    model_dir = MODEL_DIR.expanduser().resolve()

    try:
        inference = _load_mlx_and_generate(
            model_dir=model_dir,
            prompt=PROMPT,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics only.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "inference_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "model_dir": str(model_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    summary = {
        "status": "passed" if inference["status"] == "passed" else "failed",
        "model_dir": str(model_dir),
        "inference": {
            key: value for key, value in inference.items() if key != "output_text"
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if PRINT_GENERATED_TEXT:
        print("\n===== GENERATED TEXT =====")
        print(inference["output_text"])

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

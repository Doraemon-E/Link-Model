from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranslationResult:
    text: str
    output_token_count: int


@dataclass(frozen=True)
class LoadedRuntime:
    runtime: object
    cold_start_ms: float


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def start(self) -> None:
        import psutil

        process = psutil.Process()
        self._peak_rss_bytes = process.memory_info().rss

        def run() -> None:
            while not self._stop_event.is_set():
                try:
                    rss_bytes = process.memory_info().rss
                    self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)
                except Exception:
                    pass
                self._stop_event.wait(self.interval_seconds)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)
        return float(self._peak_rss_bytes) / 1_048_576


class TranslationRuntime:
    def __init__(self, model_dir: Path) -> None:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix())
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_dir.as_posix(),
            local_files_only=True,
            provider="CPUExecutionProvider",
            use_merged=False,
        )

    @classmethod
    def load(cls, model_dir: Path) -> LoadedRuntime:
        started_at = time.perf_counter()
        runtime = cls(model_dir)
        return LoadedRuntime(runtime=runtime, cold_start_ms=(time.perf_counter() - started_at) * 1000)

    def translate(
        self,
        text: str,
        *,
        source_lang: str | None,
        target_lang: str | None,
        batch_size: int,
        do_sample: bool,
        num_beams: int,
        max_new_tokens: int,
    ) -> TranslationResult:
        tokenizer = self.tokenizer
        if source_lang is not None and hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = source_lang

        encoded = tokenizer(text, return_tensors="pt")
        if batch_size != 1:
            raise ValueError("Only batch_size=1 is supported.")

        generate_kwargs = {
            "do_sample": do_sample,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
        }
        if target_lang is not None and hasattr(tokenizer, "get_lang_id"):
            generate_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(target_lang)

        outputs = self.model.generate(**encoded, **generate_kwargs)
        translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        try:
            output_token_count = int(outputs.shape[-1])
        except Exception:
            output_token_count = len(outputs[0])

        return TranslationResult(text=translated_text, output_token_count=output_token_count)


class GGUFTranslationRuntime:
    def __init__(self, model_dir: Path) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=resolve_single_gguf_model(model_dir).as_posix(),
            n_ctx=4096,
            n_gpu_layers=0,
            n_threads=max(os.cpu_count() or 1, 1),
            verbose=False,
            seed=0,
        )

    @classmethod
    def load(cls, model_dir: Path) -> LoadedRuntime:
        started_at = time.perf_counter()
        runtime = cls(model_dir)
        return LoadedRuntime(runtime=runtime, cold_start_ms=(time.perf_counter() - started_at) * 1000)

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        batch_size: int,
        do_sample: bool,
        num_beams: int,
        max_new_tokens: int,
    ) -> TranslationResult:
        if batch_size != 1:
            raise ValueError("Only batch_size=1 is supported.")
        if num_beams != 1:
            raise ValueError("GGUF runtime only supports num_beams=1.")

        prompt = build_translation_prompt(text, source_lang=source_lang, target_lang=target_lang)
        response = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.7 if do_sample else 0.0,
            top_p=0.6,
            top_k=20,
            repeat_penalty=1.05,
            echo=False,
        )
        translated_text = sanitize_translation_output(
            str(response["choices"][0]["text"]),
            source_text=text,
        )
        usage = response.get("usage", {})
        output_token_count = int(usage.get("completion_tokens") or 0)
        if output_token_count <= 0:
            output_token_count = len(self.model.tokenize(translated_text.encode("utf-8"), add_bos=False))
        return TranslationResult(text=translated_text, output_token_count=output_token_count)


def build_translation_prompt(text: str, *, source_lang: str, target_lang: str) -> str:
    normalized_source = source_lang.strip().lower()
    normalized_target = target_lang.strip().lower()
    language_names = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "ru": "Russian",
        "es": "Spanish",
        "it": "Italian",
    }

    if normalized_source == "zh" or normalized_target == "zh":
        target_name_cn = {
            "zh": "中文",
            "en": "英语",
            "ja": "日语",
            "ko": "韩语",
            "fr": "法语",
            "de": "德语",
            "ru": "俄语",
            "es": "西班牙语",
            "it": "意大利语",
        }.get(normalized_target, normalized_target)
        source_name_cn = {
            "zh": "中文",
            "en": "英语",
            "ja": "日语",
            "ko": "韩语",
            "fr": "法语",
            "de": "德语",
            "ru": "俄语",
            "es": "西班牙语",
            "it": "意大利语",
        }.get(normalized_source, normalized_source)
        return f"""你是一个翻译引擎。
任务：把给定文本从{source_name_cn}翻译成{target_name_cn}。

规则：
- 只做翻译，不要解释。
- 只在 <translation> 和 </translation> 标签内输出译文。
- 不要输出原文，不要添加引号、编号或备注。
- 保留原文语气与换行。
- 输出完成后立刻写 </translation>。

<source_text>
{text}
</source_text>
<translation>
"""

    source_name = language_names.get(normalized_source, source_lang)
    target_name = language_names.get(normalized_target, target_lang)
    return f"""You are a translation engine.
Translate the source text from {source_name} to {target_name}.

Rules:
- Return the translation only.
- Output only inside <translation> and </translation>.
- Do not explain, annotate, quote, or repeat the source text.
- Preserve tone and line breaks where possible.
- Finish by writing </translation>.

<source_text>
{text}
</source_text>
<translation>
"""


def sanitize_translation_output(output: str, *, source_text: str) -> str:
    normalized = output.replace("\r\n", "\n").strip()

    if "<translation>" in normalized:
        normalized = normalized.split("<translation>", 1)[1].strip()

    cut_positions = [
        normalized.find(marker)
        for marker in (
            "</translation>",
            "<source_text>",
            "Source text:",
            "Rules:",
            "规则：",
            "任务：",
        )
        if marker in normalized
    ]
    if cut_positions:
        normalized = normalized[: min(cut_positions)].strip()

    for prefix in ("Translation:", "translation:", "译文：", "译文:", "答案：", "答案:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    if len(source_text.strip()) <= 12:
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if lines:
            normalized = lines[0]

    if normalized.startswith('"') and normalized.endswith('"') and len(normalized) >= 2:
        normalized = normalized[1:-1].strip()

    return normalized


def resolve_single_gguf_model(model_dir: Path) -> Path:
    matches = sorted(path for path in model_dir.rglob("*.gguf") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one GGUF model under {model_dir}, found {matches or 'none'}")
    return matches[0]


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return ordered[lower_index] + ((ordered[upper_index] - ordered[lower_index]) * fraction)

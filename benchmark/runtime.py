from __future__ import annotations

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

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix())
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_dir.as_posix(),
            local_files_only=True,
            provider="CPUExecutionProvider",
            use_merged=False,
        )

    @classmethod
    def load(cls, model_dir: Path) -> LoadedRuntime:
        start = time.perf_counter()
        runtime = cls(model_dir)
        cold_start_ms = (time.perf_counter() - start) * 1000
        return LoadedRuntime(runtime=runtime, cold_start_ms=cold_start_ms)

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
            raise ValueError("Only batch_size=1 is supported in benchmark v1.")

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

        return TranslationResult(
            text=translated_text,
            output_token_count=output_token_count,
        )


class CausalTranslationRuntime:
    def __init__(self, model_dir: Path) -> None:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix())
        self.model = ORTModelForCausalLM.from_pretrained(
            model_dir.as_posix(),
            local_files_only=True,
            provider="CPUExecutionProvider",
        )

    @classmethod
    def load(cls, model_dir: Path) -> LoadedRuntime:
        start = time.perf_counter()
        runtime = cls(model_dir)
        cold_start_ms = (time.perf_counter() - start) * 1000
        return LoadedRuntime(runtime=runtime, cold_start_ms=cold_start_ms)

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
            raise ValueError("Only batch_size=1 is supported in benchmark v1.")

        tokenizer = self.tokenizer
        prompt = build_translation_prompt(text, source_lang=source_lang, target_lang=target_lang)

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
            if not isinstance(encoded, dict):
                encoded = {
                    "input_ids": encoded,
                    "attention_mask": encoded.ne(tokenizer.pad_token_id or 0).long(),
                }
        else:
            encoded = tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **encoded,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        input_length = encoded["input_ids"].shape[-1]
        generated_tokens = outputs[:, input_length:]
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        try:
            output_token_count = int(generated_tokens.shape[-1])
        except Exception:
            output_token_count = len(generated_tokens[0])

        return TranslationResult(
            text=translated_text,
            output_token_count=output_token_count,
        )


def build_translation_prompt(text: str, *, source_lang: str, target_lang: str) -> str:
    source_name = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
    }.get(source_lang, source_lang)
    target_name = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
    }.get(target_lang, target_lang)
    return (
        f"Translate the following {source_name} text into {target_name}. "
        "Return only the translation, with no explanation.\n\n"
        f"{text}"
    )


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

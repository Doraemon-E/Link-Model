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
            top_p=1.0,
            top_k=1,
            echo=False,
        )
        translated_text = str(response["choices"][0]["text"]).strip()
        usage = response.get("usage", {})
        output_token_count = int(usage.get("completion_tokens") or 0)
        if output_token_count <= 0:
            output_token_count = len(self.model.tokenize(translated_text.encode("utf-8"), add_bos=False))
        return TranslationResult(text=translated_text, output_token_count=output_token_count)


def build_translation_prompt(text: str, *, source_lang: str, target_lang: str) -> str:
    source_name = {"zh": "Chinese", "en": "English", "ja": "Japanese"}.get(source_lang, source_lang)
    target_name = {"zh": "Chinese", "en": "English", "ja": "Japanese"}.get(target_lang, target_lang)
    return (
        f"Translate the following {source_name} text into {target_name}. "
        "Return only the translation, with no explanation.\n\n"
        f"{text}"
    )


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

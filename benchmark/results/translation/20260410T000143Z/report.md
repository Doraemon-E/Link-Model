# Translation Benchmark Report

## 模型与许可证
- `llm` / `zh-en` / `hy-mt1.5-1.8b-gguf-q4km`: models=tencent/HY-MT1.5-1.8B-GGUF; licenses=unknown
- `llm` / `zh-ja` / `hy-mt1.5-1.8b-gguf-q4km`: models=tencent/HY-MT1.5-1.8B-GGUF; licenses=unknown

## 指标说明
- COMET: unavailable in this run (dependency not installed); report falls back to chrF++ / BLEU / mustPreserve for quality comparison.

## 量化产物
- `llm` / `zh-en` / `hy-mt1.5-1.8b-gguf-q4km`: quantized_size=1133080512 bytes
- `llm` / `zh-ja` / `hy-mt1.5-1.8b-gguf-q4km`: quantized_size=1133080512 bytes

## 质量排行
### llm / zh-en
- #1 `hy-mt1.5-1.8b-gguf-q4km` comet=N/A
### llm / zh-ja
- #1 `hy-mt1.5-1.8b-gguf-q4km` comet=N/A

## 性能排行
### llm / zh-en
- #1 `hy-mt1.5-1.8b-gguf-q4km` p50_ms=1308.48, peak_rss_mb=2380.78, quantized_size=1133080512
### llm / zh-ja
- #1 `hy-mt1.5-1.8b-gguf-q4km` p50_ms=1692.09, peak_rss_mb=2360.92, quantized_size=1133080512

## 淘汰原因
### llm / zh-en
- `hy-mt1.5-1.8b-gguf-q4km` eliminated: must_preserve_rate < 0.85
### llm / zh-ja
- `hy-mt1.5-1.8b-gguf-q4km` eliminated: must_preserve_rate < 0.85

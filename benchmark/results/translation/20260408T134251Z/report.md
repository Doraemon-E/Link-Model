# Translation Benchmark Report

## 模型与许可证
- `llm` / `zh-en` / `granite-3-1-2b-instruct`: models=ibm-granite/granite-3.1-2b-instruct; licenses=unknown
- `llm` / `zh-ja` / `granite-3-1-2b-instruct`: models=ibm-granite/granite-3.1-2b-instruct; licenses=unknown

## 指标说明
- COMET: unavailable in this run (dependency not installed); report falls back to chrF++ / BLEU / mustPreserve for quality comparison.

## 量化产物
- `llm` / `zh-en` / `granite-3-1-2b-instruct`: int8_size=2642260109 bytes
- `llm` / `zh-ja` / `granite-3-1-2b-instruct`: int8_size=2642260109 bytes

## 质量排行
### llm / zh-en
- #1 `granite-3-1-2b-instruct` comet=N/A
### llm / zh-ja
- #1 `granite-3-1-2b-instruct` comet=N/A

## 性能排行
### llm / zh-en
- #1 `granite-3-1-2b-instruct` p50_ms=2220.97, peak_rss_mb=2900.00, int8_size=2642260109
### llm / zh-ja
- #1 `granite-3-1-2b-instruct` p50_ms=4034.27, peak_rss_mb=2942.44, int8_size=2642260109

## 淘汰原因
### llm / zh-en
- `granite-3-1-2b-instruct` eliminated: must_preserve_rate < 0.85
### llm / zh-ja
- `granite-3-1-2b-instruct` eliminated: must_preserve_rate < 0.85

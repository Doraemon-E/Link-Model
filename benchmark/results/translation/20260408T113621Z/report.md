# Translation Benchmark Report

## 模型与许可证
- `seq2seq` / `zh-en` / `m2m100-418m`: models=facebook/m2m100_418M; licenses=mit
- `seq2seq` / `zh-en` / `marian-direct`: models=Helsinki-NLP/opus-mt-zh-en; licenses=cc-by-4.0
- `seq2seq` / `zh-en` / `marian-pivot`: models=Helsinki-NLP/opus-mt-zh-en; licenses=cc-by-4.0
- `seq2seq` / `zh-ja` / `m2m100-418m`: models=facebook/m2m100_418M; licenses=mit
- `seq2seq` / `zh-ja` / `marian-direct`: models=Helsinki-NLP/opus-mt-tc-big-zh-ja; licenses=cc-by-4.0
- `seq2seq` / `zh-ja` / `marian-pivot`: models=Helsinki-NLP/opus-mt-zh-en, Helsinki-NLP/opus-mt-en-jap; licenses=cc-by-4.0, apache-2.0

## 指标说明
- COMET: unavailable in this run (dependency not installed); report falls back to chrF++ / BLEU / mustPreserve for quality comparison.

## 量化产物
- `seq2seq` / `zh-en` / `m2m100-418m`: int8_size=1206789818 bytes
- `seq2seq` / `zh-en` / `marian-direct`: int8_size=238549357 bytes
- `seq2seq` / `zh-en` / `marian-pivot`: int8_size=238549357 bytes
- `seq2seq` / `zh-ja` / `m2m100-418m`: int8_size=1206789818 bytes
- `seq2seq` / `zh-ja` / `marian-direct`: int8_size=435966729 bytes
- `seq2seq` / `zh-ja` / `marian-pivot`: int8_size=428919314 bytes

## 质量排行
### seq2seq / zh-en
- #1 `marian-direct` comet=N/A
- #2 `marian-pivot` comet=N/A
- #3 `m2m100-418m` comet=N/A
### seq2seq / zh-ja
- #1 `m2m100-418m` comet=N/A
- #2 `marian-pivot` comet=N/A
- #3 `marian-direct` comet=N/A

## 性能排行
### seq2seq / zh-en
- #1 `marian-pivot` p50_ms=82.26, peak_rss_mb=713.02, int8_size=238549357
- #2 `marian-direct` p50_ms=83.26, peak_rss_mb=901.42, int8_size=238549357
- #3 `m2m100-418m` p50_ms=300.70, peak_rss_mb=2329.05, int8_size=1206789818
### seq2seq / zh-ja
- #1 `marian-direct` p50_ms=110.72, peak_rss_mb=1230.53, int8_size=435966729
- #2 `marian-pivot` p50_ms=197.00, peak_rss_mb=938.42, int8_size=428919314
- #3 `m2m100-418m` p50_ms=321.24, peak_rss_mb=2259.50, int8_size=1206789818

## 淘汰原因
### seq2seq / zh-en
- `m2m100-418m` eliminated: must_preserve_rate < 0.85
- `marian-direct` eliminated: must_preserve_rate < 0.85
- `marian-pivot` eliminated: must_preserve_rate < 0.85
### seq2seq / zh-ja
- `m2m100-418m` eliminated: must_preserve_rate < 0.85
- `marian-direct` eliminated: empty_output_count > 0, must_preserve_rate < 0.85
- `marian-pivot` eliminated: must_preserve_rate < 0.85

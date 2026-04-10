# Translation Benchmark Report

## 模型与许可证
- `zh-en` / `hy-mt1.5-1.8b-gguf-q4km`: models=tencent/HY-MT1.5-1.8B-GGUF; licenses=unknown
- `zh-en` / `opus-mt-direct`: models=Helsinki-NLP/opus-mt-zh-en; licenses=cc-by-4.0
- `zh-ja` / `hy-mt1.5-1.8b-gguf-q4km`: models=tencent/HY-MT1.5-1.8B-GGUF; licenses=unknown
- `zh-ja` / `opus-mt-pivot-via-en`: models=Helsinki-NLP/opus-mt-zh-en, Helsinki-NLP/opus-mt-en-jap; licenses=cc-by-4.0, apache-2.0

## 指标说明
- COMET: unavailable in this run; gate is skipped and the report still shows BLEU / chrF++ / mustPreserve.

## Route Summary
### zh-en

| system | strategy | mustPreserve hits/total (%) | BLEU | chrF++ | COMET | cold_start_ms | p50_ms | p95_ms | tokens_per_second | peak_rss_mb | quantized_size | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hy-mt1.5-1.8b-gguf-q4km` | `direct` | `345/440 (78.41%)` | 26.06 | 56.26 | N/A | 1175.59 | 1220.12 | 4974.10 | 34.60 | 2568.45 | 1133081156 | eliminated |
| `opus-mt-direct` | `direct` | `290/440 (65.91%)` | 21.75 | 46.44 | N/A | 315.25 | 78.51 | 331.53 | 453.47 | 858.72 | 238549975 | eliminated |

### zh-ja

| system | strategy | mustPreserve hits/total (%) | BLEU | chrF++ | COMET | cold_start_ms | p50_ms | p95_ms | tokens_per_second | peak_rss_mb | quantized_size | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hy-mt1.5-1.8b-gguf-q4km` | `direct` | `305/445 (68.54%)` | 0.00 | 30.69 | N/A | 1246.16 | 1498.81 | 6877.96 | 36.00 | 2472.56 | 1133081156 | eliminated |
| `opus-mt-pivot-via-en` | `pivot_via_en` | `5/445 (1.12%)` | 0.00 | 5.55 | N/A | 572.23 | 205.17 | 514.16 | 452.84 | 1028.61 | 428920550 | eliminated |

## Gate Results
### zh-en
- `hy-mt1.5-1.8b-gguf-q4km`: must_preserve_rate 78.41% < 85.00%
- `opus-mt-direct`: must_preserve_rate 65.91% < 85.00%

### zh-ja
- `hy-mt1.5-1.8b-gguf-q4km`: must_preserve_rate 68.54% < 85.00%
- `opus-mt-pivot-via-en`: must_preserve_rate 1.12% < 85.00%

## Pivot Hop Summary
### zh-ja / opus-mt-pivot-via-en
- `hop1` avg_latency_ms=136.42, avg_output_tokens=61.33
- `hop2` avg_latency_ms=113.10, avg_output_tokens=51.67

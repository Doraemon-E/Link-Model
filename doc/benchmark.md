# 翻译模型 Benchmark Pipeline（Python 离线版）

## Summary

- 目标是在不改 `Link` app 运行时的前提下，在 `Link-Model` 内新增一套离线 benchmark pipeline，对 `zh->en` 和 `zh->ja` 做 Marian 基线与商业友好多语模型的对比。
- v1 模型池固定为两个系统：`marian-pivot` 与 `m2m100-418m`。
- `marian-pivot` 固定用 `Helsinki-NLP/opus-mt-zh-en` 做 `zh->en`，用 `Helsinki-NLP/opus-mt-zh-en` + `Helsinki-NLP/opus-mt-en-jap` 两跳做 `zh->ja`。
- `m2m100-418m` 固定用 `facebook/m2m100_418M` 直接跑 `zh->en` 和 `zh->ja`。
- 默认不纳入 [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)，因为当前模型卡标记为 `cc-by-nc-4.0`，与你“只看可商用”这一约束冲突。
- benchmark 默认统一走 `下载 -> 量化产物准备 -> CPU 推理`；ONNX 系统使用 `ONNX INT8`，官方 GGUF 系统直接复用 Hugging Face 上的 `Q4_K_M` 量化产物。
- 当前新增专用配置 [translation-hy-mt-official-gguf.yaml](/Users/doracmon/Code/Aura/link-model/benchmark/configs/translation-hy-mt-official-gguf.yaml)，固定测试 `tencent/HY-MT1.5-1.8B-GGUF` 的 `Q4_K_M` 版本。
- 解码参数固定为 `batch_size=1`、`do_sample=False`、`num_beams=1`、`greedy decode`、`max_new_tokens=256`。

## Key Changes

- 新增独立 benchmark 包，不复用现有面向 app 打包的 Marian manifest 逻辑；现有 `Link-Model/transform` 和 `Link` Swift 运行时保持不变。
- benchmark 保留通用配置 [translation.yaml](/Users/doracmon/Code/Aura/link-model/benchmark/configs/translation.yaml)，并新增官方 GGUF 专用配置 [translation-hy-mt-official-gguf.yaml](/Users/doracmon/Code/Aura/link-model/benchmark/configs/translation-hy-mt-official-gguf.yaml)。
- 配置字段固定包含 `artifacts_root`、`results_root`、`systems`、`routes`、`quantization`、`decode`、`metrics`、`decision_policy`。
- artifact 目录固定为 `Link-Model/models/benchmark_translation/{downloaded,exported,quantized}/<artifact_id>`。
- 结果目录固定为 `Link-Model/benchmark/results/translation/<timestamp>/`。
- v1 官方语料直接读取 [translation_performance_corpus.json](/Users/doracmon/Code/Aura/Link/linkTests/Fixtures/translation_performance_corpus.json)，保持现有 schema，不复制数据，不改字段。
- `marian-pivot` 的 `zh->ja` 必须保存 hop1(`zh->en`) 与 hop2(`en->ja`) 的中间输出与分段时延，最终报告单独展示。
- benchmark CLI 固定提供三个命令：`prepare`、`run`、`report`。

## Pipeline

- `prepare` 会按 artifact 类型分支：ONNX 系统执行 `下载 -> 导出 -> INT8 量化`；官方 GGUF 系统执行 `列远端文件 -> 下载 Q4_K_M GGUF -> stage 到 quantized 目录`，并写出 `artifact-manifest.json`。
- Marian 与 M2M100 都统一使用 Optimum `ORTModelForSeq2SeqLM` 做 ONNX 导出。
- 量化默认走 ONNX Runtime dynamic quantization；这是 transformer 模型的默认策略。
- 每个 artifact manifest 必须记录 `model_id`、`revision`、`license`、`source_langs`、`target_langs`、`fp32_size`、`quantized_size`、`quantization_ratio`、`quantization_format`、`quantization_source`、`export_success`、`quantize_success`。
- `run` 固定只跑 `zh->en` 与 `zh->ja` 两个方向。
- 正式 benchmark 前会先拿 1 条 `short` 中文语料做 `zh->en` 与 `zh->ja` smoke，若快速生成失败则直接终止本轮正式跑数。
- 每个系统每个方向先 warmup 2 次，再对 15 条语料做 5 轮正式测量。
- 每次运行都记录 `cold_start_ms`、`sentence_latency_ms`、`p50_ms`、`p95_ms`、`total_duration_s`、`tokens_per_second`、`peak_rss_mb`、`empty_output_count`、`error_count`。
- `judge` 阶段不做单一混合总分，而是拆成质量榜、效率榜、Pareto 前沿三份结果。
- 质量主指标固定为 `COMET`。
- 质量辅指标固定为 `chrF++`、`BLEU`、`mustPreserve 命中率`。
- `mustPreserve` 默认不是“纯原样 substring”判断，而是“标准化 + 软匹配”：会处理大小写、全半角数字、常见日文字形差异、英文词形变化与简单否定形式，尽量减少明显的误判。
- 自动淘汰规则固定为：`error_count > 0`、或 `empty_output_count > 0`、或 `mustPreserve_rate < 85%`、或 `COMET < Marian基线 - 0.03`。
- 自动推荐规则固定为：先按上面的质量门槛筛模型，剩余模型里选 `p50 latency` 最低者；若 10% 内打平，依次比较 `peak RSS` 与 `quantized_size`。
- `report` 固定产出 `predictions.jsonl`、`metrics.json`、`leaderboard.csv`、`report.md`。
- `report.md` 固定分为模型与许可证、量化产物、质量排行、性能排行、淘汰原因五部分。

## Test Plan

- 校验语料完整性：总数必须为 15，`short/medium/long` 各 5 条。
- 校验系统注册表：`marian-pivot` 必须展开为两个 artifact，`m2m100-418m` 必须正确映射 `zh -> en/ja` 的语言码与 `forced_bos_token_id`。
- 校验 prepare 产物：ONNX artifact 必须同时存在原始模型、ONNX 导出结果、INT8 量化结果，且 `quantized_size < fp32_size`；官方 GGUF artifact 必须能唯一匹配并成功 stage `Q4_K_M` 文件，且 manifest 写出 `quantized_size`、源文件名与校验信息。
- 校验 run 结果：每条语料都必须有原文、译文、耗时、异常状态；`zh->ja` 的 Marian 结果必须含 hop1/hop2 两段记录。
- 校验 metrics：COMET、chrF、BLEU、mustPreserve 统计必须能对同一批预测稳定复算。
- 校验 report：凡是被淘汰的模型，报告中必须显示明确原因，不能静默消失。

## Assumptions

- v1 是“离线研究型 benchmark”，不改 iOS/Swift 运行时，不产出可直接被 app 安装的 translation package。
- 当前 15 条语料足够做产品 smoke benchmark，但不足以做论文级统计结论；报告会明确写成“工程选型对比”，不是“通用 MT SOTA 排名”。
- 商业友好是硬约束，因此默认榜单只锁 Marian 与 [M2M100 418M](https://huggingface.co/facebook/m2m100_418M)。
- 后续若要补更稳的公开评测集，优先接 [FLORES-200](https://huggingface.co/datasets/facebook/flores) 的 `zho_Hans-eng_Latn` 与 `zho_Hans-jpn_Jpan` 子集，但这不属于 v1 必做项。

## Industry References

- ONNX Runtime 官方建议对 transformer 优先使用 dynamic quantization：[Quantize ONNX Models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。
- MT 质量主指标采用 COMET：[Unbabel COMET](https://github.com/Unbabel/COMET)。
- 可复现的 BLEU/chrF 采用 sacreBLEU：[sacreBLEU](https://github.com/mjpost/sacrebleu)。

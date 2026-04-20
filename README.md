# link-model

HY-MT CoreML 8-bit（ANE 优先）转换与打包工具。

## 目标

本仓库当前方案用于把 `tencent/HY-MT1.5-1.8B` 转成可在 CoreML 运行的 stateful 模型包，并默认走 `W8A8`（8-bit activation + 8-bit weight）路径，优先使用 ANE，尽量降低 CPU/GPU 占用。

## 当前默认策略

1. 导出 stateful FP16 `mlprogram` 到 `Intermediate/causal_lm-fp16.mlpackage`。
2. 默认执行 activation 量化（A8，需校准样本）。
3. 默认执行 weight 量化（W8，`linear_symmetric` + `per_channel`）。
4. 保存量化后 `causal_lm.mlpackage`。
5. 编译为 `Compiled/causal_lm.mlmodelc`。
6. 写入 `translation-manifest.json`，默认 `preferredComputeUnits = ["cpuAndNeuralEngine", "cpuOnly", "all"]`。
7. 打包 zip，并输出结构化报告 JSON。
8. 默认执行 Python smoke（默认 compute unit: `cpuAndNeuralEngine`）。
9. 默认执行 NE 覆盖率验证（`MLComputePlan`，默认阈值 `0.70`）。

## 快速开始

```bash
cd /Users/doracmon/Code/Aura/link-model
uv sync
```

可选：下载模型快照（原始 + MLX 8bit 对照）。

```bash
./.venv/bin/python tools/download_hy_mt_models.py
```

执行默认方案（W8A8 + ANE 优先）：

```bash
./.venv/bin/python tools/convert_hy_mt_to_coreml.py
```

## 常用命令

生成 W8 回退包（仅权重量化）：

```bash
./.venv/bin/python tools/convert_hy_mt_to_coreml.py \
  --quantization-mode w8 \
  --output-dir models/translation/converted/coreml-w8/hy-mt1.5-1.8b-coreml \
  --packaged-zip models/translation/packaged/hy-mt1.5-1.8b-coreml-w8.zip \
  --report-path models/translation/reports/hy-mt1.5-1.8b-coreml-w8-report.json
```

仅导出 FP16（不做量化）：

```bash
./.venv/bin/python tools/convert_hy_mt_to_coreml.py --quantization-mode none
```

跳过 NE 覆盖率验证（仅在环境受限或调试时使用）：

```bash
./.venv/bin/python tools/convert_hy_mt_to_coreml.py --no-verify-ne-plan
```

指定 compute unit smoke：

```bash
./.venv/bin/python tools/convert_hy_mt_to_coreml.py --smoke-compute-unit cpuOnly
```

## 关键参数

`tools/convert_hy_mt_to_coreml.py` 关键参数如下：

- `--quantization-mode {w8a8,w8,none}`：默认 `w8a8`
- `--quantization-granularity {per_channel,per_tensor,per_block}`：默认 `per_channel`
- `--quantization-dtype {int8,uint8}`：默认 `int8`
- `--activation-calibration-jsonl`：默认 `tools/calibration/hy_mt_coreml_calibration.jsonl`
- `--calibration-op-group-size`：默认 `32`
- `--compute-units`：默认 `cpuAndNeuralEngine,cpuOnly,all`
- `--smoke-compute-unit`：默认 `cpuAndNeuralEngine`
- `--verify-ne-plan / --no-verify-ne-plan`：默认开启
- `--minimum-ne-coverage`：默认 `0.70`
- `--run-python-smoke / --no-run-python-smoke`：默认开启

## 校准数据

默认校准文件为：

- `tools/calibration/hy_mt_coreml_calibration.jsonl`

每行一个 JSON，字段必须包含：

- `route`（要求覆盖：`zh-en`、`zh-ja`、`zh-ko`、`zh-zh`）
- `target_language`
- `source_text`

示例：

```json
{"route":"zh-en","target_language":"English","source_text":"今天下午 3:30 在 5A 会议室同步 v1.5.8 发布计划。"}
```

## 输出结构

默认输出目录：

- `models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml`

主要产物：

- `Intermediate/causal_lm-fp16.mlpackage`
- `causal_lm.mlpackage`
- `Compiled/causal_lm.mlmodelc`
- `translation-manifest.json`
- tokenizer/config 运行时文件

默认打包输出：

- `models/translation/packaged/hy-mt1.5-1.8b-coreml-int8.zip`

默认报告输出：

- `models/translation/reports/hy-mt1.5-1.8b-coreml-int8-report.json`

## 报告字段

报告包含以下重点字段：

- `status`
- `quantization_mode`
- `smoke_compute_unit`
- `calibration_samples`
- `artifact_sizes`
- `compute_plan_summary`
- `steps`
- `python_smoke`

## 验收建议

1. `status == "completed"`。
2. `python_smoke.status == "passed"` 且输出非空。
3. `compute_plan_summary.status == "passed"`。
4. `compute_plan_summary.neural_engine_preferred_ratio >= minimum_ne_coverage`。
5. 对比 FP16 基线报告，观察包体积、加载内存、首 token 延迟、总时延。


# link-model

HY-MT CoreML 8-bit（ANE 可运行优先）转换与打包工具。

## 目标

本仓库当前方案用于把 `tencent/HY-MT1.5-1.8B` 转成可在 CoreML 运行的 8-bit 权重量化模型包，并默认启用 stateful KV cache，优先保证 `cpuAndNeuralEngine` 可执行。

## 当前默认策略

1. 在 Torch 侧执行 Linear 权重 PTQ（默认 `W8/G64/affine`）。
2. 导出 CoreML `mlprogram`（默认 stateful cache 路径，decode-only）。
3. 将导出的 `.mlpackage` 预编译为 `.mlmodelc`（默认开启）。
4. 写入 `translation-manifest.json` 并复制 tokenizer/config 运行时文件（manifest 默认指向 `.mlmodelc`）。
5. 打包 zip（带目录父层级），并默认把 `.mlmodelc` 一并打进包内。
6. 通过 `test/test_coreml_model.py` 使用 `cpuAndNeuralEngine` 做 smoke。

默认采用固定命名目录，不再按时间戳生成。可通过 `--profile` 直接导出不同变体。

## 快速开始

```bash
cd /Users/doracmon/Code/Aura/link-model
uv sync
```

可选：下载模型快照（原始 + MLX 8bit 对照）。

```bash
./.venv/bin/python tools/download_hy_mt_models.py
```

执行默认方案（`cache` 变体，W8 + stateful cache）：

```bash
./.venv/bin/python covert_to_coreml.py --profile cache
```

## 常用命令

生成 `nocache` 变体（仅权重量化，无 state cache）：

```bash
./.venv/bin/python covert_to_coreml.py --profile nocache
```

生成优化版 cache（更小 context，默认 `128`）：

```bash
./.venv/bin/python covert_to_coreml.py --profile cache-opt
```

一次生成全部三类模型（固定命名，可复用）：

```bash
./.venv/bin/python covert_to_coreml.py --profile all
```

生成更激进的 cache 分档（同样 stateful cache，不同 context）：

```bash
./.venv/bin/python covert_to_coreml.py \
  --profile cache-tiers \
  --cache-tier-contexts 256,192,128,96,64
```

对比测试三类模型（加载内存/整体耗时）：

```bash
./.venv/bin/python test/benchmark_coreml_variants.py
```

输出 cache 分档基准表（速度/内存）：

```bash
./.venv/bin/python test/benchmark_coreml_cache_tiers.py \
  --tier-contexts 256,192,128,96,64 \
  --build-missing
```

一次测试当前支持的全部 CoreML 模型（逐个测试）+ MLX 模型，并把原始结果+汇总写入文件：

```bash
./.venv/bin/python test/benchmark_coreml_all_models.py
```

说明：脚本会在每个 CoreML 模型目录内自动物化 `hy_mt_w8_from_torch.mlmodelc`，并把 `translation-manifest.json` 切换到该路径。

仅测试 CoreML（不含 MLX）：

```bash
./.venv/bin/python test/benchmark_coreml_all_models.py --no-include-mlx
```

默认写入：

- `test/results/coreml-model-benchmark-results.json`
- `test/results/coreml-model-benchmark-summary.json`
- `test/results/coreml-model-benchmark-summary.md`

如果遇到 CoreML 编译缓存占满空间，可加：

```bash
./.venv/bin/python test/benchmark_coreml_cache_tiers.py \
  --tier-contexts 256,192,128,96,64 \
  --cleanup-temp-coremlc
```

## 关键参数

`covert_to_coreml.py` 关键参数如下：

- `--profile {cache,nocache,cache-opt,cache-tiers,all}`：默认 `cache`
- `--context-length`：默认 `256`
- `--optimized-context-length`：默认 `128`（用于 `cache-opt`）
- `--cache-tier-contexts`：默认 `256,192,128,96,64`（用于 `cache-tiers`）
- `--cache-tier-prefix`：默认 `cache-c`
- `--q-bits`：默认 `8`
- `--q-group-size`：默认 `64`
- `--q-mode {affine,symmetric}`：默认 `affine`
- `--decode-only / --no-decode-only`：默认 `--decode-only`
- `--compile-mlmodelc / --no-compile-mlmodelc`：默认 `--compile-mlmodelc`
- `--package-include-mlmodelc / --no-package-include-mlmodelc`：默认 `--package-include-mlmodelc`
- `--force-rebuild / --no-force-rebuild`：默认不强制重建（命中固定目录时复用）

## 输出结构

默认输出目录：

- `models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml-int8-nocache`
- `models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml-int8-cache`
- `models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml-int8-cache-opt`

主要产物：

- `hy_mt_w8_from_torch.mlpackage`
- `hy_mt_w8_from_torch.mlmodelc`（默认运行时加载目标）
- `translation-manifest.json`
- tokenizer/config 运行时文件

对应打包产物：

- `models/translation/packaged/hy-mt1.5-1.8b-coreml-int8-nocache.zip`
- `models/translation/packaged/hy-mt1.5-1.8b-coreml-int8-cache.zip`
- `models/translation/packaged/hy-mt1.5-1.8b-coreml-int8-cache-opt.zip`

默认会在 zip 内携带 `.mlmodelc`（即使 manifest 指向 `.mlpackage`），便于 App 导入后直接优先加载预编译模型，避免首次再编译。

## 对比指标

`test/test_coreml_model.py` 和 `test/benchmark_coreml_variants.py` 当前会输出以下关键指标：

- `memory_rss_before_load_mb`
- `memory_rss_after_load_mb`
- `memory_rss_delta_load_mb`
- `translation_total_seconds`
- `load_seconds`
- `prefill_seconds`
- `first_token_latency_seconds`
- `generate_seconds`

`test/benchmark_coreml_cache_tiers.py` 额外输出分档对比字段：

- `speedup_vs_max_context`
- `load_memory_saved_vs_max_context_mb`
- `after_generate_memory_saved_vs_max_context_mb`

## 验收建议

1. `status == "completed"`。
2. `stateful_runtime == true`（cache / cache-opt）。
3. `memory_rss_delta_load_mb` 与 `translation_total_seconds` 能在三类模型间做可比对比。

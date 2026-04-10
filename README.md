# link-model

现在项目拆成两组独立 pipeline，并共用一份根配置 [pipelines.yaml](/Users/doracmon/Code/Aura/link-model/pipelines.yaml)：

- `translation`：下载、导出、量化、打包、benchmark、report、catalog
- `speech`：下载、打包、catalog

## 目录布局

translation 资产：

- `models/translation/downloaded/`
- `models/translation/exported/`
- `models/translation/quantized/`
- `models/translation/packaged/`

speech 资产：

- `models/speech/downloaded/`
- `models/speech/packaged/`

benchmark 结果：

- `benchmark/results/translation/<timestamp>/`

## 统一 CLI

```bash
./.venv/bin/python main.py translation prepare
./.venv/bin/python main.py translation benchmark
./.venv/bin/python main.py translation report
./.venv/bin/python main.py translation package
./.venv/bin/python main.py translation catalog
./.venv/bin/python main.py translation all

./.venv/bin/python main.py speech prepare
./.venv/bin/python main.py speech package
./.venv/bin/python main.py speech catalog
./.venv/bin/python main.py speech all
```

## 常用示例

```bash
# 1. 跑完整 translation 链路
./.venv/bin/python main.py translation all

# 2. 只准备模型资产，不跑 benchmark
./.venv/bin/python main.py translation prepare

# 3. 分步跑 translation benchmark
./.venv/bin/python main.py translation benchmark
./.venv/bin/python main.py translation report

# 4. 指定一个固定时间戳，方便重复覆盖同一轮结果
./.venv/bin/python main.py translation benchmark --timestamp 20260410T120000Z
./.venv/bin/python main.py translation report --result-dir benchmark/results/translation/20260410T120000Z

# 5. 只打 translation 包和 catalog
./.venv/bin/python main.py translation package
./.venv/bin/python main.py translation catalog

# 6. 跑完整 speech 链路
./.venv/bin/python main.py speech all

# 7. 只更新 speech catalog
./.venv/bin/python main.py speech catalog
```

如果想先看命令结构，可以直接跑：

```bash
./.venv/bin/python main.py -h
./.venv/bin/python main.py translation -h
./.venv/bin/python main.py speech -h
```

`translation all` 会执行：

1. 下载模型
2. 导出 ONNX
3. 量化
4. 跑 benchmark
5. 生成 report

`speech all` 会执行：

1. 下载语音模型
2. 打包 zip
3. 生成 speech catalog

## 当前 translation benchmark 范围

- `opus-mt-direct`
- `opus-mt-pivot-via-en`
- `hy-mt1.5-1.8b-gguf-q4km`

路由固定为：

- `zh -> en`
- `zh -> ja`

其中 `zh -> ja` 会同时保留：

- `opus-mt-direct`
- `opus-mt-pivot-via-en`
- `hy-mt1.5-1.8b-gguf-q4km`

report 会直接展示每个 system/route 的精确分数，包括：

- `mustPreserve hits/total (%)`
- `BLEU`
- `chrF++`
- `COMET`
- `cold_start_ms / p50_ms / p95_ms`
- `tokens_per_second`
- `peak_rss_mb`
- `quantized_size`
- `status`

淘汰原因会显示实际值和阈值，而不是只显示“不满足 85%”。

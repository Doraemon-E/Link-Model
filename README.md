# link-model

批量量化 `models/` 目录里的 ONNX 模型：

```bash
# 在仓库任意位置执行都默认处理当前项目的 models/
./.venv/bin/python transform/quantize_onnx.py
```

默认会遍历所有 `models/*-onnx/` 目录，并为下面三个文件生成对应的 `*_int8.onnx`：

- `encoder_model.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

常用参数：

```bash
# 重新生成已存在的量化文件
./.venv/bin/python transform/quantize_onnx.py --overwrite

# 只量化 encoder
./.venv/bin/python transform/quantize_onnx.py --file-name encoder_model.onnx

# 指定其他 models 目录
./.venv/bin/python transform/quantize_onnx.py --models-dir ./models
```

把量化后的模型打成单独的 `int8` 目录和 zip 包：

```bash
# 自动扫描当前项目 models/，一次性打包全部已量化模型
./.venv/bin/python transform/package_quantized_onnx.py
```

这个脚本会默认扫描当前项目的 `models/*-onnx/`，把其中已经量化完成的模型复制到新的 `*-onnx-int8/` 目录，重命名回标准文件名，并打成 zip。这样解压后不会和原始 ONNX 混在一起，也能直接作为独立安装包使用。
这个脚本会默认扫描当前项目的 `models/*-onnx/`，把其中已经量化完成的模型复制到新的 `*-onnx-int8/` 目录，重命名回标准文件名，并打成 zip。这样解压后不会和原始 ONNX 混在一起，也能直接作为独立安装包使用。

批量上传 `models/` 目录里的 zip 文件到 Cloudflare R2：

```bash
./.venv/bin/python transform/upload_to_cloudflare_r2.py
```

这个脚本会默认上传当前项目 `models/` 下所有 `*-onnx-int8.zip` 到固定 bucket `link-translation-prod` 的 `link/translation/packages/` 前缀，不需要再传路径或前缀参数。

根据本地 zip 自动生成带 `sha256` 和大小信息的 `translation-catalog.json`：

```bash
./.venv/bin/python transform/generate_translation_catalog.py
```

默认会扫描当前项目 `models/` 下的 `*-onnx-int8.zip`，并更新 `../link/link/Resource/translation-catalog.json`。这样客户端就能拿到真实的校验和、压缩包大小和安装后大小，用于下载校验与断点续传状态展示。

# link-model

翻译模型现在统一走固定的 4 个阶段目录：

- `models/translation/downloaded/`：Hugging Face 原始模型
- `models/translation/exported/`：原始 ONNX 导出结果
- `models/translation/quantized/`：量化后的 ONNX 和运行资源
- `models/translation/packaged/`：最终 zip 包

默认 pipeline 会先尝试把旧的 `models/<local_name>`、`models/<local_name>-onnx`、`models/<local_name>-onnx-int8` 和旧 zip 迁移到新布局，然后继续补齐缺失阶段。

## 运行完整 pipeline

```bash
./.venv/bin/python main.py
```

完整 pipeline 固定按下面顺序执行，不需要传路径参数：

1. 下载模型
2. 导出 ONNX
3. 量化 ONNX
4. 打包 zip

模型清单由 `transform/download_manifest.py` 单独维护，默认只处理清单里的翻译模型。

## 单独运行某个阶段

```bash
./.venv/bin/python transform/downloader.py
./.venv/bin/python transform/trans_to_onnx.py
./.venv/bin/python transform/quantize_onnx.py
./.venv/bin/python transform/package_quantized_onnx.py
```

这些阶段脚本也都只使用默认目录，不需要额外传路径。

## 生成 translation catalog

```bash
./.venv/bin/python transform/generate_translation_catalog.py
```

这个脚本会扫描 `models/translation/packaged/` 下的 `*-onnx-int8.zip`，并更新 `../link/link/Resource/translation-catalog.json`。

## 手动验收建议

```bash
# 1. 跑完整链路
./.venv/bin/python main.py

# 2. 再跑一次，确认已完成阶段会跳过
./.venv/bin/python main.py

# 3. 重新生成 catalog
./.venv/bin/python transform/generate_translation_catalog.py
```

## 语音模型 pipeline

`whisper` 现在单独走 speech pipeline，不参与翻译模型的 ONNX 量化流程。

- `models/speech/downloaded/`：从 Hugging Face 下载的语音模型文件
- `models/speech/packaged/`：最终 zip 包

默认 speech pipeline 也会先把旧的 `models/ggml-base-q5_1.bin` 和 `models/whisper-base-q5_1.zip` 迁移到新布局，再继续补齐缺失阶段。

```bash
./.venv/bin/python speech_main.py
```

这条链路固定执行：

1. 迁移旧 speech 资产
2. 从 Hugging Face 下载 `ggml-base-q5_1.bin`
3. 打成 `whisper-base-q5_1.zip`

也可以单独运行 speech 阶段：

```bash
./.venv/bin/python transform/speech_downloader.py
./.venv/bin/python transform/package_speech_model.py
./.venv/bin/python transform/generate_speech_catalog.py
```

`generate_speech_catalog.py` 会扫描 `models/speech/packaged/`，并更新 `../link/link/Resource/speech-catalog.json`。

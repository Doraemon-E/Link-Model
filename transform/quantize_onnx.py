from __future__ import annotations

import shutil

from onnxruntime.quantization import QuantType, quantize_dynamic

try:
    from .download_manifest import MODEL_SPECS, ModelSpec
    from .paths import (
        QUANTIZED_MODELS_DIR,
        RAW_ONNX_FILE_NAMES,
        ensure_stage_directories,
        exported_model_dir,
        quantized_model_dir,
    )
    from .stage_helpers import (
        copy_regular_files,
        create_temporary_directory,
        is_export_complete,
        is_quantized_complete,
        replace_directory,
    )
    from .translation_manifest import write_translation_manifest
except ImportError:
    from download_manifest import MODEL_SPECS, ModelSpec
    from paths import (
        QUANTIZED_MODELS_DIR,
        RAW_ONNX_FILE_NAMES,
        ensure_stage_directories,
        exported_model_dir,
        quantized_model_dir,
    )
    from stage_helpers import (
        copy_regular_files,
        create_temporary_directory,
        is_export_complete,
        is_quantized_complete,
        replace_directory,
    )
    from translation_manifest import write_translation_manifest


def quantize_model(model_path, output_path) -> None:
    quantize_dynamic(
        model_input=model_path.as_posix(),
        model_output=output_path.as_posix(),
        weight_type=QuantType.QInt8,
    )


def quantize_model_directory(spec: ModelSpec) -> bool:
    source_dir = exported_model_dir(spec.local_name)
    target_dir = quantized_model_dir(spec.local_name)

    if is_quantized_complete(target_dir):
        print(f"跳过已量化目录: {target_dir}")
        return False

    if not is_export_complete(source_dir):
        raise FileNotFoundError(f"未找到可量化的导出模型目录: {source_dir}")

    temp_dir = create_temporary_directory(
        QUANTIZED_MODELS_DIR,
        f"tmp-{spec.local_name}",
    )

    try:
        copy_regular_files(
            source_dir,
            temp_dir,
            exclude_suffixes={".onnx"},
            exclude_names={"translation-manifest.json"},
        )

        for file_name in RAW_ONNX_FILE_NAMES:
            model_path = source_dir / file_name
            if not model_path.exists():
                continue
            output_path = temp_dir / file_name
            print(f"量化中: {model_path} -> {output_path}")
            quantize_model(model_path, output_path)

        write_translation_manifest(spec, temp_dir)
        replace_directory(temp_dir, target_dir)
        return True
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> None:
    ensure_stage_directories()

    for spec in MODEL_SPECS:
        source_dir = exported_model_dir(spec.local_name)
        target_dir = quantized_model_dir(spec.local_name)
        print(f"开始量化 ONNX: {source_dir}")
        quantized = quantize_model_directory(spec)
        if quantized:
            print(f"量化完成: {target_dir}")


if __name__ == "__main__":
    main()

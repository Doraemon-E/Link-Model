from __future__ import annotations

import shutil
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

try:
    from .download_manifest import MODEL_SPECS, ModelSpec
    from .paths import EXPORTED_MODELS_DIR, downloaded_model_dir, ensure_stage_directories, exported_model_dir
    from .stage_helpers import (
        create_temporary_directory,
        is_download_complete,
        is_export_complete,
        replace_directory,
    )
except ImportError:
    from download_manifest import MODEL_SPECS, ModelSpec
    from paths import EXPORTED_MODELS_DIR, downloaded_model_dir, ensure_stage_directories, exported_model_dir
    from stage_helpers import (
        create_temporary_directory,
        is_download_complete,
        is_export_complete,
        replace_directory,
    )


def export_to_onnx(spec: ModelSpec) -> bool:
    model_dir = downloaded_model_dir(spec.local_name)
    onnx_dir = exported_model_dir(spec.local_name)

    if is_export_complete(onnx_dir):
        print(f"跳过已导出的 ONNX: {onnx_dir}")
        return False

    if not is_download_complete(model_dir):
        raise FileNotFoundError(f"未找到可导出的原始模型目录: {model_dir}")

    temp_dir = create_temporary_directory(
        EXPORTED_MODELS_DIR,
        f"tmp-{spec.local_name}",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, export=True)
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        replace_directory(temp_dir, onnx_dir)
        return True
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> None:
    ensure_stage_directories()

    for spec in MODEL_SPECS:
        source_dir = downloaded_model_dir(spec.local_name)
        target_dir = exported_model_dir(spec.local_name)
        print(f"开始导出 ONNX: {source_dir}")
        exported = export_to_onnx(spec)
        if exported:
            print(f"导出完成: {target_dir}")


if __name__ == "__main__":
    main()

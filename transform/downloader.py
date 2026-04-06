from __future__ import annotations

import shutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from .download_manifest import MODEL_SPECS, ModelSpec
    from .paths import DOWNLOADED_MODELS_DIR, downloaded_model_dir, ensure_stage_directories
    from .stage_helpers import (
        create_temporary_directory,
        is_download_complete,
        replace_directory,
    )
except ImportError:
    from download_manifest import MODEL_SPECS, ModelSpec
    from paths import DOWNLOADED_MODELS_DIR, downloaded_model_dir, ensure_stage_directories
    from stage_helpers import (
        create_temporary_directory,
        is_download_complete,
        replace_directory,
    )


def download_model(spec: ModelSpec) -> bool:
    save_dir = downloaded_model_dir(spec.local_name)
    if is_download_complete(save_dir):
        print(f"跳过已下载模型: {save_dir}")
        return False

    temp_dir = create_temporary_directory(
        DOWNLOADED_MODELS_DIR,
        f"tmp-{spec.local_name}",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(spec.model_name)
        tokenizer.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)
        replace_directory(temp_dir, save_dir)
        return True
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> None:
    ensure_stage_directories()

    for spec in MODEL_SPECS:
        print(f"开始下载模型 {spec.language_pair}: {spec.model_name}")
        downloaded = download_model(spec)
        if downloaded:
            print(f"下载完成，已保存到: {downloaded_model_dir(spec.local_name)}")


if __name__ == "__main__":
    main()

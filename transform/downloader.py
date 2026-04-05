from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model_registry import MODEL_SPECS, model_directory_name

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def download_model(model_name: str, save_name: str) -> None:
    save_dir = MODELS_DIR / save_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        language_pair = spec["language_pair"]
        model_name = spec["model_name"]
        save_name = model_directory_name(spec)
        print(f"开始下载模型 {language_pair}: {model_name}")
        download_model(model_name, save_name)
        print(f"下载完成，已保存到: {MODELS_DIR / save_name}")


if __name__ == "__main__":
    main()

from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

from model_registry import MODEL_SPECS, model_directory_name

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [
    REPO_ROOT / "models" / model_directory_name(spec) for spec in MODEL_SPECS
]


def export_to_onnx(model_dir: Path) -> None:
    onnx_dir = model_dir.parent / f"{model_dir.name}-onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, export=True)

    model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)


def main() -> None:
    for model_dir in MODEL_DIRS:
        print(f"开始导出 ONNX: {model_dir}")
        export_to_onnx(model_dir)
        print(f"导出完成: {model_dir.parent / f'{model_dir.name}-onnx'}")


if __name__ == "__main__":
    main()

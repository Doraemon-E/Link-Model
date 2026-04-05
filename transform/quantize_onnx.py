from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

DEFAULT_MODEL_NAMES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-quantize ONNX models in the models directory."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory that contains exported *-onnx model folders.",
    )
    parser.add_argument(
        "--suffix",
        default="_int8",
        help="Suffix inserted before the .onnx extension for quantized files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite quantized files if they already exist.",
    )
    parser.add_argument(
        "--file-name",
        action="append",
        dest="file_names",
        help=(
            "Specific ONNX file name to quantize. "
            "Repeat this option to quantize more than one file."
        ),
    )
    return parser.parse_args()


def resolve_target_files(models_dir: Path, file_names: tuple[str, ...]) -> list[Path]:
    if not models_dir.exists():
        raise FileNotFoundError(f"models directory not found: {models_dir}")

    model_files: list[Path] = []
    for model_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        if not model_dir.name.endswith("-onnx"):
            continue

        for file_name in file_names:
            model_file = model_dir / file_name
            if model_file.exists():
                model_files.append(model_file)

    return model_files


def summarize_onnx_directories(models_dir: Path, file_names: tuple[str, ...]) -> tuple[int, int]:
    onnx_dir_count = 0
    empty_onnx_dir_count = 0

    for model_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        if not model_dir.name.endswith("-onnx"):
            continue

        onnx_dir_count += 1
        matched = any((model_dir / file_name).exists() for file_name in file_names)
        if not matched:
            empty_onnx_dir_count += 1

    return onnx_dir_count, empty_onnx_dir_count


def quantize_model(model_path: Path, output_path: Path) -> None:
    quantize_dynamic(
        model_input=model_path.as_posix(),
        model_output=output_path.as_posix(),
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    args = parse_args()
    file_names = tuple(args.file_names or DEFAULT_MODEL_NAMES)
    models_dir = args.models_dir.resolve()

    model_files = resolve_target_files(models_dir, file_names)
    if not model_files:
        onnx_dir_count, empty_onnx_dir_count = summarize_onnx_directories(
            models_dir, file_names
        )
        if onnx_dir_count == 0:
            print(f"未找到任何 *-onnx 模型目录: {models_dir}")
            return

        if empty_onnx_dir_count == onnx_dir_count:
            print(
                "未找到可量化的 ONNX 文件："
                f" {models_dir} 下共有 {onnx_dir_count} 个 *-onnx 目录，"
                "但都不包含默认的 encoder/decoder ONNX 文件。"
            )
            print("请确认这些目录没有被清空，或先重新导出 ONNX。")
            return

        print(f"未找到可量化的 ONNX 文件: {models_dir}")
        return

    quantized_count = 0
    skipped_count = 0

    for model_path in model_files:
        output_path = model_path.with_name(
            f"{model_path.stem}{args.suffix}{model_path.suffix}"
        )

        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            print(f"跳过已存在文件: {output_path}")
            continue

        print(f"量化中: {model_path} -> {output_path}")
        quantize_model(model_path, output_path)
        quantized_count += 1
        print(f"量化完成: {output_path}")

    print(
        "处理完成："
        f" 量化 {quantized_count} 个文件，"
        f" 跳过 {skipped_count} 个文件。"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

QUANTIZED_FILE_MAP = {
    "encoder_model_int8.onnx": "encoder_model.onnx",
    "decoder_model_int8.onnx": "decoder_model.onnx",
    "decoder_with_past_model_int8.onnx": "decoder_with_past_model.onnx",
}
REQUIRED_QUANTIZED_FILE_NAMES = {
    "encoder_model_int8.onnx",
    "decoder_model_int8.onnx",
}

ASSET_SUFFIXES = {".json", ".model", ".spm"}
ASSET_FILE_NAMES = {"vocab.json"}


def iter_source_model_dirs(models_dir: Path) -> list[Path]:
    model_dirs: list[Path] = []
    for path in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if not path.name.endswith("-onnx"):
            continue
        model_dirs.append(path)
    return model_dirs


def find_missing_quantized_files(model_dir: Path) -> list[str]:
    return [
        source_name
        for source_name in REQUIRED_QUANTIZED_FILE_NAMES
        if not (model_dir / source_name).exists()
    ]


def copy_assets(source_dir: Path, target_dir: Path) -> None:
    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.name in QUANTIZED_FILE_MAP:
            continue
        if source_path.suffix in ASSET_SUFFIXES or source_path.name in ASSET_FILE_NAMES:
            shutil.copy2(source_path, target_dir / source_path.name)


def copy_quantized_models(source_dir: Path, target_dir: Path) -> None:
    for source_name, target_name in QUANTIZED_FILE_MAP.items():
        source_path = source_dir / source_name
        if not source_path.exists():
            continue
        shutil.copy2(source_path, target_dir / target_name)


def package_model(model_dir: Path) -> tuple[Path, Path]:
    package_dir = model_dir.parent / f"{model_dir.name}-int8"
    zip_base = model_dir.parent / package_dir.name

    if package_dir.exists():
        shutil.rmtree(package_dir)

    zip_path = zip_base.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    package_dir.mkdir(parents=True, exist_ok=True)
    copy_assets(model_dir, package_dir)
    copy_quantized_models(model_dir, package_dir)

    shutil.make_archive(zip_base.as_posix(), "zip", root_dir=package_dir.parent, base_dir=package_dir.name)
    return package_dir, zip_path


def main() -> None:
    models_dir = DEFAULT_MODELS_DIR.resolve()
    if not models_dir.exists():
        raise FileNotFoundError(f"models directory not found: {models_dir}")

    source_model_dirs = iter_source_model_dirs(models_dir)
    if not source_model_dirs:
        print(f"未找到可打包的模型目录: {models_dir}")
        return

    packaged_count = 0
    skipped_count = 0

    for model_dir in source_model_dirs:
        missing = find_missing_quantized_files(model_dir)
        if missing:
            skipped_count += 1
            missing_text = ", ".join(missing)
            print(f"跳过未量化完整的目录: {model_dir} 缺少 {missing_text}")
            continue

        package_dir, zip_path = package_model(model_dir)
        print(f"已生成目录: {package_dir}")
        print(f"已生成压缩包: {zip_path}")
        packaged_count += 1

    print(
        "处理完成："
        f" 生成 {packaged_count} 个发布目录和 zip，"
        f" 跳过 {skipped_count} 个未量化完整的目录。"
    )


if __name__ == "__main__":
    main()

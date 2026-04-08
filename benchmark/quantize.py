from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def quantize_exported_model(export_dir: Path, output_dir: Path, weight_type: str = "qint8") -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    resolved_weight_type = _resolve_weight_type(weight_type)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tmp-{output_dir.name}-", dir=output_dir.parent))
    onnx_sources = [path for path in sorted(export_dir.iterdir()) if path.is_file() and path.suffix == ".onnx"]
    source_external_data_files = {
        external_data_file
        for onnx_source in onnx_sources
        for external_data_file in _external_data_filenames(onnx_source)
    }

    try:
        for source_path in sorted(export_dir.iterdir()):
            if not source_path.is_file():
                continue

            if source_path.name in source_external_data_files:
                continue

            destination_path = temp_dir / source_path.name
            if source_path.suffix == ".onnx":
                quantize_dynamic(
                    model_input=source_path.as_posix(),
                    model_output=destination_path.as_posix(),
                    weight_type=resolved_weight_type,
                    use_external_data_format=_uses_external_model_data(source_path),
                )
            else:
                shutil.copy2(source_path, destination_path)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(temp_dir.as_posix(), output_dir.as_posix())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def has_quantized_onnx_payload(export_dir: Path, quantized_dir: Path) -> bool:
    if not quantized_dir.exists():
        return False
    if not any(path.is_file() and path.suffix == ".onnx" for path in quantized_dir.rglob("*")):
        return False
    return not any((quantized_dir / external_data_file).exists() for external_data_file in _export_external_data_files(export_dir))


def _resolve_weight_type(weight_type: str):
    from onnxruntime.quantization import QuantType

    normalized = weight_type.strip().lower()
    if normalized == "qint8":
        return QuantType.QInt8
    if normalized == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"Unsupported quantization weight type: {weight_type}")


def _uses_external_model_data(model_path: Path) -> bool:
    return bool(_external_data_filenames(model_path))


def _export_external_data_files(export_dir: Path) -> set[str]:
    return {
        external_data_file
        for onnx_source in export_dir.iterdir()
        if onnx_source.is_file() and onnx_source.suffix == ".onnx"
        for external_data_file in _external_data_filenames(onnx_source)
    }


def _external_data_filenames(model_path: Path) -> set[str]:
    import onnx

    model = onnx.load_model(model_path.as_posix(), load_external_data=False)
    external_data_files: set[str] = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location" and entry.value:
                external_data_files.add(Path(entry.value).name)
    return external_data_files

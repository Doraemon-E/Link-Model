from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def quantize_exported_model(export_dir: Path, output_dir: Path, weight_type: str = "qint8") -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    resolved_weight_type = _resolve_weight_type(weight_type)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tmp-{output_dir.name}-", dir=output_dir.parent))

    try:
        for source_path in sorted(export_dir.iterdir()):
            if not source_path.is_file():
                continue

            destination_path = temp_dir / source_path.name
            if source_path.suffix == ".onnx":
                quantize_dynamic(
                    model_input=source_path.as_posix(),
                    model_output=destination_path.as_posix(),
                    weight_type=resolved_weight_type,
                )
            else:
                shutil.copy2(source_path, destination_path)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(temp_dir.as_posix(), output_dir.as_posix())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _resolve_weight_type(weight_type: str):
    from onnxruntime.quantization import QuantType

    normalized = weight_type.strip().lower()
    if normalized == "qint8":
        return QuantType.QInt8
    if normalized == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"Unsupported quantization weight type: {weight_type}")


from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _load_convert_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "convert_hy_mt_to_coreml.py"
    spec = importlib.util.spec_from_file_location("convert_hy_mt_to_coreml_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConvertActivationRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.convert_module = _load_convert_module()

    def test_w8a8_uses_fork_stateful_backend(self) -> None:
        convert = self.convert_module
        sample_data = [{"input_ids": np.asarray([[1, 2]], dtype=np.int32)}]
        sample_meta = {"sample_count": 1}
        config = object()
        fp16_model = object()

        with (
            patch.object(convert, "_build_activation_calibration_samples", return_value=(sample_data, sample_meta)) as build_samples,
            patch.object(convert, "_build_activation_quantization_config", return_value=config) as build_config,
            patch.object(convert, "linear_quantize_activations_stateful", return_value="A8_MODEL") as a8_quantize,
        ):
            quantized_model, calibration_meta, a8_backend = convert._apply_activation_quantization_for_w8a8(
                quantized_model=fp16_model,
                model_dir=Path("fake-model"),
                activation_calibration_jsonl=Path("fake-calib.jsonl"),
                context_length=128,
                quantization_dtype="int8",
                calibration_op_group_size=32,
                calibration_compute_units=convert.ct.ComputeUnit.ALL,
            )

        self.assertEqual(quantized_model, "A8_MODEL")
        self.assertEqual(calibration_meta, sample_meta)
        self.assertEqual(a8_backend, "fork_stateful")
        build_samples.assert_called_once_with(
            model_dir=Path("fake-model"),
            calibration_jsonl=Path("fake-calib.jsonl"),
            context_length=128,
        )
        build_config.assert_called_once_with(dtype="int8")
        a8_quantize.assert_called_once_with(
            fp16_model,
            config=config,
            sample_data=sample_data,
            calibration_op_group_size=32,
            calibration_compute_units=convert.ct.ComputeUnit.ALL,
        )

    def test_w8a8_error_is_raised_without_fallback(self) -> None:
        convert = self.convert_module
        sample_data = [{"input_ids": np.asarray([[9]], dtype=np.int32)}]
        sample_meta = {"sample_count": 1}

        with (
            patch.object(convert, "_build_activation_calibration_samples", return_value=(sample_data, sample_meta)),
            patch.object(convert, "_build_activation_quantization_config", return_value=object()),
            patch.object(
                convert,
                "linear_quantize_activations_stateful",
                side_effect=RuntimeError("A8 calibration failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "A8 calibration failed"):
                convert._apply_activation_quantization_for_w8a8(
                    quantized_model=object(),
                    model_dir=Path("fake-model"),
                    activation_calibration_jsonl=Path("fake-calib.jsonl"),
                    context_length=64,
                    quantization_dtype="int8",
                    calibration_op_group_size=16,
                    calibration_compute_units=convert.ct.ComputeUnit.ALL,
                )


if __name__ == "__main__":
    unittest.main()

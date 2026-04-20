#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto
from torch.export import Dim

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from covert_to_coreml import _build_linear_ptq_config
from helper.coreml_quantization_helpers import (
    assert_coreml_weight_quantization_ops,
    assert_torch_quantization_metadata,
)


class _TinyLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Keep input channel > group_size to make per-block(64) valid.
        self.proj_in = torch.nn.Linear(128, 128, bias=False)
        self.proj_out = torch.nn.Linear(128, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = torch.relu(x)
        x = self.proj_out(x)
        return x.to(torch.float16)


def _run() -> dict[str, object]:
    if ct.utils._macos_version() < (15, 0):
        return {
            "status": "skipped",
            "reason": "iOS18 CoreML conversion is only supported on macOS 15+",
        }

    model = _TinyLinearModel().eval()
    ptq_config = _build_linear_ptq_config(
        q_bits=8,
        q_group_size=64,
        q_mode="affine",
    )
    quantizer = cto.torch.quantization.PostTrainingQuantizer(model, ptq_config)
    quantized_model = quantizer.compress().eval()
    assert_torch_quantization_metadata(quantized_model)

    sample_input = torch.ones((1, 4, 128), dtype=torch.float32)
    dynamic_shapes = {"x": {1: Dim("query_length", min=1, max=8)}}
    exported_program = torch.export.export(
        quantized_model,
        args=(sample_input,),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    exported_program = exported_program.run_decompositions({})

    coreml_model = ct.convert(
        exported_program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(
                name="x",
                shape=(1, ct.RangeDim(lower_bound=1, upper_bound=8, default=1), 128),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(dtype=np.float16)],
    )
    assert_coreml_weight_quantization_ops(coreml_model)

    quant_ops = (
        coreml_model._mil_program.functions["main"]
        .find_ops(op_type="constexpr_blockwise_shift_scale")
    )
    quant_scale_keys = [
        key
        for key in quantized_model.state_dict().keys()
        if "_COREML_/" in key and "/quantization_scale" in key
    ]
    return {
        "status": "passed",
        "torch_quant_scale_key_count": len(quant_scale_keys),
        "coreml_quant_op_count": len(quant_ops),
    }


def main() -> int:
    result = _run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"passed", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main())

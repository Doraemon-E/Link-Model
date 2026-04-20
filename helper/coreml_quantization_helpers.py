from __future__ import annotations

import coremltools as ct
import torch


def assert_torch_quantization_metadata(quantized_model: torch.nn.Module) -> None:
    state_dict_keys = list(quantized_model.state_dict().keys())
    quant_scale_keys = [
        key
        for key in state_dict_keys
        if "_COREML_/" in key and "/quantization_scale" in key
    ]
    if quant_scale_keys:
        return

    quant_metadata_keys = [key for key in state_dict_keys if "_COREML_/" in key]
    sample_keys = quant_metadata_keys[:10]
    raise RuntimeError(
        "Torch PTQ metadata is missing. "
        f"Found {len(quant_metadata_keys)} _COREML_ keys and 0 quantization_scale keys. "
        f"Sample keys: {sample_keys}"
    )


def assert_coreml_weight_quantization_ops(coreml_model: ct.models.MLModel) -> None:
    mil_program = getattr(coreml_model, "_mil_program", None)
    if mil_program is None:
        raise RuntimeError("CoreML MIL program is missing; unable to validate quant ops")

    main_function = mil_program.functions.get("main")
    if main_function is None:
        raise RuntimeError("CoreML MIL main function is missing; unable to validate quant ops")

    quant_ops = main_function.find_ops(op_type="constexpr_blockwise_shift_scale")
    if quant_ops:
        return

    raise RuntimeError(
        "CoreML model does not contain constexpr_blockwise_shift_scale ops. "
        "Quantized weights were not materialized into the exported model."
    )

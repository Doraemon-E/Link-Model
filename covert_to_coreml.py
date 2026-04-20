from transformers import AutoModelForCausalLM
import coremltools as ct
import coremltools.optimize as cto
import torch
from pathlib import Path
from datetime import datetime
from helper.stateful_hunyuan_for_coreml import StatefulHunYuanForCoreML
from helper.coreml_quantization_helpers import (
    assert_coreml_weight_quantization_ops,
    assert_torch_quantization_metadata,
)
from torch.export import Dim
import numpy as np

from helper.coreml_bundle_helpers import (
    build_timestamped_coreml_paths,
    copy_runtime_files,
    make_zip_with_parent,
    package_coreml_bundle,
    write_translation_manifest,
)


DEFAULT_MODEL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")

DEFAULT_COREML_OUTPUT_ROOT = Path("models/translation/converted/coreml-int8")
DEFAULT_COREML_ARTIFACT_STEM = "hy-mt1.5-1.8b-coreml-int8"
DEFAULT_COREML_PACKAGED_ROOT = Path("models/translation/packaged")

DEFAULT_MLX_OUTPUT_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
DEFAULT_MLX_PACKAGED_ZIP = Path(
    "models/translation/packaged/hy-mt1.5-1.8b-mlx-int8.zip"
)

DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_COREML_Q_BITS = 8
DEFAULT_COREML_Q_GROUP_SIZE = 64
DEFAULT_COREML_Q_MODE = "affine"

DEFAULT_MLX_Q_BITS = 8
DEFAULT_MLX_Q_GROUP_SIZE = 64
DEFAULT_MLX_Q_MODE = "affine"


def _load_base_model(model_dir: Path) -> torch.nn.Module:
    # 加载并固定PyTorch模型状态
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # 推理模式, 关闭dropout等训练行为
    model.eval()

    # 更稳的 attention 路径
    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"

    # 更保守的 rope 设置
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.rope_type = "default"

    return model


def _resolve_coreml_quantization_options(
    q_bits: int,
    q_mode: str,
) -> tuple[str, str]:
    if q_bits == 8:
        weight_dtype = "int8"
    elif q_bits == 4:
        weight_dtype = "int4"
    else:
        raise ValueError(f"unsupported q_bits={q_bits}, only 4 or 8 are supported")

    normalized_q_mode = q_mode.lower()
    if normalized_q_mode not in {"affine", "symmetric"}:
        raise ValueError(
            f"unsupported q_mode={q_mode}, expected one of: affine, symmetric"
        )
    return weight_dtype, normalized_q_mode


def _build_linear_ptq_config(
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> cto.torch.quantization.PostTrainingQuantizerConfig:
    if q_group_size <= 0:
        raise ValueError(f"q_group_size must be > 0, got {q_group_size}")

    weight_dtype, quantization_scheme = _resolve_coreml_quantization_options(
        q_bits=q_bits,
        q_mode=q_mode,
    )
    return cto.torch.quantization.PostTrainingQuantizerConfig.from_dict(
        {
            "module_type_configs": {
                torch.nn.Linear: {
                    "weight_dtype": weight_dtype,
                    "granularity": "per_block",
                    "block_size": q_group_size,
                    "quantization_scheme": quantization_scheme,
                },
            },
        }
    )


def _load_quantized_torch_model(
    model_dir: Path,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> torch.nn.Module:
    """
    先在 Torch 侧做 weight-only PTQ，然后返回“带压缩信息”的 torch model。
    这样 ct.convert() 时会自动尝试生成压缩后的 Core ML 表达。
    """
    model = _load_base_model(model_dir)

    # 只量化 Linear，采用 blockwise 量化对齐 MLX 的 W8/G64/affine 策略。
    config = _build_linear_ptq_config(
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )

    quantizer = cto.torch.quantization.PostTrainingQuantizer(model, config)
    quantized_model = quantizer.compress()
    quantized_model.eval()
    assert_torch_quantization_metadata(quantized_model)
    return quantized_model


def _convert_coreml(
    model_dir: Path,
    output_dir: Path,
    context_length: int,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> Path:
    # 先在 Torch 侧量化/压缩
    quantized_torch_model = _load_quantized_torch_model(
        model_dir=model_dir,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )

    wrapper = StatefulHunYuanForCoreML(
        model=quantized_torch_model, max_cache_len=context_length
    )
    wrapper.eval()
    wrapper.reset_cache()

    sample_input_ids = torch.ones((1, 8), dtype=torch.int32)
    dynamic_shapes = {"input_ids": {1: Dim("query_length", min=1, max=context_length)}}

    exported_program = torch.export.export(
        wrapper, args=(sample_input_ids,), dynamic_shapes=dynamic_shapes, strict=False
    )
    # 算子分解，转成CoreML兼容的算子
    exported_program = exported_program.run_decompositions({})

    # 构建输出路径
    output_dir.mkdir(parents=True, exist_ok=True)
    coreml_path = output_dir / "hy_mt_w8_from_torch.mlpackage"

    # 自定义CoreML 转换的 流水线
    default_passes = list(ct.PassPipeline.DEFAULT.passes)
    custom_pass_pipeline = ct.PassPipeline(
        pass_names=default_passes, pipeline_name="hy_mt_coreml_export"
    )
    # 防止 kv cache 被破坏
    custom_pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

    # 转换成CoreML
    coreml_model = ct.convert(
        exported_program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=custom_pass_pipeline,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(
                    1,
                    ct.RangeDim(lower_bound=1, upper_bound=context_length, default=1),
                ),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=_build_coreml_states(wrapper),
    )
    assert_coreml_weight_quantization_ops(coreml_model)
    coreml_model.save(str(coreml_path))
    return coreml_path


def _convert_mlx(
    model_dir: Path,
    output_dir: Path,
    q_bits: int = DEFAULT_MLX_Q_BITS,
    q_group_size: int = DEFAULT_MLX_Q_GROUP_SIZE,
    q_mode: str = DEFAULT_MLX_Q_MODE,
) -> Path:
    """
    把 Hugging Face / 本地模型目录转成 MLX 量化版本。
    这里走 mlx-lm 的 Python API，不再用 subprocess。
    """
    from mlx_lm import convert

    # 官方公开示例是 convert(repo, quantize=True, ...)
    # 这里再补上本地输出目录和 q_bits。
    convert(
        str(model_dir),
        mlx_path=str(output_dir),
        quantize=True,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )

    return output_dir


# helper
def _make_state(name: str, shape: tuple[int, ...], dtype) -> ct.StateType:
    return ct.StateType(
        wrapped_type=ct.TensorType(
            shape=shape,
            dtype=dtype,
        ),
        name=name,
    )


def _build_coreml_states(
    wrapper,
    cache_dtype=np.float16,
    position_dtype=np.float16,
) -> list[ct.StateType]:
    states: list[ct.StateType] = []

    for layer_idx in range(wrapper.num_layers):
        key_cache = getattr(wrapper, f"key_cache_{layer_idx}")
        states.append(
            _make_state(
                name=f"key_cache_{layer_idx}",
                shape=tuple(key_cache.shape),
                dtype=cache_dtype,
            )
        )

    for layer_idx in range(wrapper.num_layers):
        value_cache = getattr(wrapper, f"value_cache_{layer_idx}")
        states.append(
            _make_state(
                name=f"value_cache_{layer_idx}",
                shape=tuple(value_cache.shape),
                dtype=cache_dtype,
            )
        )

    states.append(
        _make_state(
            name="cache_position",
            shape=tuple(wrapper.cache_position.shape),
            dtype=position_dtype,
        )
    )

    return states


def run():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    coreml_output_dir, coreml_packaged_zip = build_timestamped_coreml_paths(
        timestamp,
        output_root=DEFAULT_COREML_OUTPUT_ROOT,
        packaged_root=DEFAULT_COREML_PACKAGED_ROOT,
        artifact_stem=DEFAULT_COREML_ARTIFACT_STEM,
    )

    # 1. Core ML W8
    coreml_path = _convert_coreml(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=coreml_output_dir,
        context_length=DEFAULT_CONTEXT_LENGTH,
        q_bits=DEFAULT_COREML_Q_BITS,
        q_group_size=DEFAULT_COREML_Q_GROUP_SIZE,
        q_mode=DEFAULT_COREML_Q_MODE,
    )
    copy_runtime_files(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=coreml_output_dir,
    )
    write_translation_manifest(
        output_dir=coreml_output_dir,
        model_file_name=coreml_path.name,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )
    package_coreml_bundle(
        source_dir=coreml_output_dir,
        model_file_name=coreml_path.name,
        zip_path=coreml_packaged_zip,
    )

    # 2. MLX W8
    # mlx_path = _convert_mlx(
    #     model_dir=DEFAULT_MODEL_DIR,
    #     output_dir=DEFAULT_MLX_OUTPUT_DIR,
    #     q_bits=DEFAULT_MLX_Q_BITS,
    #     q_group_size=DEFAULT_MLX_Q_GROUP_SIZE,
    #     q_mode=DEFAULT_MLX_Q_MODE,
    # )
    # make_zip_with_parent(
    #     source_dir=mlx_path,
    #     zip_path=DEFAULT_MLX_PACKAGED_ZIP,
    # )


if __name__ == "__main__":
    run()

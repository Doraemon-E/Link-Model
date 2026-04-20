from transformers import AutoModelForCausalLM
import coremltools as ct
import coremltools.optimize as cto
import torch
from pathlib import Path
import subprocess
import shutil
import os
import tempfile
from stateful_hunyuan_for_coreml import StatefulHunYuanForCoreML
from torch.export import Dim
import numpy as np


DEFAULT_MODEL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")

DEFAULT_COREML_OUTPUT_DIR = Path(
    "models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml"
)
DEFAULT_COREML_PACKAGED_ZIP = Path(
    "models/translation/packaged/hy-mt1.5-1.8b-coreml-int8.zip"
)

DEFAULT_MLX_OUTPUT_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
DEFAULT_MLX_PACKAGED_ZIP = Path(
    "models/translation/packaged/hy-mt1.5-1.8b-mlx-int8.zip"
)

DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_MLX_Q_BITS = 8

DEFAULT_FP16_COREML_NAME = "hy_mt_fp16.mlpackage"
DEFAULT_W8_COREML_NAME = "hy_mt_w8.mlpackage"
COMPRESSED_WEIGHT_OPS = (
    "constexpr_affine_dequantize",
    "constexpr_blockwise_shift_scale",
    "constexpr_lut_to_dense",
)


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


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _configure_coreml_tmpdir(base_dir: Path) -> Path:
    coreml_tmpdir = (base_dir / "CoreMLTemp").resolve()
    coreml_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(coreml_tmpdir)
    tempfile.tempdir = str(coreml_tmpdir)
    return coreml_tmpdir


def _compile_mlpackage_to_output(mlpackage_path: Path, output_mlmodelc_path: Path) -> Path:
    _configure_coreml_tmpdir(output_mlmodelc_path.parent)
    compiled_tmp_path = Path(ct.models.utils.compile_model(str(mlpackage_path)))
    output_mlmodelc_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mlmodelc_path.exists():
        shutil.rmtree(output_mlmodelc_path)
    shutil.copytree(compiled_tmp_path, output_mlmodelc_path)
    return output_mlmodelc_path


def _read_weight_bin_size(model_path: Path) -> int | None:
    weight_bin = model_path / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    if weight_bin.is_file():
        return weight_bin.stat().st_size
    return None


def _validate_w8_coreml_artifact(
    *,
    fp16_mlpackage_path: Path,
    w8_mlpackage_path: Path,
    output_dir: Path,
) -> None:
    compiled_dir = output_dir / "Compiled"
    fp16_mlmodelc = _compile_mlpackage_to_output(
        fp16_mlpackage_path, compiled_dir / "hy_mt_fp16.mlmodelc"
    )
    w8_mlmodelc = _compile_mlpackage_to_output(
        w8_mlpackage_path, compiled_dir / "hy_mt_w8.mlmodelc"
    )

    w8_mil_path = w8_mlmodelc / "model.mil"
    if not w8_mil_path.is_file():
        raise RuntimeError(f"W8 model.mil not found: {w8_mil_path}")
    w8_mil_text = w8_mil_path.read_text(encoding="utf-8", errors="ignore")
    matched_compressed_ops = [name for name in COMPRESSED_WEIGHT_OPS if name in w8_mil_text]
    if not matched_compressed_ops:
        raise RuntimeError(
            "W8 quantization validation failed: no compressed weight constexpr op found in model.mil. "
            f"expected one of {COMPRESSED_WEIGHT_OPS}, mil_path={w8_mil_path}"
        )

    fp16_pkg_size = _path_size_bytes(fp16_mlpackage_path)
    w8_pkg_size = _path_size_bytes(w8_mlpackage_path)
    pkg_reduced = w8_pkg_size < fp16_pkg_size

    fp16_weight_size = _read_weight_bin_size(fp16_mlpackage_path)
    w8_weight_size = _read_weight_bin_size(w8_mlpackage_path)
    weight_reduced = (
        fp16_weight_size is not None
        and w8_weight_size is not None
        and w8_weight_size < fp16_weight_size
    )

    if not (pkg_reduced or weight_reduced):
        raise RuntimeError(
            "W8 quantization validation failed: neither package size nor weight.bin size reduced. "
            f"fp16_pkg_bytes={fp16_pkg_size}, w8_pkg_bytes={w8_pkg_size}, "
            f"fp16_weight_bytes={fp16_weight_size}, w8_weight_bytes={w8_weight_size}, "
            f"fp16_path={fp16_mlpackage_path}, w8_path={w8_mlpackage_path}"
        )


def _export_fp16_coreml(model_dir: Path, output_dir: Path, context_length: int) -> Path:
    model = _load_base_model(model_dir)
    wrapper = StatefulHunYuanForCoreML(
        model=model, max_cache_len=context_length
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
    fp16_path = output_dir / DEFAULT_FP16_COREML_NAME
    if fp16_path.exists():
        shutil.rmtree(fp16_path)

    # 自定义CoreML 转换的 流水线
    default_passes = list(ct.PassPipeline.DEFAULT.passes)
    custom_pass_pipeline = ct.PassPipeline(
        pass_names=default_passes, pipeline_name="hy_mt_coreml_export"
    )
    # 防止 kv cache 被破坏
    custom_pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

    # 转换成CoreML
    fp16_model = ct.convert(
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
    fp16_model.save(str(fp16_path))
    return fp16_path


def _quantize_coreml_w8(fp16_mlpackage_path: Path, output_dir: Path) -> Path:
    fp16_model = ct.models.MLModel(str(fp16_mlpackage_path))
    quant_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
        )
    )
    try:
        w8_model = cto.coreml.linear_quantize_weights(fp16_model, config=quant_config)
    except Exception as exc:
        # State-heavy models can fail in canonicalize_inplace_pattern during re-conversion.
        # In that case, run the same quantization graph pass but skip this pass explicitly.
        message = str(exc)
        if (
            "canonicalize_inplace_pattern" not in message
            and "coreml_update_state" not in message
        ):
            raise
        w8_model = _linear_quantize_weights_skip_inplace_pattern(fp16_model, quant_config)
    w8_mlpackage_path = output_dir / DEFAULT_W8_COREML_NAME
    if w8_mlpackage_path.exists():
        shutil.rmtree(w8_mlpackage_path)
    w8_model.save(str(w8_mlpackage_path))
    return w8_mlpackage_path


def _linear_quantize_weights_skip_inplace_pattern(
    mlmodel: ct.models.MLModel,
    config: cto.coreml.OptimizationConfig,
) -> ct.models.MLModel:
    from coremltools.converters.mil.converter import mil_convert
    from coremltools.converters.mil.mil.passes.graph_pass import PassOption
    from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
    from coremltools.models import utils as model_utils

    quantizer_pass = PASS_REGISTRY["compression::linear_quantize_weights"]
    quantizer_pass.set_options(
        [PassOption("config", config), PassOption("joint_compression", False)]
    )
    prog = model_utils._apply_graph_pass(
        mlmodel,
        quantizer_pass,
        return_pymil_prog=True,
    )
    model_spec = mlmodel.get_spec()
    pass_pipeline = ct.PassPipeline(
        pass_names=list(ct.PassPipeline.DEFAULT.passes),
        pipeline_name="stateful_quantization_safe",
    )
    pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

    return mil_convert(
        prog,
        convert_from="milinternal",
        convert_to="mlprogram",
        compute_units=mlmodel.compute_unit,
        specification_version=model_spec.specificationVersion,
        model_description=model_spec.description,
        pass_pipeline=pass_pipeline,
        skip_model_load=mlmodel.__proxy__ is None,
    )


def _convert_mlx(
    model_dir: Path,
    output_dir: Path,
    q_bits: int = 8,
) -> Path:
    """
    把 Hugging Face / 本地模型目录转成 MLX 量化版本。
    这里走 mlx-lm 的 Python API，不再用 subprocess。
    """
    # 官方公开示例是 convert(repo, quantize=True, ...)
    # 这里再补上本地输出目录和 q_bits。
    from mlx_lm import convert

    convert(
        str(model_dir),
        mlx_path=str(output_dir),
        quantize=True,
        q_bits=q_bits,
    )

    return output_dir


def _make_zip_with_parent(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(
        ["/usr/bin/ditto", "-c", "-k", "--keepParent", str(source_dir), str(zip_path)],
        check=True,
    )


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


def run(run_mlx: bool = False):
    # 1. Core ML W8 (default path)
    output_dir = DEFAULT_COREML_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fp16_mlpackage_path = _export_fp16_coreml(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=output_dir,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )
    w8_mlpackage_path = _quantize_coreml_w8(
        fp16_mlpackage_path=fp16_mlpackage_path,
        output_dir=output_dir,
    )
    _validate_w8_coreml_artifact(
        fp16_mlpackage_path=fp16_mlpackage_path,
        w8_mlpackage_path=w8_mlpackage_path,
        output_dir=output_dir,
    )
    _make_zip_with_parent(
        source_dir=w8_mlpackage_path,
        zip_path=DEFAULT_COREML_PACKAGED_ZIP,
    )

    # 2. MLX W8 (optional)
    if not run_mlx:
        return

    mlx_path = _convert_mlx(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_MLX_OUTPUT_DIR,
        q_bits=DEFAULT_MLX_Q_BITS,
    )
    _make_zip_with_parent(
        source_dir=mlx_path,
        zip_path=DEFAULT_MLX_PACKAGED_ZIP,
    )


if __name__ == "__main__":
    run()

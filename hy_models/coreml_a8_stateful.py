from __future__ import annotations

from collections import defaultdict
import gc
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_16
from coremltools import _SPECIFICATION_VERSION_IOS_17
from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
from coremltools.converters.mil.mil.passes.graph_pass import PassOption
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.models import model as _model
from coremltools.models import utils as _model_utils
from coremltools.optimize.coreml import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.coreml.experimental._model_debugger import ModelDebugger, ModelInfo


def _safe_rmtree(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        return


def _predict_with_optional_state(model: ct.models.MLModel, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_descriptions = model.get_spec().description.state
    if len(state_descriptions) == 0:
        return model.predict(inputs)
    try:
        state = model.make_state()
    except Exception as exc:
        framework_error = getattr(model, "_framework_error", None)
        if framework_error is not None:
            raise RuntimeError(
                "stateful calibration failed to create MLState via make_state(); "
                f"framework_error={framework_error}"
            ) from exc
        raise RuntimeError("stateful calibration failed to create MLState via make_state()") from exc
    if state is None:
        raise RuntimeError("stateful calibration expected MLState from make_state(), got None")
    return model.predict(inputs, state=state)


class _StateAwareModelDebugger(ModelDebugger):
    def predict_intermediate_outputs(
        self,
        inputs: dict[str, np.ndarray],
        intermediate_output_names: list[str],
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_ONLY,
    ) -> dict[str, np.ndarray]:
        cloned_spec = self.__class__.clone_spec(self.model_info.spec)
        if self.model_info.spec.specificationVersion < _SPECIFICATION_VERSION_IOS_16:
            cloned_spec.specificationVersion = max(
                self.model_info.spec.specificationVersion,
                _SPECIFICATION_VERSION_IOS_16,
            )
        cloned_model_info = ModelInfo(self.__class__.get_program_info(cloned_spec.mlProgram), cloned_spec)
        cloned_block_info = self.__class__.get_any_block(cloned_model_info)

        for output_name in intermediate_output_names:
            cloned_output_type = self.__class__.get_output_feature_type(output_name, self.block_info.operations)
            if cloned_output_type is None:
                continue
            cloned_block_info.spec.outputs.append(output_name)
            cloned_output = ct.proto.Model_pb2.FeatureDescription()
            cloned_output.name = output_name
            cloned_output.type.multiArrayType.dataType = cloned_output_type
            cloned_model_info.spec.description.output.append(cloned_output)

        model = ct.models.MLModel(
            cloned_spec,
            is_temp_package=True,
            weights_dir=self.weights_dir,
            compute_units=compute_units,
            skip_model_load=False,
        )
        compiled_model_path: Path | None = None
        package_path = Path(model.package_path) if getattr(model, "package_path", None) else None
        try:
            compiled_model_path = Path(model.get_compiled_model_path())
        except Exception:
            compiled_model_path = None
        try:
            return _predict_with_optional_state(model, inputs)
        finally:
            del model
            gc.collect()
            _safe_rmtree(compiled_model_path)
            _safe_rmtree(package_path)


def _update_tensor_range(
    tensor_name: str,
    tensor_value: Union[int, float, np.ndarray],
    activation_stats_dict: Dict[str, Dict[str, float]],
) -> None:
    tensor_min = np.min(np.array(tensor_value).flatten())
    tensor_max = np.max(np.array(tensor_value).flatten())
    activation_stats_dict[tensor_name]["rmin"] = tensor_min
    activation_stats_dict[tensor_name]["rmax"] = tensor_max
    if tensor_name in activation_stats_dict:
        activation_stats_dict[tensor_name]["rmin"] = min(tensor_min, activation_stats_dict[tensor_name]["rmin"])
        activation_stats_dict[tensor_name]["rmax"] = max(tensor_max, activation_stats_dict[tensor_name]["rmax"])
    else:
        activation_stats_dict[tensor_name]["rmin"] = tensor_min
        activation_stats_dict[tensor_name]["rmax"] = tensor_max


def _combine_lists_with_common_elements(data: List[List[str]]) -> List[List[str]]:
    merged = []
    for item in data:
        item_set = set(item)
        not_exist = True
        for result in merged:
            if result & item_set:
                result.update(item_set)
                not_exist = False
                break
        if not_exist:
            merged.append(item_set)
    return [sorted(group) for group in merged]


def _adjust_concat_surrounding_activation_stats(
    concat_op_info_list: List[List[str]],
    activation_stats_dict: Dict[str, Dict[str, float]],
) -> None:
    if concat_op_info_list is None:
        return

    concat_list_adjusted = _combine_lists_with_common_elements(concat_op_info_list)
    for concat_group in concat_list_adjusted:
        group_rmin_list, group_rmax_list = [], []
        for tensor_name in concat_group:
            if tensor_name in activation_stats_dict:
                group_rmin_list.append(activation_stats_dict[tensor_name]["rmin"])
                group_rmax_list.append(activation_stats_dict[tensor_name]["rmax"])
        if len(group_rmin_list) == 0:
            raise ValueError("None of the calibration run succeeded. Please check calibration logs.")
        group_rmin, group_rmax = min(group_rmin_list), max(group_rmax_list)
        for tensor_name in concat_group:
            if tensor_name in activation_stats_dict:
                activation_stats_dict[tensor_name]["rmin"] = group_rmin
                activation_stats_dict[tensor_name]["rmax"] = group_rmax


def _get_activation_calibration_stats_stateful(
    fpmodel: ct.models.MLModel,
    sample_data: list[dict[str, np.ndarray]],
    calibration_op_group_size: int = -1,
    calibration_compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> Dict[str, Dict[str, float]]:
    debugger = _StateAwareModelDebugger(fpmodel)
    activation_stats_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    intermediate_output_names = debugger.get_intermediate_output_names(lambda op: (op.spec.type != "const"))

    for data in sample_data:
        for input_name in data:
            _update_tensor_range(input_name, data[input_name], activation_stats_dict)

    model_spec = fpmodel.get_spec()
    output_names = {desc.name for desc in model_spec.description.output}
    intermediate_output_names = [name for name in intermediate_output_names if name not in output_names]

    for data in tqdm(
        sample_data,
        desc="Running compression pass linear_quantize_activations",
        unit=" calibration samples",
    ):
        debugger.step(
            inputs=data,
            activation_stats_dict=activation_stats_dict,
            intermediate_output_names=intermediate_output_names,
            compute_units=calibration_compute_units,
            op_group_size=calibration_op_group_size,
        )

    _adjust_concat_surrounding_activation_stats(debugger._get_concat_op_info(), activation_stats_dict)
    return activation_stats_dict


def linear_quantize_activations_stateful(
    mlmodel: _model.MLModel,
    config: _OptimizationConfig,
    sample_data: List[Dict[Optional[str], np.ndarray]],
    calibration_op_group_size: int = -1,
    calibration_compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> _model.MLModel:
    if hasattr(mlmodel, "_is_multifunction") and mlmodel._is_multifunction():
        raise ValueError("linear_quantize_activations_stateful is not supported for a multifunction model.")

    for sample in sample_data:
        if None in sample.keys():
            input_spec = mlmodel.get_spec().description.input
            if len(sample.keys()) > 1 or len(input_spec) > 1:
                raise ValueError(
                    "When the model has multiple inputs, please provide the name for each data in `sample_data`"
                )
            inferred_input_name = input_spec[0].name
            sample[inferred_input_name] = sample[None]
            del sample[None]

    insert_prefix_quantize_dequantize_pair = PASS_REGISTRY["compression::insert_prefix_quantize_dequantize_pair"]
    insert_prefix_quantize_dequantize_pair.set_options([PassOption("config", config)])
    activation_stats = _get_activation_calibration_stats_stateful(
        mlmodel,
        sample_data,  # type: ignore[arg-type]
        calibration_op_group_size,
        calibration_compute_units,
    )
    insert_prefix_quantize_dequantize_pair.set_options([PassOption("activation_stats", activation_stats)])

    insert_suffix_quantize_dequantize_pair = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
    insert_suffix_quantize_dequantize_pair.set_options([PassOption("config", config)])
    insert_suffix_quantize_dequantize_pair.set_options([PassOption("activation_stats", activation_stats)])

    graph_passes = [
        insert_prefix_quantize_dequantize_pair,
        insert_suffix_quantize_dequantize_pair,
        PASS_REGISTRY["common::dequantize_quantize_pair_elimination"],
    ]
    return _model_utils._apply_graph_pass(
        mlmodel,
        graph_passes,
        spec_version=_SPECIFICATION_VERSION_IOS_17,
        pymil_load_func=_milproto_to_pymil.load,
        skip_model_load=mlmodel.__proxy__ is None,
    )

#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import inspect
from typing import Callable

import numpy as np
from pyspark import SparkContext, TaskContext
from pyspark.ml.param import Param, Params
from pyspark.sql import SparkSession


def _get_spark_session() -> SparkSession:
    """Get or create spark session.
    Note: This function can only be invoked from driver side."""
    if TaskContext.get() is not None:
        # safety check.
        raise RuntimeError(
            "_get_spark_session should not be invoked from executor side."
        )
    return SparkSession.builder.getOrCreate()


def _is_local(sc: SparkContext) -> bool:
    """Whether it is Spark local mode"""
    return sc._jsc.sc().isLocal()  # type: ignore


def _get_gpu_id(task_context: TaskContext) -> int:
    """Get the gpu id from the task resources"""
    if task_context is None:
        # safety check.
        raise RuntimeError("_get_gpu_id should not be invoked from driver side.")
    resources = task_context.resources()
    if "gpu" not in resources:
        raise RuntimeError(
            "Couldn't get the gpu id, Please check the GPU resource configuration."
        )
    # return the first gpu id.
    return int(resources["gpu"].addresses[0].strip())


def _get_default_params_from_func(func: Callable, unsupported_set: list[str] = []):
    """
    Returns a dictionary of parameters and their default value of function fn.
    Only the parameters with a default value will be included.
    """
    sig = inspect.signature(func)
    filtered_params_dict = {}
    for parameter in sig.parameters.values():
        # Remove parameters without a default value and those in the unsupported_set
        if (
            parameter.default is not parameter.empty
            and parameter.name not in unsupported_set
        ):
            filtered_params_dict[parameter.name] = parameter.default
    return filtered_params_dict


def _get_class_name(cls: type) -> str:
    """
    Return the class name.
    """
    return f"{cls.__module__}.{cls.__name__}"


def _set_pyspark_cuml_cls_param_attrs(pyspark_estimator_class, pyspark_model_class):
    """
    To set pyspark parameter attributes according to cuml parameters.
    This function must be called after you finished the subclass design of _CumlEstimator_CumlModel

    Eg,

    class SparkDummy(_CumlEstimator):
        pass
    class SparkDummyModel(_CumlModel):
        pass
    _set_pyspark_cuml_cls_param_attrs(SparkDummy, SparkDummyModel)
    """
    cuml_estimator_class_name = _get_class_name(pyspark_estimator_class._cuml_cls())
    params_dict = pyspark_estimator_class._get_cuml_params_default()

    def param_value_converter(v):
        if isinstance(v, np.generic):
            # convert numpy scalar values to corresponding python scalar values
            return np.array(v).item()
        if isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        if isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        return v

    def set_param_attrs(attr_name, param_obj_):
        param_obj_.typeConverter = param_value_converter
        setattr(pyspark_estimator_class, attr_name, param_obj_)
        setattr(pyspark_model_class, attr_name, param_obj_)

    for name in params_dict.keys():
        doc = f"Refer to CUML doc of {cuml_estimator_class_name} for this param {name}"

        param_obj = Param(Params._dummy(), name=name, doc=doc)
        set_param_attrs(name, param_obj)

#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple, Union

if TYPE_CHECKING:
    import cudf

import numpy as np
from pyspark import BarrierTaskContext, SparkContext, TaskContext
from pyspark.sql import SparkSession

_ArrayOrder = Literal["C", "F"]


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


def _str_or_numerical(x: str) -> Union[str, float, int]:
    """
    Convert to int if x is str representation of integer,
    otherwise float if x is representation of float, otherwise return input string.
    """
    try:
        _x: Union[str, int, float] = int(x)
    except:
        try:
            _x = float(x)
        except:
            _x = x
    return _x


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


def _get_default_params_from_func(
    func: Callable, unsupported_set: List[str] = []
) -> Dict[str, Any]:
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


class PartitionDescriptor:
    """
    Partition descriptor

    m: total number of rows across all workers
    n: total number of cols
    parts_rank_size: a sequence of (rank, rows per partitions)
    rank: rank to be mapped
    """

    def __init__(
        self, m: int, n: int, rank: int, parts_rank_size: List[Tuple[int, int]]
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.rank = rank
        self.parts_rank_size = parts_rank_size

    @classmethod
    def build(cls, partition_rows: List[int], total_cols: int) -> "PartitionDescriptor":
        context = BarrierTaskContext.get()
        if context is None:
            # safety check.
            raise RuntimeError("build should not be invoked from driver side.")

        rank = context.partitionId()

        # prepare (parts, rank)

        import json

        rank_size = [(rank, size) for size in partition_rows]
        messages = context.allGather(message=json.dumps(rank_size))
        parts_rank_size = [item for pair in messages for item in json.loads(pair)]
        total_rows = sum(pair[1] for pair in parts_rank_size)

        return cls(total_rows, total_cols, rank, parts_rank_size)


def _concat_and_free(
    np_array_list: List[np.ndarray], order: _ArrayOrder = "F"
) -> np.ndarray:
    """
    concatenates a list of compatible numpy arrays into a 'order' ordered output array,
    in a memory efficient way.
    Note: frees list elements so do not reuse after calling.
    """
    rows = sum(arr.shape[0] for arr in np_array_list)
    if len(np_array_list[0].shape) > 1:
        cols = np_array_list[0].shape[1]
        concat_shape: Tuple[int, ...] = (rows, cols)
    else:
        concat_shape = (rows,)
    d_type = np_array_list[0].dtype
    concated = np.empty(shape=concat_shape, order=order, dtype=d_type)
    np.concatenate(np_array_list, out=concated)
    del np_array_list[:]
    return concated


def cudf_to_cuml_array(gdf: Union["cudf.DataFrame", "cudf.Series"], order: str = "F"):  # type: ignore
    try:
        # Compatible with older cuml version (before 23.02)
        from cuml.common.input_utils import input_to_cuml_array
    except ImportError:
        from cuml.common import input_to_cuml_array
    cumlarray, _, _, _ = input_to_cuml_array(gdf, order=order)
    return cumlarray


def dtype_to_pyspark_type(dtype: Union[np.dtype, str]) -> str:
    """Convert np.dtype to the corresponding pyspark type"""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    elif dtype == np.int32:
        return "int"
    elif dtype == np.int16:
        return "short"
    else:
        raise RuntimeError("Unsupported dtype, found ", dtype)


# similar to https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/utils.py
def get_logger(cls: type, level: str = "INFO") -> logging.Logger:
    """Gets a logger by name, or creates and configures it for the first time."""
    name = _get_class_name(cls)
    logger = logging.getLogger(name)

    logger.setLevel(level)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def java_uid(sc: SparkContext, prefix: str) -> str:
    """Returns a random UID that concatenates the given prefix, "_", and 12 random hex chars."""
    assert sc._jvm is not None
    return sc._jvm.org.apache.spark.ml.util.Identifiable.randomUID(prefix)


def _fake_java_impurity_calc(sc: SparkContext, count: int = 3):  # type: ignore
    """Fake a java ImpurityCalculator"""
    assert sc._jvm is not None
    assert sc._gateway is not None

    object_class = sc._jvm.double
    fake_python_impurity_calc = [0 for _ in range(count)]
    fake_java_impurity_calc = sc._gateway.new_array(
        object_class, len(fake_python_impurity_calc)
    )
    for i in range(len(fake_python_impurity_calc)):
        fake_java_impurity_calc[i] = float(fake_java_impurity_calc[i])

    return fake_java_impurity_calc


def _create_internal_node(sc: SparkContext, impurity: str, model: Dict[str, Any], left, right):  # type: ignore
    """Return a Java InternalNode"""

    assert sc._jvm is not None
    assert sc._gateway is not None

    java_split = sc._jvm.org.apache.spark.ml.tree.ContinuousSplit(
        int(model["split_feature"]), float(model["split_threshold"])
    )

    fake_java_impurity_calc = _fake_java_impurity_calc(sc)

    if impurity == "gini":
        java_impurity_cal = sc._jvm.org.apache.spark.mllib.tree.impurity.GiniCalculator(
            fake_java_impurity_calc, int(model["instance_count"])
        )
    elif impurity == "entropy":
        java_impurity_cal = (
            sc._jvm.org.apache.spark.mllib.tree.impurity.EntropyCalculator(
                fake_java_impurity_calc, int(model["instance_count"])
            )
        )
    elif impurity == "variance":
        java_impurity_cal = (
            sc._jvm.org.apache.spark.mllib.tree.impurity.VarianceCalculator(
                fake_java_impurity_calc, int(model["instance_count"])
            )
        )
    else:
        # never reach here
        raise ValueError("Unsupported impurity! ", impurity)

    java_internal_node = sc._jvm.org.apache.spark.ml.tree.InternalNode(
        0.0,  # prediction value is nonsense for internal node, just fake it
        0.0,  # impurity value is nonsense for internal node. just fake it
        float(model["gain"]),
        left,
        right,
        java_split,
        java_impurity_cal,
    )

    return java_internal_node


def _create_leaf_node(sc: SparkContext, impurity: str, model: Dict[str, Any]):  # type: ignore
    """Return a Java LeaftNode
    Please note that, cuml trees uses probs as the leaf values while spark uses
    the stats (how many counts this node has for each label), but they are behave
    the same purpose when doing prediction
    """
    assert sc._jvm is not None
    assert sc._gateway is not None

    leaf_values = model["leaf_value"]

    if impurity == "gini" or impurity == "entropy":
        object_class = sc._jvm.double
        java_probs = sc._gateway.new_array(object_class, len(leaf_values))
        for i in range(len(leaf_values)):
            java_probs[i] = float(leaf_values[i])

        java_impurity_cal = (
            sc._jvm.org.apache.spark.mllib.tree.impurity.GiniCalculator(
                java_probs, int(model["instance_count"])
            )
            if impurity == "gini"
            else sc._jvm.org.apache.spark.mllib.tree.impurity.EntropyCalculator(
                java_probs, int(model["instance_count"])
            )
        )
        prediction = np.argmax(np.asarray(leaf_values))

    elif impurity == "variance":
        fake_java_impurity_calc = _fake_java_impurity_calc(sc, 3)
        java_impurity_cal = (
            sc._jvm.org.apache.spark.mllib.tree.impurity.VarianceCalculator(
                fake_java_impurity_calc, int(model["instance_count"])
            )
        )
        prediction = leaf_values[0]
    else:
        # never reach here
        raise ValueError("Unsupported impurity! ", impurity)

    java_leaf_node = sc._jvm.org.apache.spark.ml.tree.LeafNode(
        float(prediction),
        0.0,  # TODO calculate the impurity according to leaf value, prediction doesn't require it.
        java_impurity_cal,
    )
    return java_leaf_node


def translate_trees(sc: SparkContext, impurity: str, model: Dict[str, Any]):  # type: ignore
    """Translate Cuml RandomForest trees to PySpark trees

    Cuml trees
    [
      {
        "nodeid": 0,
        "split_feature": 3,
        "split_threshold": 0.827687974732221,
        "gain": 0.41999999999999998,
        "instance_count": 10,
        "yes": 1,
        "no": 2,
        "children": [
          {
            "nodeid": 1,
            "leaf_value": [
              1,
              0
            ],
            "instance_count": 7
          },
          {
            "nodeid": 2,
            "leaf_value": [
              0,
              1
            ],
            "instance_count": 3
          }
        ]
      }
    ]

    Spark trees,
             InternalNode {split{featureIndex=3, threshold=0.827687974732221}, gain = 0.41999999999999998}
             /         \
           left        right
           /             \
    LeafNode           LeafNode
    """
    if "split_feature" in model:
        left_child_id = model["yes"]
        right_child_id = model["no"]

        for child in model["children"]:
            if child["nodeid"] == left_child_id:
                left_child = child
            elif child["nodeid"] == right_child_id:
                right_child = child
            else:
                raise ValueError("Unexpected node id")

        return _create_internal_node(
            sc,
            impurity,
            model,
            translate_trees(sc, impurity, left_child),
            translate_trees(sc, impurity, right_child),
        )
    elif "leaf_value" in model:
        return _create_leaf_node(sc, impurity, model)

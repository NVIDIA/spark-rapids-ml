#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import cudf
    import cupy as cp
    import cupyx

import numpy as np
import pandas as pd
import scipy
from pyspark import BarrierTaskContext, SparkConf, SparkContext, TaskContext
from pyspark.sql import Column, SparkSession
from pyspark.sql.types import ArrayType, FloatType

_ArrayOrder = Literal["C", "F"]
_SinglePdDataFrameBatchType = Tuple[
    pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]
]
_SingleNpArrayBatchType = Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]

# FitInputType is type of [(feature, label), ...]
FitInputType = Union[List[_SinglePdDataFrameBatchType], List[_SingleNpArrayBatchType]]


def _method_names_from_param(spark_param_name: str) -> List[str]:
    """
    Returns getter and setter method names, per Spark ML conventions, for passed in attribute.
    """
    cap = spark_param_name[0].upper() + spark_param_name[1:]
    getter = f"get{cap}"
    setter = f"set{cap}"
    return [getter, setter]


def _unsupported_methods_attributes(clazz: Any) -> Set[str]:
    """
    Returns set of methods and attributes not supported by spark-rapids-ml for passed in class
    as determined from empty values in the dictionary returned by _param_mapping() invoked on the class.
    """
    if "_param_mapping" in [
        member_name for member_name, _ in inspect.getmembers(clazz, inspect.ismethod)
    ]:
        param_map = clazz._param_mapping()
        _unsupported_params = [k for k, v in param_map.items() if not v]
        _unsupported_methods: List[str] = sum(
            [_method_names_from_param(k) for k in _unsupported_params], []
        )
        methods_and_functions = inspect.getmembers(
            clazz,
            predicate=lambda member: inspect.isfunction(member)
            or inspect.ismethod(member),
        )
        _other_unsupported = [
            entry[0]
            for entry in methods_and_functions
            if entry and (entry[1].__doc__) == "Unsupported."
        ]
        return set(_unsupported_params + _unsupported_methods + _other_unsupported)
    else:
        return set()


def _get_spark_session() -> SparkSession:
    """Get or create spark session.
    Note: This function can only be invoked from driver side."""
    if TaskContext.get() is not None:
        # safety check.
        raise RuntimeError(
            "_get_spark_session should not be invoked from executor side."
        )

    # avoid the bug https://issues.apache.org/jira/browse/SPARK-38870
    # in spark < 3.4 when changing run time configs of active sessions
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    return SparkSession.builder.getOrCreate()


def _is_local(sc: SparkContext) -> bool:
    """Whether it is Spark local mode"""
    return sc._jsc.sc().isLocal() or sc.getConf().get("spark.master").startswith("local-cluster")  # type: ignore


def _is_standalone_or_localcluster(conf: SparkConf) -> bool:
    master = conf.get("spark.master")
    return master is not None and (
        master.startswith("spark://") or master.startswith("local-cluster")
    )


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
    import os

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        if os.environ["CUDA_VISIBLE_DEVICES"]:
            num_assigned = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            # when CUDA_VISIBLE_DEVICES is set and non-empty, use 0-th index entry
            gpu_id = 0
        else:
            raise RuntimeError(
                "Couldn't get gpu id since CUDA_VISIBLE_DEVICES is set to an empty string.  Please check the GPU resource configuration."
            )
    else:
        if task_context is None:
            # safety check.
            raise RuntimeError("_get_gpu_id should not be invoked from driver side.")
        resources = task_context.resources()
        if "gpu" not in resources:
            raise RuntimeError(
                "Couldn't get the gpu id, Please check the GPU resource configuration."
            )
        num_assigned = len(resources["gpu"].addresses)
        # return the first gpu id.
        gpu_id = int(resources["gpu"].addresses[0].strip())

    if num_assigned > 1:
        logger = get_logger(_get_gpu_id)
        logger.warning(
            f"Task got assigned {num_assigned} GPUs but using only 1.  This could be a waste of GPU resources."
        )

    return gpu_id


# When changing default rmm memory resources we retain the old ones
# in this global array singleton to so that any (C++) allocations using them can
# invoke the corresponding deallocate methods.  They will get cleaned up only when
# the process exits.  This avoids a segfault in the case of creating a new
# SAM resource with a smaller headroom.
_old_memory_resources = []

# keep track of last headroom to check if new sam mr is needed.
_last_sam_headroom_size = None


def _configure_memory_resource(
    uvm_enabled: bool = False,
    sam_enabled: bool = False,
    sam_headroom: Optional[int] = None,
) -> None:
    import cupy as cp
    import rmm
    from cuda.bindings import runtime
    from rmm.allocators.cupy import rmm_cupy_allocator

    global _last_sam_headroom_size

    _SYSTEM_MEMORY_SUPPORTED = rmm._cuda.gpu.getDeviceAttribute(  # type: ignore
        runtime.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess,
        rmm._cuda.gpu.getDevice(),  # type: ignore
    )

    if not _SYSTEM_MEMORY_SUPPORTED and sam_enabled:
        raise ValueError(
            "System allocated memory is not supported on this GPU. Please disable system allocated memory."
        )

    if uvm_enabled and sam_enabled:
        raise ValueError(
            "Both CUDA managed memory and system allocated memory cannot be enabled at the same time."
        )

    if sam_enabled and sam_headroom is None:
        if not type(rmm.mr.get_current_device_resource()) == type(
            rmm.mr.SystemMemoryResource()
        ):
            _old_memory_resources.append(rmm.mr.get_current_device_resource())
            _last_sam_headroom_size = None
            mr = rmm.mr.SystemMemoryResource()
            rmm.mr.set_current_device_resource(mr)
    elif sam_enabled and sam_headroom is not None:
        if sam_headroom != _last_sam_headroom_size or not type(
            rmm.mr.get_current_device_resource()
        ) == type(rmm.mr.SamHeadroomMemoryResource(headroom=sam_headroom)):
            _old_memory_resources.append(rmm.mr.get_current_device_resource())
            _last_sam_headroom_size = sam_headroom
            mr = rmm.mr.SamHeadroomMemoryResource(headroom=sam_headroom)
            rmm.mr.set_current_device_resource(mr)

    if uvm_enabled:
        if not type(rmm.mr.get_current_device_resource()) == type(
            rmm.mr.ManagedMemoryResource()
        ):
            _old_memory_resources.append(rmm.mr.get_current_device_resource())
            rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

    if sam_enabled or uvm_enabled:
        if not cp.cuda.get_allocator().__name__ == rmm_cupy_allocator.__name__:
            cp.cuda.set_allocator(rmm_cupy_allocator)

    if sam_enabled:
        import spark_rapids_ml.numpy_allocator


def _memadvise_cpu(data: Any, nbytes: int) -> None:
    """
    Advise data referenced by pointer to stay in cpu memory.
    For use with SAM to prevent migration of partial arrays staged in host memory to device during
    gpu concatenation.
    """
    import cuda
    import cupy as cp
    from packaging.version import parse

    # latest cupy 13.6 has a bug in the cuda13 version of the memadvise api so use low level
    # python bindings from cuda-python for that case.  Once the patch is released, we can revert to using the cupy api.
    if parse(cuda.bindings.__version__) < parse("13.0.0"):
        cp.cuda.runtime.memAdvise(data, nbytes, 3, -1)
    else:
        from cuda.bindings.runtime import (
            cudaMemLocation,
            cudaMemLocationType,
            cudaMemoryAdvise,
        )

        mem_location = cudaMemLocation()
        mem_location.type = cudaMemLocationType.cudaMemLocationTypeHost
        cuda.bindings.runtime.cudaMemAdvise(
            data,
            nbytes,
            cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation,
            mem_location,
        )


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


def _get_class_or_callable_name(cls_or_callable: Union[type, Callable]) -> str:
    """
    Return the class name.
    """
    return f"{cls_or_callable.__module__}.{cls_or_callable.__name__}"


class PartitionDescriptor:
    """
    Partition descriptor

    m: total number of rows across all workers
    n: total number of cols
    parts_rank_size: a sequence of (rank, rows per partitions)
    rank: rank to be mapped
    """

    def __init__(
        self,
        m: int,
        n: int,
        rank: int,
        parts_rank_size: List[Tuple[int, int]],
        partition_max_nnz: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.rank = rank
        self.parts_rank_size = parts_rank_size
        self.partition_max_nnz = partition_max_nnz

    @classmethod
    def build(
        cls,
        partition_rows: List[int],
        total_cols: int,
        partition_nnz: Optional[int] = None,
    ) -> "PartitionDescriptor":
        context = BarrierTaskContext.get()
        if context is None:
            # safety check.
            raise RuntimeError("build should not be invoked from driver side.")

        rank = context.partitionId()

        # prepare (parts, rank)

        import json

        rank_size = [(rank, size) for size in partition_rows]
        if partition_nnz is None:
            messages = context.allGather(message=json.dumps(rank_size))
            parts_rank_size = [item for pair in messages for item in json.loads(pair)]
            partition_max_nnz = None
        else:
            messages = context.allGather(message=json.dumps([rank_size, partition_nnz]))
            parts_rank_size = [item for msg in messages for item in json.loads(msg)[0]]
            partition_max_nnz = max([json.loads(msg)[1] for msg in messages])

        total_rows = sum(pair[1] for pair in parts_rank_size)

        return cls(total_rows, total_cols, rank, parts_rank_size, partition_max_nnz)


def _concat_and_free(
    array_list: Union[
        List["cp.ndarray"],
        List[np.ndarray],
        List[scipy.sparse.csr_matrix],
        List["cupyx.scipy.sparse.csr_matrix"],
    ],
    order: _ArrayOrder = "F",
) -> Union[
    "cp.ndarray", np.ndarray, scipy.sparse.csr_matrix, "cupyx.scipy.sparse.csr_matrix"
]:
    """
    concatenates a list of compatible numpy arrays into a 'order' ordered output array,
    in a memory efficient way.
    Note: frees list elements so do not reuse after calling.

    if the type of input arrays is scipy or cupyx csr_matrix, 'order' parameter will not be used.
    """
    import cupyx

    if len(array_list) == 1:
        return array_list[0]

    if isinstance(array_list[0], scipy.sparse.csr_matrix):
        concated = scipy.sparse.vstack(array_list)
    elif isinstance(array_list[0], cupyx.scipy.sparse.csr_matrix):
        concated = cupyx.scipy.sparse.vstack(array_list)
    else:
        import cupy as cp

        array_module = cp if isinstance(array_list[0], cp.ndarray) else np

        rows = sum(arr.shape[0] for arr in array_list)
        if len(array_list[0].shape) > 1:
            cols = array_list[0].shape[1]
            concat_shape: Tuple[int, ...] = (rows, cols)
        else:
            concat_shape = (rows,)
        d_type = array_list[0].dtype
        concated = array_module.empty(shape=concat_shape, order=order, dtype=d_type)
        array_module.concatenate(array_list, out=concated)
    del array_list[:]
    return concated


def _try_allocate_cp_empty_arrays(
    gpu_id: int,
    gpu_mem_ratio_for_data: float,
    dimension: int,
    dtype: np.dtype,
    array_order: str,
    has_label: bool,
    logger: logging.Logger,
    cuda_system_mem_enabled: bool,
) -> Tuple["cp.ndarray", Optional["cp.ndarray"]]:
    import cupy as cp

    device = cp.cuda.Device(gpu_id)
    free_mem, total_mem = device.mem_info
    nbytes_per_row = (dimension + 1 if has_label else 0) * np.dtype(dtype).itemsize

    # if sam is enabled, use the available host memory as well
    if cuda_system_mem_enabled:
        import psutil

        free_mem += psutil.virtual_memory().available

    target_mem = int(free_mem * gpu_mem_ratio_for_data)
    while target_mem >= 1_000_000:
        target_n_rows = target_mem // nbytes_per_row

        try:
            cp_features = cp.empty(
                shape=(target_n_rows, dimension), dtype=dtype, order=array_order
            )

            cp_label = cp.empty(shape=target_n_rows, dtype=dtype) if has_label else None

            logger.info(
                f"Reserved {target_mem / 1_000_000_000} GB GPU memory for training data (dim={dimension}, max_rows={target_n_rows:,}"
            )

            return (cp_features, cp_label)

        except cp.cuda.memory.OutOfMemoryError:
            logger.warning(f"OOM at {target_mem / 1_000_000_000} GB, reducing...")
            target_mem = int(target_mem * 0.9)
        except Exception as e:
            print("Unexpected error:", e)
            break

    raise ValueError("Failed to reserve GPU memory for training data.")


def _concat_with_reserved_gpu_mem(
    gpu_id: int,
    pdf_iter: Iterator[pd.DataFrame],
    gpu_mem_ratio_for_data: float,
    array_order: str,
    multi_col_names: Optional[List[str]],
    logger: logging.Logger,
    cuda_system_mem_enabled: bool,
) -> Tuple["cp.ndarray", Optional["cp.ndarray"], Optional[np.ndarray]]:
    # TODO: support sparse matrix
    # TODO: support row number

    assert array_order == "C", "F order array is currently not supported."

    assert gpu_mem_ratio_for_data > 0.0 and gpu_mem_ratio_for_data < 1.0

    import cupy as cp

    from spark_rapids_ml.core import alias

    first_batch = True
    num_rows_total = 0

    cp_label = None
    out_row_number = None

    for pdf in pdf_iter:
        # dense vector
        if multi_col_names:
            np_features: np.ndarray = np.array(pdf[multi_col_names], order=array_order)  # type: ignore
        else:
            np_features = np.array(
                list(pdf[alias.data]), order=array_order
            )  #  type: ignore
        np_label = pdf[alias.label].values if alias.label in pdf.columns else None
        np_row_number = (
            pdf[alias.row_number].values if alias.row_number in pdf.columns else None
        )

        if first_batch:
            first_batch = False

            dimension = np_features.shape[1]
            dtype = np_features.dtype
            has_label = True if np_label is not None else False
            cp_features, cp_label = _try_allocate_cp_empty_arrays(
                gpu_id,
                gpu_mem_ratio_for_data,
                dimension,
                dtype,
                array_order,
                has_label,
                logger,
                cuda_system_mem_enabled,
            )

        np_rows = np_features.shape[0]

        cp_features[num_rows_total : num_rows_total + np_rows, :] = cp.array(
            np_features, order=array_order
        )
        if np_label is not None:
            assert len(np_label) == np_rows
            assert cp_label is not None
            cp_label[num_rows_total : num_rows_total + np_rows] = cp.array(np_label)

        num_rows_total += np_features.shape[0]

    out_features = cp_features[0:num_rows_total]
    out_label = None if cp_label is None else cp_label[0:num_rows_total]
    return (out_features, out_label, out_row_number)


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
    elif dtype == np.int64:
        return "long"
    elif dtype == np.int16:
        return "short"
    elif dtype == np.int64:
        return "long"
    else:
        raise RuntimeError("Unsupported dtype, found ", dtype)


# similar to https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/utils.py
def get_logger(
    cls_or_callable: Union[type, Callable, str], level: str = "INFO"
) -> logging.Logger:
    """Gets a logger by name, or creates and configures it for the first time."""
    name = (
        cls_or_callable
        if isinstance(cls_or_callable, str)
        else _get_class_or_callable_name(cls_or_callable)
    )
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


def translate_tree(sc: SparkContext, impurity: str, model: Dict[str, Any]):  # type: ignore
    """Translate Treelite JSON representation to PySpark trees

    Converts Treelite JSON format to Spark MLlib tree format.
    
    Args:
        sc: SparkContext
        impurity: Impurity type ("gini", "entropy", or "variance")
        model: Treelite JSON model portion representing a single tree
        
    Returns:
        Spark tree 
        
 (see https://treelite.readthedocs.io/en/latest/tutorials/builder.html#example-regressor)
    Example TreeliteJson Tree:
    {
        "num_nodes": 5,
        "has_categorical_split": false,
        "nodes": [{
                "node_id": 0,
                "split_feature_id": 0,
                "default_left": true,
                "node_type": "numerical_test_node",
                "comparison_op": "<",
                "threshold": 5.0,
                "left_child": 1,
                "right_child": 2
            }, {
                "node_id": 1,
                "split_feature_id": 2,
                "default_left": false,
                "node_type": "numerical_test_node",
                "comparison_op": "<",
                "threshold": -3.0,
                "left_child": 3,
                "right_child": 4
            }, {
                "node_id": 2,
                "leaf_value": 0.6000000238418579
            }, {
                "node_id": 3,
                "leaf_value": -0.4000000059604645
            }, {
                "node_id": 4,
                "leaf_value": 1.2000000476837159
            }]
    }

    Spark tree,
             InternalNode {split{featureIndex=3, threshold=0.827687974732221}, gain = 0.41999999999999998}
             /         \
           left        right
           /             \
    LeafNode           LeafNode
    """
    tree = model

    root_id = 0
    nodes = tree["nodes"]

    # Create a mapping from node_id to node data
    node_map = {node["node_id"]: node for node in nodes}

    # Convert the tree starting from root
    spark_tree = _convert_treelite_node(sc, impurity, node_map, root_id)
    # spark_trees.append(spark_tree)

    return spark_tree


def _convert_treelite_node(sc: SparkContext, impurity: str, node_map: Dict[int, Dict[str, Any]], node_id: int):  # type: ignore
    """Convert a single Treelite node to Spark MLlib node

    Args:
        sc: SparkContext
        impurity: Impurity type
        node_map: Dictionary mapping node_id to node data
        node_id: ID of the node to convert

    Returns:
        Spark MLlib node (InternalNode or LeafNode)
    """
    node = node_map[node_id]

    # Check if this is a leaf node
    if "leaf_value" in node:
        # Convert leaf node
        leaf_model = {
            "leaf_value": (
                node["leaf_value"]
                if isinstance(node["leaf_value"], list)
                else [node["leaf_value"]]
            ),
            "instance_count": node.get("instance_count", 1),
        }
        return _create_leaf_node(sc, impurity, leaf_model)

    # This is an internal node
    left_child_id = node["left_child"]
    right_child_id = node["right_child"]

    # Convert children recursively
    left_child = _convert_treelite_node(sc, impurity, node_map, left_child_id)
    right_child = _convert_treelite_node(sc, impurity, node_map, right_child_id)

    # Create internal node model in the format expected by _create_internal_node
    internal_model = {
        "split_feature": node["split_feature_id"],
        "split_threshold": node["threshold"],
        "gain": node.get("gain", 0.0),  # Treelite doesn't always provide gain
        "instance_count": node.get("instance_count", 1),
        "yes": left_child_id,
        "no": right_child_id,
    }

    return _create_internal_node(sc, impurity, internal_model, left_child, right_child)


# to the XGBOOST _get_unwrap_udt_fn in https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/core.py
def _get_unwrap_udt_fn() -> Callable[[Union[Column, str]], Column]:
    try:
        from pyspark.sql.functions import unwrap_udt  # type: ignore

        return unwrap_udt
    except ImportError:
        pass

    try:
        from pyspark.databricks.sql.functions import unwrap_udt as databricks_unwrap_udt

        return databricks_unwrap_udt
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import pyspark `unwrap_udt` function. Please install pyspark>=3.4 "
            "or run on Databricks Runtime."
        ) from exc


from pyspark.ml.base import Estimator, Transformer


def setInputOrFeaturesCol(
    pstage: Union[Estimator, Transformer],
    features_col_value: Union[str, List[str]],
    label_col_value: Optional[str] = None,
) -> None:
    setter = (
        getattr(pstage, "setFeaturesCol")
        if hasattr(pstage, "setFeaturesCol")
        else getattr(pstage, "setInputCol")
    )
    setter(features_col_value)

    # clear to keep only one of cols and col set
    if isinstance(features_col_value, str):
        for col_name in {"featuresCols", "inputCols"}:
            if pstage.hasParam(col_name):
                pstage.clear(getattr(pstage, col_name))
    else:
        assert isinstance(features_col_value, List) and all(
            isinstance(x, str) for x in features_col_value
        )
        for col_name in {"featuresCol", "inputCol"}:
            if pstage.hasParam(col_name):
                pstage.clear(getattr(pstage, col_name))

    label_setter = (
        getattr(pstage, "setLabelCol") if hasattr(pstage, "setLabelCol") else None
    )
    if label_setter is not None and label_col_value is not None:
        label_setter(label_col_value)


def getInputOrFeaturesCols(est: Union[Estimator, Transformer]) -> str:
    getter = (
        getattr(est, "getFeaturesCol")
        if hasattr(est, "getFeaturesCol")
        else getattr(est, "getInputCol")
    )
    return getter()


def _standardize_dataset(
    data: FitInputType, pdesc: PartitionDescriptor, fit_intercept: bool
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """Inplace standardize the dataset feature and optionally label columns

    Args:
        data: dataset to standardize (including features and label)
        pdesc: Partition descriptor
        fit_intercept: Whether to fit intercept in calling fit function.

    Returns:
        Mean and standard deviation of features and label columns (latter is last element if present)
        Modifies data entries by replacing entries with standardized data on gpu.
        If data is already on gpu, modifies in place (i.e. no copy is made).
    """
    import cupy as cp

    mean_partials_labels = (
        cp.zeros(1, dtype=data[0][1].dtype) if data[0][1] is not None else None
    )
    mean_partials = [cp.zeros(pdesc.n, dtype=data[0][0].dtype), mean_partials_labels]
    for i in range(len(data)):
        _data = []
        for j in range(2):
            if data[i][j] is not None:

                if isinstance(data[i][j], cp.ndarray):
                    _data.append(data[i][j])  # type: ignore
                elif isinstance(data[i][j], np.ndarray):
                    _data.append(cp.array(data[i][j]))  # type: ignore
                elif isinstance(data[i][j], pd.DataFrame) or isinstance(
                    data[i][j], pd.Series
                ):
                    _data.append(cp.array(data[i][j].values))  # type: ignore
                else:
                    raise ValueError("Unsupported data type: ", type(data[i][j]))
                mean_partials[j] += _data[j].sum(axis=0) / pdesc.m  # type: ignore
            else:
                _data.append(None)
        data[i] = (_data[0], _data[1], data[i][2])  # type: ignore

    import json

    from pyspark import BarrierTaskContext

    context = BarrierTaskContext.get()

    def all_gather_then_sum(
        cp_array: cp.ndarray, dtype: Union[np.float32, np.float64]
    ) -> cp.ndarray:
        msgs = context.allGather(json.dumps(cp_array.tolist()))
        arrays = [json.loads(p) for p in msgs]
        array_sum = np.sum(arrays, axis=0).astype(dtype)
        return cp.array(array_sum)

    if mean_partials[1] is not None:
        mean_partial = cp.concatenate(mean_partials)  # type: ignore
    else:
        mean_partial = mean_partials[0]
    mean = all_gather_then_sum(mean_partial, mean_partial.dtype)

    _mean = (mean[:-1], mean[-1]) if mean_partials[1] is not None else (mean, None)

    var_partials_labels = (
        cp.zeros(1, dtype=data[0][1].dtype) if data[0][1] is not None else None
    )
    var_partials = [cp.zeros(pdesc.n, dtype=data[0][0].dtype), var_partials_labels]
    for i in range(len(data)):
        for j in range(2):
            if data[i][j] is not None and _mean[j] is not None:
                __data = data[i][j]
                __data -= _mean[j]  # type: ignore
                l2 = cp.linalg.norm(__data, ord=2, axis=0)
                var_partials[j] += l2 * l2 / (pdesc.m - 1)

    if var_partials[1] is not None:
        var_partial = cp.concatenate((var_partials[0], var_partials[1]))
    else:
        var_partial = var_partials[0]
    var = all_gather_then_sum(var_partial, var_partial.dtype)

    assert cp.all(
        var >= 0
    ), "numeric instable detected when calculating variance. Got negative variance"

    stddev = cp.sqrt(var)
    stddev_inv = cp.where(stddev != 0, 1.0 / stddev, 1.0)
    _stddev_inv = (
        (stddev_inv[:-1], stddev_inv[-1])
        if var_partials[1] is not None
        else (stddev_inv, None)
    )

    if fit_intercept is False:
        for i in range(len(data)):
            for j in range(2):
                if data[i][j] is not None and _mean[j] is not None:
                    __data = data[i][j]
                    __data += _mean[j]  # type: ignore

    for i in range(len(data)):
        for j in range(2):
            if data[i][j] is not None and _stddev_inv[j] is not None:
                __data = data[i][j]
                __data *= _stddev_inv[j]  # type: ignore

    return mean, stddev

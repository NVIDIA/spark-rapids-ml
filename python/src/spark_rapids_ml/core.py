#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
import json
import os
import threading
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from pyspark import RDD, SparkConf, TaskContext
from pyspark.ml import Estimator, Model
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.ml.param.shared import (
    HasLabelCol,
    HasOutputCol,
    HasPredictionCol,
    HasProbabilityCol,
    HasRawPredictionCol,
)
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)
from pyspark.ml.wrapper import JavaParams
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, struct
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegralType,
    Row,
    StructType,
)
from scipy.sparse import csr_matrix

from .common.cuml_context import CumlContext
from .metrics import EvalMetricInfo
from .params import _CumlParams
from .utils import (
    _ArrayOrder,
    _get_gpu_id,
    _get_spark_session,
    _is_local,
    _is_standalone_or_localcluster,
    dtype_to_pyspark_type,
    get_logger,
)

if TYPE_CHECKING:
    import cudf
    import cupy as cp
    from pyspark.ml._typing import ParamMap

CumlT = Any

_SinglePdDataFrameBatchType = Tuple[
    pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]
]
_SingleNpArrayBatchType = Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]

# FitInputType is type of [(feature, label), ...]
FitInputType = Union[List[_SinglePdDataFrameBatchType], List[_SingleNpArrayBatchType]]

# TransformInput type
TransformInputType = Union["cudf.DataFrame", np.ndarray]

# Function to construct cuml instances on the executor side
_ConstructFunc = Callable[..., Union[CumlT, List[CumlT]]]

# Function to do the inference using cuml instance constructed by _ConstructFunc
_TransformFunc = Union[
    Callable[[CumlT, TransformInputType], pd.DataFrame],
    Callable[[CumlT, TransformInputType], "cp.ndarray"],
]

# Function to do evaluation based on the prediction result got from _TransformFunc
_EvaluateFunc = Callable[
    [
        TransformInputType,  # input dataset with label column
        "cp.ndarray",  # inferred dataset with prediction column
    ],
    pd.DataFrame,
]

# Global constant for defining column alias
Alias = namedtuple(
    "Alias",
    (
        "featureVectorType",
        "featureVectorSize",
        "featureVectorIndices",
        "data",
        "label",
        "row_number",
    ),
)

# Avoid same naming. `echo spark-rapids-ml | base64` = c3BhcmstcmFwaWRzLW1sCg==
col_name_unique_tag = "c3BhcmstcmFwaWRzLW1sCg=="

alias = Alias(
    f"vector_type_{col_name_unique_tag}",
    f"vector_size_{col_name_unique_tag}",
    f"vector_indices_{col_name_unique_tag}",
    f"cuml_values_{col_name_unique_tag}",
    "cuml_label",
    "unique_id",
)

# Global prediction names
Pred = namedtuple(
    "Pred", ("prediction", "probability", "model_index", "raw_prediction")
)
pred = Pred("prediction", "probability", "model_index", "raw_prediction")

# Global parameter alias used by core and subclasses.
ParamAlias = namedtuple(
    "ParamAlias",
    ("cuml_init", "handle", "num_cols", "part_sizes", "loop", "fit_multiple_params"),
)
param_alias = ParamAlias(
    "cuml_init", "handle", "num_cols", "part_sizes", "loop", "fit_multiple_params"
)

CumlModel = TypeVar("CumlModel", bound="_CumlModel")

from .utils import _get_unwrap_udt_fn


# similar to the XGBOOST _get_unwrapped_vec_cols in https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/core.py
def _get_unwrapped_vec_cols(feature_col: Column, float32_inputs: bool) -> List[Column]:
    unwrap_udt = _get_unwrap_udt_fn()
    features_unwrapped_vec_col = unwrap_udt(feature_col)

    # After a `pyspark.ml.linalg.VectorUDT` type column being unwrapped, it becomes
    # a pyspark struct type column, the struct fields are:
    #  - `type`: byte
    #  - `size`: int
    #  - `indices`: array<int>
    #  - `values`: array<double>
    # For sparse vector, `type` field is 0, `size` field means vector dimension,
    # `indices` field is the array of active element indices, `values` field
    # is the array of active element values.
    # For dense vector, `type` field is 1, `size` and `indices` fields are None,
    # `values` field is the array of the vector element values.

    values_col = features_unwrapped_vec_col.values
    if float32_inputs is True:
        values_col = values_col.cast(ArrayType(FloatType()))

    return [
        features_unwrapped_vec_col.type.alias(alias.featureVectorType),
        features_unwrapped_vec_col.size.alias(alias.featureVectorSize),
        features_unwrapped_vec_col.indices.alias(alias.featureVectorIndices),
        values_col.alias(alias.data),
    ]


def _use_sparse_in_cuml(dataset: DataFrame) -> bool:
    return (
        alias.featureVectorType in dataset.schema.fieldNames()
        and alias.featureVectorSize in dataset.schema.fieldNames()
        and alias.featureVectorIndices in dataset.schema.fieldNames()
    )  # use sparse array in cuml only if features vectorudt column was unwrapped


# similar to the XGBOOST _read_csr_matrix_from_unwrapped_spark_vec in https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/data.py
def _read_csr_matrix_from_unwrapped_spark_vec(part: pd.DataFrame) -> csr_matrix:
    # variables for constructing csr_matrix
    csr_indices_list, csr_indptr_list, csr_values_list = [], [0], []

    n_features = 0

    # TBD: investigate if there is a more efficient 'vectorized' approach to doing this. Iterating in python can be slow
    for vec_type, vec_size_, vec_indices, vec_values in zip(
        part[alias.featureVectorType],
        part[alias.featureVectorSize],
        part[alias.featureVectorIndices],
        part[alias.data],
    ):
        if vec_type == 0:
            # sparse vector
            vec_size = int(vec_size_)
            csr_indices = vec_indices
            csr_values = vec_values
        else:
            # dense vector
            # Note: According to spark ML VectorUDT format,
            # when type field is 1, the size field is also empty.
            # we need to check the values field to get vector length.
            vec_size = len(vec_values)
            csr_indices = np.arange(vec_size, dtype=np.int32)
            csr_values = vec_values

        if n_features == 0:
            n_features = vec_size
        assert n_features == vec_size, "all vectors must be of the same dimension"

        csr_indices_list.append(csr_indices)
        csr_indptr_list.append(csr_indptr_list[-1] + len(csr_indices))
        assert len(csr_indptr_list) == 1 + len(csr_indices_list)

        csr_values_list.append(csr_values)

    assert len(csr_indptr_list) == 1 + len(part)

    csr_indptr_arr = np.array(csr_indptr_list)
    csr_indices_arr = np.concatenate(csr_indices_list)
    csr_values_arr = np.concatenate(csr_values_list)

    return csr_matrix(
        (csr_values_arr, csr_indices_arr, csr_indptr_arr), shape=(len(part), n_features)
    )


class _CumlEstimatorWriter(MLWriter):
    """
    Write the parameters of _CumlEstimator to the file
    """

    def __init__(self, instance: "_CumlEstimator") -> None:
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        DefaultParamsWriter.saveMetadata(
            self.instance,
            path,
            self.sc,
            extraMetadata={
                "_cuml_params": self.instance._cuml_params,
                "_num_workers": self.instance._num_workers,
                "_float32_inputs": self.instance._float32_inputs,
            },
        )  # type: ignore


class _CumlEstimatorReader(MLReader):
    """
    Instantiate the _CumlEstimator from the file.
    """

    def __init__(self, cls: Type) -> None:
        super().__init__()
        self.estimator_cls = cls

    def load(self, path: str) -> "_CumlEstimator":
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        cuml_estimator = self.estimator_cls()
        cuml_estimator._resetUid(metadata["uid"])
        DefaultParamsReader.getAndSetParams(cuml_estimator, metadata)
        cuml_estimator._cuml_params = metadata["_cuml_params"]
        cuml_estimator._num_workers = metadata["_num_workers"]
        cuml_estimator._float32_inputs = metadata["_float32_inputs"]
        return cuml_estimator


class _CumlModelWriter(MLWriter):
    """
    Write the parameters of _CumlModel to the file
    """

    def __init__(self, instance: "_CumlModel") -> None:
        super().__init__()
        self.instance: "_CumlModel" = instance

    def saveImpl(self, path: str) -> None:
        DefaultParamsWriter.saveMetadata(
            self.instance,
            path,
            self.sc,
            extraMetadata={
                "_cuml_params": self.instance._cuml_params,
                "_num_workers": self.instance._num_workers,
                "_float32_inputs": self.instance._float32_inputs,
            },
        )
        data_path = os.path.join(path, "data")
        model_attributes = self.instance._get_model_attributes()
        model_attributes_str = json.dumps(model_attributes)
        self.sc.parallelize([model_attributes_str], 1).saveAsTextFile(data_path)


class _CumlModelReader(MLReader):
    """
    Instantiate the _CumlModel from the file.
    """

    def __init__(self, cls: Type) -> None:
        super().__init__()
        self.model_cls = cls

    def load(self, path: str) -> "_CumlEstimator":
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        data_path = os.path.join(path, "data")
        model_attr_str = self.sc.textFile(data_path).collect()[0]
        model_attr_dict = json.loads(model_attr_str)
        instance = self.model_cls(**model_attr_dict)
        DefaultParamsReader.getAndSetParams(instance, metadata)
        instance._cuml_params = metadata["_cuml_params"]
        instance._num_workers = metadata["_num_workers"]
        instance._float32_inputs = metadata["_float32_inputs"]
        return instance


class _CumlCommon(MLWritable, MLReadable):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _get_gpu_device(
        context: Optional[TaskContext], is_local: bool, is_transform: bool = False
    ) -> int:
        """
        Get gpu device according to the spark task resources.

        If it is local mode, we use partition id as gpu id for training
        and (partition id ) % gpus for transform.
        """
        # Get the GPU ID from resources
        assert context is not None

        import cupy

        if is_local:
            partition_id = context.partitionId()
            if is_transform:
                # For transform local mode, default the gpu_id to (partition id ) % gpus.
                total_gpus = cupy.cuda.runtime.getDeviceCount()
                gpu_id = partition_id % total_gpus
            else:
                gpu_id = partition_id
        else:
            gpu_id = _get_gpu_id(context)

        return gpu_id

    @staticmethod
    def _set_gpu_device(
        context: Optional[TaskContext], is_local: bool, is_transform: bool = False
    ) -> None:
        """
        Set gpu device according to the spark task resources.

        If it is local mode, we use partition id as gpu id for training
        and (partition id ) % gpus for transform.
        """
        # Get the GPU ID from resources
        assert context is not None

        import cupy

        gpu_id = _CumlCommon._get_gpu_device(context, is_local, is_transform)

        cupy.cuda.Device(gpu_id).use()

    @staticmethod
    def _initialize_cuml_logging(verbose: Optional[Union[bool, int]]) -> None:
        """Initializes the logger for cuML.

        Parameters
        ----------
        verbose : Optional[Union[bool, int]]
            If True, sets the log_level to 5.  If integer value, sets the log_level to the value.
        """
        if verbose is not None:
            from cuml.common import logger as cuml_logger

            # below is from https://docs.rapids.ai/api/cuml/stable/api.html#verbosity-levels
            if isinstance(verbose, bool):
                if verbose:
                    log_level = 5
                else:
                    log_level = 4
            elif isinstance(verbose, int):
                log_level = verbose
            else:
                raise ValueError(f"invalid value for verbose parameter: {verbose}")

            cuml_logger.set_level(log_level)

    def _pyspark_class(self) -> Optional[ABCMeta]:
        """
        Subclass should override to return corresponding pyspark.ml class
        Ex. logistic regression should return pyspark.ml.classification.LogisticRegression
        Return None if no corresponding class in pyspark, e.g. knn
        """
        raise NotImplementedError(
            "pyspark.ml class corresponding to estimator not specified."
        )


class _CumlCaller(_CumlParams, _CumlCommon):
    """
    This class is responsible for calling cuml function (e.g. fit or kneighbor) on pyspark dataframe,
    to run a multi-node multi-gpu algorithm on the dataframe. A function usually comes from a multi-gpu cuml class,
    such as cuml.decomposition.pca_mg.PCAMG or cuml.neighbors.nearest_neighbors_mg.NearestNeighborsMG.
    This class converts dataframe into cuml input type, and leverages NCCL or UCX for communicator. To use this class,
    developers can override the key methods including _out_schema(...) and _get_cuml_fit_func(...). Examples can be found in
    spark_rapids_ml.clustering.KMeans and spark_rapids_ml.regression.LinearRegression.
    """

    def __init__(self) -> None:
        super().__init__()
        self._initialize_cuml_params()

    @abstractmethod
    def _out_schema(self) -> Union[StructType, str]:
        """
        The output schema of the estimator, which will be used to
        construct the returning pandas dataframe
        """
        raise NotImplementedError()

    def _repartition_dataset(self, dataset: DataFrame) -> DataFrame:
        """
        Repartition the dataset to the desired number of workers.
        """
        return dataset.repartition(self.num_workers)

    def _pre_process_data(self, dataset: DataFrame) -> Tuple[
        List[Column],
        Optional[List[str]],
        int,
        Union[Type[FloatType], Type[DoubleType]],
    ]:
        select_cols = []

        # label column will be cast to feature type if needed.
        feature_type: Union[Type[FloatType], Type[DoubleType]] = FloatType

        input_col, input_cols = self._get_input_columns()

        if input_col is not None:
            # Single Column
            input_datatype = dataset.schema[input_col].dataType
            first_record = dataset.first()

            if isinstance(input_datatype, ArrayType):
                # Array type
                if (
                    isinstance(input_datatype.elementType, DoubleType)
                    and not self._float32_inputs
                ):
                    select_cols.append(col(input_col).alias(alias.data))
                    feature_type = DoubleType
                elif (
                    isinstance(input_datatype.elementType, DoubleType)
                    and self._float32_inputs
                ):
                    select_cols.append(
                        col(input_col).cast(ArrayType(feature_type())).alias(alias.data)
                    )
                else:
                    # FloatType array
                    select_cols.append(col(input_col).alias(alias.data))
            elif isinstance(input_datatype, VectorUDT):
                vector_element_type = "float32" if self._float32_inputs else "float64"
                first_vectorudt_type = (
                    DenseVector
                    if first_record is None
                    or type(first_record[input_col]) is DenseVector
                    else SparseVector
                )
                use_sparse = self.hasParam(
                    "enable_sparse_data_optim"
                ) and self.getOrDefault("enable_sparse_data_optim")

                if use_sparse is True or (
                    use_sparse is None and first_vectorudt_type is SparseVector
                ):
                    # Sparse Vector type
                    select_cols += _get_unwrapped_vec_cols(
                        col(input_col), self._float32_inputs
                    )
                else:
                    # Dense Vector type
                    assert use_sparse is False or (
                        use_sparse is None and first_vectorudt_type is DenseVector
                    )
                    select_cols.append(
                        vector_to_array(col(input_col), vector_element_type).alias(alias.data)  # type: ignore
                    )

                if not self._float32_inputs:
                    feature_type = DoubleType
            else:
                raise ValueError("Unsupported input type.")

            dimension = len(first_record[input_col])  # type: ignore

        elif input_cols is not None:
            # if self._float32_inputs is False and if any columns are double type, convert all to double type
            # otherwise convert all to float type
            any_double_types = any(
                [isinstance(dataset.schema[c].dataType, DoubleType) for c in input_cols]
            )
            if not self._float32_inputs and any_double_types:
                feature_type = DoubleType
            dimension = len(input_cols)
            for c in input_cols:
                col_type = dataset.schema[c].dataType
                if (
                    isinstance(col_type, IntegralType)
                    or isinstance(col_type, FloatType)
                    or isinstance(col_type, DoubleType)
                ):
                    if not isinstance(col_type, feature_type):
                        select_cols.append(col(c).cast(feature_type()).alias(c))
                    else:
                        select_cols.append(col(c))
                else:
                    raise ValueError(
                        "All columns must be integral types or float/double types."
                    )
        else:
            # should never get here
            raise Exception("Unable to determine input column(s).")

        return select_cols, input_cols, dimension, feature_type

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        """
        If enable or disable communication layer (NCCL or UCX).
        Return (False, False) if no communication layer is required.
        Return (True, False) if only NCCL is required.
        Return (True, True) if UCX is required. Cuml UCX backend currently also requires NCCL.
        """
        return (True, False)

    def _validate_parameters(self) -> None:
        cls_name = self._pyspark_class()

        if cls_name is not None:
            pyspark_est = cls_name()
            # Both pyspark and cuml may have a parameter with the same name,
            # but cuml might have additional optional values that can be set.
            # If we transfer these cuml-specific values to the Spark JVM,
            # it would result in an exception.
            # To avoid this issue, we skip transferring these parameters
            # since the mapped parameters have been validated in _get_cuml_mapping_value.
            cuml_est = self.copy()
            cuml_params = cuml_est._param_value_mapping().keys()
            param_mapping = cuml_est._param_mapping()
            pyspark_params = [k for k, v in param_mapping.items() if v in cuml_params]
            for p in pyspark_params:
                cuml_est.clear(cuml_est.getParam(p))

            cuml_est._copyValues(pyspark_est)
            # validate the parameters
            pyspark_est._transfer_params_to_java()

            del pyspark_est
            del cuml_est

    @abstractmethod
    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        """
        Subclass must implement this function to return a cuml fit function that will be
        sent to executor to run.

        Eg,

        def _get_cuml_fit_func(self, dataset: DataFrame, extra_params: Optional[List[Dict[str, Any]]] = None):
            ...
            def _cuml_fit(df: CumlInputType, params: Dict[str, Any]) -> Dict[str, Any]:
                "" "
                df:  a sequence of (X, Y)
                params: a series of parameters stored in dictionary,
                    especially, the parameters of __init__ is stored in params[param_alias.init]
                "" "
                ...
            ...

            return _cuml_fit

        _get_cuml_fit_func itself runs on the driver side, while the returned _cuml_fit will
        run on the executor side.
        """
        raise NotImplementedError()

    def _call_cuml_fit_func(
        self,
        dataset: DataFrame,
        partially_collect: bool = True,
        paramMaps: Optional[Sequence["ParamMap"]] = None,
    ) -> RDD:
        """
        Fits a model to the input dataset. This is called by the default implementation of fit.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            input dataset

        Returns
        -------
        :class:`Transformer`
            fitted model
        """
        self._validate_parameters()

        cls = self.__class__

        select_cols, multi_col_names, dimension, _ = self._pre_process_data(dataset)

        num_workers = self.num_workers

        dataset = dataset.select(*select_cols)

        if dataset.rdd.getNumPartitions() != num_workers:
            dataset = self._repartition_dataset(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )
        if cuda_managed_mem_enabled:
            get_logger(cls).info("CUDA managed memory enabled.")

        # parameters passed to subclass
        params: Dict[str, Any] = {
            param_alias.cuml_init: self.cuml_params,
        }

        # Convert the paramMaps into cuml paramMaps
        fit_multiple_params = []
        if paramMaps is not None:
            for paramMap in paramMaps:
                tmp_fit_multiple_params = {}
                for k, v in paramMap.items():
                    name = self._get_cuml_param(k.name, False)
                    assert name is not None
                    tmp_fit_multiple_params[name] = self._get_cuml_mapping_value(
                        name, v
                    )
                fit_multiple_params.append(tmp_fit_multiple_params)
        params[param_alias.fit_multiple_params] = fit_multiple_params

        cuml_fit_func = self._get_cuml_fit_func(
            dataset, None if len(fit_multiple_params) == 0 else fit_multiple_params
        )

        array_order = self._fit_array_order()

        cuml_verbose = self.cuml_params.get("verbose", False)

        use_sparse_array = _use_sparse_in_cuml(dataset)

        (enable_nccl, require_ucx) = self._require_nccl_ucx()

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            import cupy as cp
            import cupyx
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()
            logger = get_logger(cls)

            # set gpu device
            _CumlCommon._set_gpu_device(context, is_local)

            if cuda_managed_mem_enabled:
                import rmm
                from rmm.allocators.cupy import rmm_cupy_allocator

                rmm.reinitialize(
                    managed_memory=True,
                    devices=_CumlCommon._get_gpu_device(context, is_local),
                )
                cp.cuda.set_allocator(rmm_cupy_allocator)

            _CumlCommon._initialize_cuml_logging(cuml_verbose)

            # handle the input
            # inputs = [(X, Optional(y)), (X, Optional(y))]
            logger.info("Loading data into python worker memory")
            inputs = []
            sizes = []

            for pdf in pdf_iter:
                sizes.append(pdf.shape[0])
                if multi_col_names:
                    features = np.array(pdf[multi_col_names], order=array_order)
                elif use_sparse_array:
                    # sparse vector
                    features = _read_csr_matrix_from_unwrapped_spark_vec(pdf)
                else:
                    # dense vector
                    features = np.array(list(pdf[alias.data]), order=array_order)

                # experiments indicate it is faster to convert to numpy array and then to cupy array than directly
                # invoking cupy array on the list
                if cuda_managed_mem_enabled and use_sparse_array is False:
                    features = cp.array(features)

                label = pdf[alias.label] if alias.label in pdf.columns else None
                row_number = (
                    pdf[alias.row_number] if alias.row_number in pdf.columns else None
                )
                inputs.append((features, label, row_number))

            if cuda_managed_mem_enabled and use_sparse_array is True:
                concated_nnz = sum(triplet[0].nnz for triplet in inputs)  # type: ignore
                if concated_nnz > np.iinfo(np.int32).max:
                    logger.warn(
                        f"The number of non-zero values of a partition exceeds the int32 index dtype. \
                        cupyx csr_matrix currently does not support int64 indices (https://github.com/cupy/cupy/issues/3513); \
                        keeping as scipy csr_matrix to avoid overflow."
                    )
                else:
                    inputs = [
                        (cupyx.scipy.sparse.csr_matrix(row[0]), row[1], row[2])
                        for row in inputs
                    ]

            if len(sizes) == 0 or all(sz == 0 for sz in sizes):
                raise RuntimeError(
                    "A python worker received no data.  Please increase amount of data or use fewer workers."
                )

            logger.info("Initializing cuml context")
            with CumlContext(
                partition_id, num_workers, context, enable_nccl, require_ucx
            ) as cc:
                params[param_alias.handle] = cc.handle
                params[param_alias.part_sizes] = sizes
                params[param_alias.num_cols] = dimension
                params[param_alias.loop] = cc._loop

                logger.info("Invoking cuml fit")

                # pyspark uses sighup to kill python workers gracefully, and for some reason
                # the signal handler for sighup needs to be explicitly reset at this point
                # to avoid having SIGHUP be swallowed during a usleep call in the nccl library.
                # this helps avoid zombie surviving python workers when some workers fail.
                import signal

                signal.signal(signal.SIGHUP, signal.SIG_DFL)

                # call the cuml fit function
                # *note*: cuml_fit_func may delete components of inputs to free
                # memory.  do not rely on inputs after this call.
                result = cuml_fit_func(inputs, params)
                logger.info("Cuml fit complete")

            if partially_collect == True:
                if enable_nccl:
                    context.barrier()

                if context.partitionId() == 0:
                    yield pd.DataFrame(data=result)
            else:
                yield pd.DataFrame(data=result)

        pipelined_rdd = (
            dataset.mapInPandas(_train_udf, schema=self._out_schema())  # type: ignore
            .rdd.barrier()
            .mapPartitions(lambda x: x)
        )

        return pipelined_rdd

    def _fit_array_order(self) -> _ArrayOrder:
        """
        preferred array order for converting single column array type to numpy arrays: "C" or "F"
        """
        return "F"


class _FitMultipleIterator(Generic[CumlModel]):
    """
    Used by default implementation of Estimator.fitMultiple to produce models in a thread safe
    iterator. This class handles the gpu version of fitMultiple where all param maps should be
    fit in a single pass.

    Parameters
    ----------
    fitMultipleModels : function
        Callable[[], CumlModel] which fits multiple models to a dataset in a single pass.
    numModels : int
        Number of models this iterator should produce.

    Notes
    -----
    See :py:meth:`Estimator.fitMultiple` for more info.
    """

    def __init__(
        self, fitMultipleModels: Callable[[], List[CumlModel]], numModels: int
    ):
        self.fitMultipleModels = fitMultipleModels
        self.numModels = numModels
        self.counter = 0
        self.lock = threading.Lock()
        self.models: List[CumlModel] = []

    def __iter__(self) -> Iterator[Tuple[int, CumlModel]]:
        return self

    def __next__(self) -> Tuple[int, CumlModel]:
        with self.lock:
            index = self.counter
            if index >= self.numModels:
                raise StopIteration("No models remaining.")
            if index == 0:
                self.models = self.fitMultipleModels()
                assert len(self.models) == self.numModels
            self.counter += 1
        return index, self.models[index]

    def next(self) -> Tuple[int, CumlModel]:
        return self.__next__()


class _CumlEstimator(Estimator, _CumlCaller):
    """
    The common estimator to handle the fit callback (_fit). It should:
    1. set the default parameters
    2. validate the parameters
    3. prepare the dataset
    4. train and return CUML model
    5. create the pyspark model
    """

    # used by keywords_only
    _input_kwargs: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger(self.__class__)

    @abstractmethod
    def _create_pyspark_model(self, result: Row) -> "_CumlModel":
        """
        Create the model according to the collected Row
        """
        raise NotImplementedError()

    def _enable_fit_multiple_in_single_pass(self) -> bool:
        """flag to indicate if fitMultiple in a single pass is supported.
        If not, fallback to super().fitMultiple"""
        return False

    def fitMultiple(
        self, dataset: DataFrame, paramMaps: Sequence["ParamMap"]
    ) -> Iterator[Tuple[int, "_CumlModel"]]:
        """
        Fits multiple models to the input dataset for all param maps in a single pass.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            input dataset.
        paramMaps : :py:class:`collections.abc.Sequence`
            A Sequence of param maps.

        Returns
        -------
        :py:class:`_FitMultipleIterator`
            A thread safe iterable which contains one model for each param map. Each
            call to `next(modelIterator)` will return `(index, model)` where model was fit
            using `paramMaps[index]`. `index` values may not be sequential.
        """

        if self._enable_fit_multiple_in_single_pass():
            estimator = self.copy()

            def fitMultipleModels() -> List["_CumlModel"]:
                return estimator._fit_internal(dataset, paramMaps)

            return _FitMultipleIterator(fitMultipleModels, len(paramMaps))
        else:
            return super().fitMultiple(dataset, paramMaps)

    def _skip_stage_level_scheduling(self, spark_version: str, conf: SparkConf) -> bool:
        """Check if stage-level scheduling is not needed,
        return true to skip stage-level scheduling"""

        if spark_version < "3.4.0":
            self.logger.info(
                "Stage-level scheduling in spark-rapids-ml requires spark version 3.4.0+"
            )
            return True

        if "3.4.0" <= spark_version < "3.5.1" and not _is_standalone_or_localcluster(
            conf
        ):
            self.logger.info(
                "For Spark %s, Stage-level scheduling in spark-rapids-ml requires spark "
                "standalone or local-cluster mode",
                spark_version,
            )
            return True

        executor_cores = conf.get("spark.executor.cores")
        executor_gpus = conf.get("spark.executor.resource.gpu.amount")
        if executor_cores is None or executor_gpus is None:
            self.logger.info(
                "Stage-level scheduling in spark-rapids-ml requires spark.executor.cores, "
                "spark.executor.resource.gpu.amount to be set."
            )
            return True

        if int(executor_cores) == 1:
            # there will be only 1 task running at any time.
            self.logger.info(
                "Stage-level scheduling in spark-rapids-ml requires spark.executor.cores > 1 "
            )
            return True

        if int(executor_gpus) > 1:
            # For spark.executor.resource.gpu.amount > 1, we suppose user knows how to configure
            # to make spark-rapids-ml run successfully.
            self.logger.info(
                "Stage-level scheduling in spark-rapids-ml will not work "
                "when spark.executor.resource.gpu.amount>1"
            )
            return True

        task_gpu_amount = conf.get("spark.task.resource.gpu.amount")

        if task_gpu_amount is None:
            # The ETL tasks will not grab a gpu when spark.task.resource.gpu.amount is not set,
            # but with stage-level scheduling, we can make training task grab the gpu.
            return False

        if float(task_gpu_amount) == float(executor_gpus):
            # spark.executor.resource.gpu.amount=spark.task.resource.gpu.amount "
            # results in only 1 task running at a time, which may cause perf issue.
            return True

        # We can enable stage-level scheduling
        return False

    def _try_stage_level_scheduling(self, rdd: RDD) -> RDD:
        ss = _get_spark_session()
        sc = ss.sparkContext

        if _is_local(sc) or self._skip_stage_level_scheduling(ss.version, sc.getConf()):
            return rdd

        # executor_cores will not be None
        executor_cores = ss.sparkContext.getConf().get("spark.executor.cores")
        assert executor_cores is not None

        from pyspark.resource.profile import ResourceProfileBuilder
        from pyspark.resource.requests import TaskResourceRequests

        # each training task requires cpu cores > total executor cores/2 which can
        # ensure each training task be sent to different executor.
        #
        # Please note that we can't set task_cores to the value which is smaller than total executor cores/2
        # because only task_gpus can't ensure the tasks be sent to different executor even task_gpus=1.0
        #
        # If spark-rapids enabled. we don't allow other ETL task running alongside training task to avoid OOM
        spark_plugins = ss.conf.get("spark.plugins", " ")
        assert spark_plugins is not None
        spark_rapids_sql_enabled = ss.conf.get("spark.rapids.sql.enabled", "true")
        assert spark_rapids_sql_enabled is not None

        task_cores = (
            int(executor_cores)
            if "com.nvidia.spark.SQLPlugin" in spark_plugins
            and "true" == spark_rapids_sql_enabled.lower()
            else (int(executor_cores) // 2) + 1
        )
        # task_gpus means how many slots per gpu address the task requires,
        # it does mean how many gpus it would like to require, so it can be any value of (0, 0.5] or 1.
        task_gpus = 1.0

        treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build

        self.logger.info(
            f"Training tasks require the resource(cores={task_cores}, gpu={task_gpus})"
        )

        return rdd.withResources(rp)

    def _fit_internal(
        self, dataset: DataFrame, paramMaps: Optional[Sequence["ParamMap"]]
    ) -> List["_CumlModel"]:
        """Fit multiple models according to the parameters maps"""
        pipelined_rdd = self._call_cuml_fit_func(
            dataset=dataset,
            partially_collect=True,
            paramMaps=paramMaps,
        )

        pipelined_rdd = self._try_stage_level_scheduling(pipelined_rdd)

        self.logger.info(
            f"Training spark-rapids-ml with {self.num_workers} worker(s) ..."
        )
        rows = pipelined_rdd.collect()
        self.logger.info("Finished training")

        models: List["_CumlModel"] = [None]  # type: ignore
        if paramMaps is not None:
            models = [None] * len(paramMaps)  # type: ignore

        for index in range(len(models)):
            model = self._create_pyspark_model(rows[index])
            model._num_workers = self._num_workers
            model._float32_inputs = self._float32_inputs

            if paramMaps is not None:
                self._copyValues(model, paramMaps[index])
            else:
                self._copyValues(model)

            self._copy_cuml_params(model)  # type: ignore

            models[index] = model  # type: ignore

        return models

    def _fit(self, dataset: DataFrame) -> "_CumlModel":
        """fit only 1 model"""
        return self._fit_internal(dataset, None)[0]

    def write(self) -> MLWriter:
        return _CumlEstimatorWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlEstimatorReader(cls)

    def _supportsTransformEvaluate(self, evaluator: Evaluator) -> bool:
        """If supporting _transformEvaluate in a single pass based on the evaluator

        Please note that this function should only be used in CrossValidator for quick
        fallback if unsupported."""
        return False


class _CumlEstimatorSupervised(_CumlEstimator, HasLabelCol):
    """
    Base class for Cuml Supervised machine learning.
    """

    def _pre_process_label(
        self, dataset: DataFrame, feature_type: Union[Type[FloatType], Type[DoubleType]]
    ) -> Column:
        """Convert label according to feature type by default"""
        label_name = self.getLabelCol()
        label_datatype = dataset.schema[label_name].dataType
        if isinstance(label_datatype, (IntegralType, FloatType, DoubleType)):
            if isinstance(label_datatype, IntegralType) or not isinstance(
                label_datatype, feature_type
            ):
                label_col = col(label_name).cast(feature_type()).alias(alias.label)
            else:
                label_col = col(label_name).alias(alias.label)
        else:
            raise ValueError(
                "Label column must be integral types or float/double types."
            )

        return label_col

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[
        List[Column], Optional[List[str]], int, Union[Type[FloatType], Type[DoubleType]]
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = super()._pre_process_data(dataset)

        select_cols.append(self._pre_process_label(dataset, feature_type))

        return select_cols, multi_col_names, dimension, feature_type


class _CumlModel(Model, _CumlParams, _CumlCommon):
    """
    Abstract class for spark-rapids-ml models that are fitted by spark-rapids-ml estimators.
    """

    def __init__(
        self,
        *,
        dtype: Optional[str] = None,
        n_cols: Optional[int] = None,
        **model_attributes: Any,
    ) -> None:
        """
        Subclass must pass the model attributes which will be saved in model persistence.
        """
        super().__init__()
        self._initialize_cuml_params()

        # model_data is the native data which will be saved for model persistence
        self._model_attributes = model_attributes
        self._model_attributes["dtype"] = dtype
        self._model_attributes["n_cols"] = n_cols
        self.dtype = dtype
        self.n_cols = n_cols

    def cpu(self) -> Model:
        """Return the equivalent PySpark CPU model"""
        raise NotImplementedError()

    def _get_model_attributes(self) -> Optional[Dict[str, Any]]:
        """Return model attributes as a dictionary."""
        return self._model_attributes

    @classmethod
    def _from_row(cls, model_attributes: Row):  # type: ignore
        """
        Default to pass all the attributes of the model to the model constructor,
        So please make sure if the constructor can accept all of them.
        """
        attr_dict = model_attributes.asDict()
        return cls(**attr_dict)

    @abstractmethod
    def _get_cuml_transform_func(
        self,
        dataset: DataFrame,
        eval_metric_info: Optional[EvalMetricInfo] = None,
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc]]:
        """
        Subclass must implement this function to return three functions,
        1. a function to construct cuml counterpart instance
        2. a function to transform the dataset
        3. an optional function to evaluate.

        Eg,

        def _get_cuml_transform_func(self, dataset: DataFrame):
            ...
            def _construct_cuml_object() -> CumlT
                ...
            def _cuml_transform(cuml_obj: CumlT, df: Union[pd.DataFrame, np.ndarray]) ->pd.DataFrame:
                ...
            def _evaluate(input_df: Union[pd.DataFrame, np.ndarray], transformed_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
                ...
            ...

            # please note that if eval_metric_info is None, the evaluate function will be None.
            return _construct_cuml_object, _cuml_transform, _evaluate

        _get_cuml_transform_func itself runs on the driver side, while the returned
        _construct_cuml_object and _cuml_transform, _evaluate will run on the executor side.
        """
        raise NotImplementedError()

    def _transform_array_order(self) -> _ArrayOrder:
        """
        preferred array order for converting single column array type to numpy arrays: "C" or "F"
        """
        return "F"

    @abstractmethod
    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        """
        The output schema of the model, which will be used to
        construct the returning pandas dataframe
        """
        raise NotImplementedError()

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[DataFrame, List[str], bool, List[str]]:
        """Pre-handle the dataset before transform.

        Please note that, this function just transforms the input column if necessary, and
        it will keep the unused columns.

        return (dataset, list of feature names, bool value to indicate if it is multi-columns input, list of temporary columns to be dropped)
        """
        select_cols = []
        tmp_cols = []
        input_is_multi_cols = True

        input_col, input_cols = self._get_input_columns()

        if input_col is not None:
            input_col_type = dataset.schema[input_col].dataType
            if isinstance(input_col_type, VectorUDT):
                # Vector type
                vector_element_type = "float32" if self._float32_inputs else "float64"

                if self.hasParam("enable_sparse_data_optim") is False:
                    use_cuml_sparse = False
                elif self.getOrDefault("enable_sparse_data_optim") is None:
                    first_record = dataset.first()
                    first_vectorudt_type = (
                        DenseVector
                        if first_record is None
                        or type(first_record[input_col]) is DenseVector
                        else SparseVector
                    )
                    use_cuml_sparse = first_vectorudt_type is SparseVector
                else:
                    use_cuml_sparse = self.getOrDefault("enable_sparse_data_optim")

                if use_cuml_sparse:
                    type_col, size_col, indices_col, data_col = _get_unwrapped_vec_cols(
                        col(input_col), self._float32_inputs
                    )

                    dataset = dataset.withColumn(alias.featureVectorType, type_col)

                    dataset = dataset.withColumn(alias.featureVectorSize, size_col)

                    dataset = dataset.withColumn(
                        alias.featureVectorIndices, indices_col
                    )

                    dataset = dataset.withColumn(alias.data, data_col.alias(alias.data))

                    for col_name in [
                        alias.featureVectorType,
                        alias.featureVectorSize,
                        alias.featureVectorIndices,
                        alias.data,
                    ]:
                        select_cols.append(col_name)
                        tmp_cols.append(col_name)
                else:
                    dataset = dataset.withColumn(
                        alias.data,
                        vector_to_array(col(input_col), vector_element_type),
                    )
                    select_cols.append(alias.data)
                    tmp_cols.append(alias.data)
            elif isinstance(input_col_type, ArrayType):
                if (
                    isinstance(input_col_type.elementType, DoubleType)
                    and not self._float32_inputs
                ):
                    select_cols.append(input_col)
                elif (
                    isinstance(input_col_type.elementType, DoubleType)
                    and self._float32_inputs
                ):
                    dataset = dataset.withColumn(
                        alias.data, col(input_col).cast(ArrayType(FloatType()))
                    )
                    select_cols.append(alias.data)
                    tmp_cols.append(alias.data)
                else:
                    # FloatType array
                    select_cols.append(input_col)
            elif not isinstance(input_col_type, ArrayType):
                # not Array type
                raise ValueError("Unsupported input type.")
            input_is_multi_cols = False
        elif input_cols is not None:
            any_double_types = any(
                [isinstance(dataset.schema[c].dataType, DoubleType) for c in input_cols]
            )
            feature_type: Union[Type[FloatType], Type[DoubleType]] = FloatType
            if not self._float32_inputs and any_double_types:
                feature_type = DoubleType

            for c in input_cols:
                col_type = dataset.schema[c].dataType
                if (
                    isinstance(col_type, IntegralType)
                    or isinstance(col_type, FloatType)
                    or isinstance(col_type, DoubleType)
                ):
                    if not isinstance(col_type, feature_type):
                        tmp_input_col = f"{c}_{col_name_unique_tag}"
                        select_cols.append(tmp_input_col)
                        tmp_cols.append(tmp_input_col)
                    else:
                        select_cols.append(c)
                else:
                    raise ValueError(
                        "All columns must be integral types or float/double types."
                    )

            taglen = len(col_name_unique_tag) + 1
            added_tmp_cols = [
                col(c[:-taglen]).cast(feature_type()).alias(c) for c in tmp_cols
            ]
            dataset = dataset.select("*", *added_tmp_cols)
        else:
            # should never get here
            raise Exception("Unable to determine input column(s).")

        return dataset, select_cols, input_is_multi_cols, tmp_cols

    def _concate_pdf_batches(self) -> bool:
        return False

    def _transform_evaluate_internal(
        self,
        dataset: DataFrame,
        schema: Union[StructType, str],
        eval_metric_info: Optional[EvalMetricInfo] = None,
    ) -> DataFrame:
        """Internal API to support transform and evaluation in a single pass"""
        dataset, select_cols, input_is_multi_cols, _ = self._pre_process_data(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        (
            construct_cuml_object_func,
            cuml_transform_func,
            evaluate_func,
        ) = self._get_cuml_transform_func(dataset, eval_metric_info)
        if evaluate_func:
            dataset = dataset.select(alias.label, *select_cols)
        else:
            dataset = dataset.select(*select_cols)

        array_order = self._transform_array_order()

        use_sparse_array = _use_sparse_in_cuml(dataset)
        concate_pdf_batches = self._concate_pdf_batches()

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )
        if cuda_managed_mem_enabled:
            get_logger(self.__class__).info("CUDA managed memory enabled.")

        def _transform_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import TaskContext

            context = TaskContext.get()

            _CumlCommon._set_gpu_device(context, is_local, True)

            if cuda_managed_mem_enabled:
                import cupy as cp
                import rmm
                from rmm.allocators.cupy import rmm_cupy_allocator

                rmm.reinitialize(
                    managed_memory=True,
                    devices=_CumlCommon._get_gpu_device(
                        context, is_local, is_transform=True
                    ),
                )
                cp.cuda.set_allocator(rmm_cupy_allocator)

            # Construct the cuml counterpart object
            cuml_instance = construct_cuml_object_func()
            cuml_objects = (
                cuml_instance if isinstance(cuml_instance, list) else [cuml_instance]
            )

            def process_pdf_iter(
                pdf_iter: Iterator[pd.DataFrame],
            ) -> Iterator[pd.DataFrame]:
                if concate_pdf_batches is False:
                    for pdf in pdf_iter:
                        yield pdf
                else:
                    pdfs = [pdf for pdf in pdf_iter]
                    if (len(pdfs)) > 0:
                        yield pd.concat(pdfs, ignore_index=True)

            processed_pdf_iter = process_pdf_iter(pdf_iter)
            has_row_number = None
            for pdf in processed_pdf_iter:
                if has_row_number is None:
                    has_row_number = True if alias.row_number in pdf.columns else False
                else:
                    assert has_row_number == (alias.row_number in pdf.columns)

                for index, cuml_object in enumerate(cuml_objects):
                    if has_row_number:
                        data = cuml_transform_func(cuml_object, pdf)
                    elif use_sparse_array:
                        features = _read_csr_matrix_from_unwrapped_spark_vec(
                            pdf[select_cols]
                        )
                        data = cuml_transform_func(cuml_object, features)
                    elif input_is_multi_cols:
                        data = cuml_transform_func(cuml_object, pdf[select_cols])
                    else:
                        nparray = np.array(list(pdf[select_cols[0]]), order=array_order)
                        data = cuml_transform_func(cuml_object, nparray)

                    # Evaluate the dataset if necessary.
                    if evaluate_func is not None:
                        data = evaluate_func(pdf, data)
                        data[pred.model_index] = index

                    yield data

        return dataset.mapInPandas(_transform_udf, schema=schema)  # type: ignore

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            input dataset.

        Returns
        -------
        :py:class:`pyspark.sql.DataFrame`
            transformed dataset
        """
        return self._transform_evaluate_internal(
            dataset, schema=self._out_schema(dataset.schema)
        )

    def write(self) -> MLWriter:
        return _CumlModelWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlModelReader(cls)

    def _transformEvaluate(
        self,
        dataset: DataFrame,
        evaluator: Evaluator,
        params: Optional["ParamMap"] = None,
    ) -> List[float]:
        """
        Transforms and evaluates the input dataset with optional parameters in a single pass.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            a dataset that contains labels/observations and predictions
        evaluator: :py:class:`pyspark.ml.evaluation.Evaluator`
            an evaluator user intends to use
        params : dict, optional
            an optional param map that overrides embedded params

        Returns
        -------
        list of float
            metrics
        """
        raise NotImplementedError()

    @classmethod
    def _combine(cls: Type["_CumlModel"], models: List["_CumlModel"]) -> "_CumlModel":
        """Combine a list of same type models into a model"""
        raise NotImplementedError()


class _CumlModelWithColumns(_CumlModel):
    """Cuml base model for generating extra predicted columns"""

    def _is_single_pred(self, input_schema: StructType) -> bool:
        """Indicate if the transform is only predicting 1 column"""
        schema = self._out_schema(input_schema)
        if isinstance(schema, str):
            return False if "," in schema else True
        elif isinstance(schema, StructType):
            return False if len(schema.names) > 1 else True

    def _has_probability_col(self) -> bool:
        """This API is needed and can be overwritten by subclass which
        hasn't implemented predict probability yet"""

        return True if isinstance(self, HasProbabilityCol) else False

    def _has_raw_pred_col(self) -> bool:
        """This API is needed and can be overwritten by subclass which
        hasn't implemented predict raw yet"""

        return True if isinstance(self, HasRawPredictionCol) else False

    def _use_prob_as_raw_pred_col(self) -> bool:
        """This API is needed and can be overwritten by subclass which
        doesn't support raw predictions in cuml to use copy of probability
        column instead.
        """

        return False

    def _get_prediction_name(self) -> str:
        """Different algos have different prediction names,
        eg, PCA: value of outputCol param, RF/LR/Kmeans: value of predictionCol name"""
        if isinstance(self, HasPredictionCol):
            return self.getPredictionCol()
        elif isinstance(self, HasOutputCol):
            return self.getOutputCol()
        else:
            raise ValueError("Please set predictionCol or outputCol")

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """This version of transform is directly adding extra columns to the dataset"""
        dataset, select_cols, input_is_multi_cols, tmp_cols = self._pre_process_data(
            dataset
        )

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        (
            construct_cuml_object_func,
            cuml_transform_func,
            _,
        ) = self._get_cuml_transform_func(dataset)

        array_order = self._transform_array_order()

        use_sparse_array = _use_sparse_in_cuml(dataset)

        output_schema = self._out_schema(dataset.schema)

        @pandas_udf(output_schema)  # type: ignore
        def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            from pyspark import TaskContext

            context = TaskContext.get()
            _CumlCommon._set_gpu_device(context, is_local, True)
            cuml_objects = construct_cuml_object_func()
            cuml_object = (
                cuml_objects[0] if isinstance(cuml_objects, list) else cuml_objects
            )
            for pdf in iterator:
                if use_sparse_array:
                    data = _read_csr_matrix_from_unwrapped_spark_vec(pdf[select_cols])
                elif not input_is_multi_cols:
                    data = np.array(list(pdf[select_cols[0]]), order=array_order)
                else:
                    data = pdf[select_cols]
                # for normal transform, we don't allow multiple models.
                res = cuml_transform_func(cuml_object, data)
                del data
                yield res

        pred_name = self._get_prediction_name()
        pred_col = predict_udf(struct(*select_cols))

        if self._is_single_pred(dataset.schema):
            output_schema_str = (
                output_schema
                if isinstance(output_schema, str)
                else output_schema.simpleString()
            )
            if (
                "array<float>" in output_schema_str
                or "array<double>" in output_schema_str
            ):
                input_col, input_cols = self._get_input_columns()
                if input_col is not None:
                    input_datatype = dataset.schema[input_col].dataType
                    if isinstance(input_datatype, VectorUDT):
                        pred_col = array_to_vector(pred_col)

            return dataset.withColumn(pred_name, pred_col).drop(*tmp_cols)
        else:
            pred_struct_col_name = f"_prediction_struct_{col_name_unique_tag}"
            dataset = dataset.withColumn(pred_struct_col_name, pred_col)

            # 1. Add prediction in the base class
            dataset = dataset.withColumn(
                pred_name, getattr(col(pred_struct_col_name), pred.prediction)
            )

            # 2. Handle probability columns
            if self._has_probability_col():
                probability_col = self.getOrDefault("probabilityCol")
                dataset = dataset.withColumn(
                    probability_col,
                    array_to_vector(
                        getattr(col(pred_struct_col_name), pred.probability)
                    ),
                )
            # 2a. Handle raw prediction - for algos that have it in spark but not yet supported in cuml,
            # we duplicate probability col for interop with default raw prediction col
            # in spark evaluators. i.e. auc works equivalently with probabilities.
            # TBD replace with rawPredictions in individual algos as support is added
            if self._has_raw_pred_col():
                raw_pred_col = self.getOrDefault("rawPredictionCol")
                if self._use_prob_as_raw_pred_col():
                    dataset = dataset.withColumn(
                        raw_pred_col,
                        col(probability_col),
                    )
                else:
                    # class supports raw predictions from cuml layer
                    dataset = dataset.withColumn(
                        raw_pred_col,
                        array_to_vector(
                            getattr(col(pred_struct_col_name), pred.raw_prediction)
                        ),
                    )

            # 3. Drop the unused column
            dataset = dataset.drop(pred_struct_col_name)

            return dataset.drop(*tmp_cols)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        assert self.dtype is not None

        schema = f"{pred.prediction} double"
        if self._has_probability_col():
            schema = f"{schema}, {pred.probability} array<double>"
            if self._has_raw_pred_col() and not self._use_prob_as_raw_pred_col():
                schema = f"{schema}, {pred.raw_prediction} array<double>"
        else:
            schema = f"double"

        return schema


class _CumlModelWithPredictionCol(_CumlModelWithColumns, HasPredictionCol):
    """Cuml base model with prediction col"""

    @property  # type: ignore[misc]
    def numFeatures(self) -> int:
        """
        Returns the number of features the model was trained on. If unknown, returns -1
        """

        num_features = self.n_cols if self.n_cols else -1
        return num_features

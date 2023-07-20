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
import json
import os
import threading
from abc import abstractmethod
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
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
from pyspark import RDD, TaskContext
from pyspark.ml import Estimator, Model
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param.shared import (
    HasLabelCol,
    HasOutputCol,
    HasPredictionCol,
    HasProbabilityCol,
)
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)
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

from .common.cuml_context import CumlContext
from .params import _CumlParams
from .utils import (
    _ArrayOrder,
    _get_gpu_id,
    _get_spark_session,
    _is_local,
    dtype_to_pyspark_type,
    get_logger,
)

if TYPE_CHECKING:
    import cudf
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
_TransformFunc = Callable[[CumlT, TransformInputType], pd.DataFrame]

# Function to do evaluation based on the prediction result got from _TransformFunc
_EvaluateFunc = Callable[
    [
        TransformInputType,  # input dataset with label column
        TransformInputType,  # inferred dataset with prediction column
    ],
    pd.DataFrame,
]

# Global constant for defining column alias
Alias = namedtuple("Alias", ("data", "label", "row_number"))
alias = Alias("cuml_values", "cuml_label", "unique_id")

# Global prediction names
Pred = namedtuple("Pred", ("prediction", "probability", "model_index"))
pred = Pred("prediction", "probability", "model_index")

# Global parameter alias used by core and subclasses.
ParamAlias = namedtuple(
    "ParamAlias",
    ("cuml_init", "handle", "num_cols", "part_sizes", "loop", "fit_multiple_params"),
)
param_alias = ParamAlias(
    "cuml_init", "handle", "num_cols", "part_sizes", "loop", "fit_multiple_params"
)

CumlModel = TypeVar("CumlModel", bound="_CumlModel")

# Global parameter used by core and subclasses.
TransformEvaluate = namedtuple("TransformEvaluate", ("transform", "transform_evaluate"))
transform_evaluate = TransformEvaluate("transform", "transform_evaluate")


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
            },
        )
        data_path = os.path.join(path, "data")
        model_attributes = self.instance.get_model_attributes()
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
        return instance


class _CumlCommon(MLWritable, MLReadable):
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger(self.__class__)

    @staticmethod
    def set_gpu_device(
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

        cupy.cuda.Device(gpu_id).use()

    @staticmethod
    def initialize_cuml_logging(verbose: Optional[Union[bool, int]]) -> None:
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
        self.initialize_cuml_params()

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

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[
        List[Column], Optional[List[str]], int, Union[Type[FloatType], Type[DoubleType]]
    ]:
        select_cols = []

        # label column will be cast to feature type if needed.
        feature_type: Union[Type[FloatType], Type[DoubleType]] = FloatType

        input_col, input_cols = self._get_input_columns()

        if input_col is not None:
            # Single Column
            input_datatype = dataset.schema[input_col].dataType

            if isinstance(input_datatype, ArrayType):
                # Array type
                select_cols.append(col(input_col).alias(alias.data))
                if isinstance(input_datatype.elementType, DoubleType):
                    feature_type = DoubleType
            elif isinstance(input_datatype, VectorUDT):
                # Vector type
                select_cols.append(
                    vector_to_array(col(input_col)).alias(alias.data)  # type: ignore
                )
                feature_type = DoubleType
            else:
                raise ValueError("Unsupported input type.")

            dimension = len(dataset.first()[input_col])  # type: ignore

        elif input_cols is not None:
            dimension = len(input_cols)
            for c in input_cols:
                col_type = dataset.schema[c].dataType
                if isinstance(col_type, IntegralType):
                    # Convert integral type to float.
                    select_cols.append(col(c).cast(feature_type()).alias(c))
                elif isinstance(col_type, FloatType):
                    select_cols.append(col(c))
                elif isinstance(col_type, DoubleType):
                    select_cols.append(col(c))
                    feature_type = DoubleType
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

    def _use_fit_generator(self) -> bool:
        """
        If the fit func is implemented as a generator function to return data row-by-row in the output RDD.
        """
        return False

    @abstractmethod
    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
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

    def _get_cuml_fit_generator_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]], Generator[Dict[str, Any], None, None]
    ]:
        """
        Alternative to _get_cuml_fit_func() that returns a generator function. Used when
        row-by-row data generation is desired in the _call_cuml_fit_func() output RDD.

        If _use_fit_generator() is set to True, subclass must implement this function.
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

        use_generator = self._use_fit_generator()

        if not use_generator:
            cuml_fit_func = self._get_cuml_fit_func(
                dataset, None if len(fit_multiple_params) == 0 else fit_multiple_params
            )
        else:
            cuml_fit_func = self._get_cuml_fit_generator_func(dataset, None)  # type: ignore

        array_order = self._fit_array_order()

        cuml_verbose = self.cuml_params.get("verbose", False)

        (enable_nccl, require_ucx) = self._require_nccl_ucx()

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import BarrierTaskContext

            logger = get_logger(cls)
            logger.info("Initializing cuml context")

            import cupy as cp

            if cuda_managed_mem_enabled:
                import rmm
                from rmm.allocators.cupy import rmm_cupy_allocator

                rmm.reinitialize(managed_memory=True)
                cp.cuda.set_allocator(rmm_cupy_allocator)

            _CumlCommon.initialize_cuml_logging(cuml_verbose)

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()

            # set gpu device
            _CumlCommon.set_gpu_device(context, is_local)

            with CumlContext(
                partition_id, num_workers, context, enable_nccl, require_ucx
            ) as cc:
                # handle the input
                # inputs = [(X, Optional(y)), (X, Optional(y))]
                logger.info("Loading data into python worker memory")
                inputs = []
                sizes = []
                for pdf in pdf_iter:
                    sizes.append(pdf.shape[0])
                    if multi_col_names:
                        features = np.array(pdf[multi_col_names], order=array_order)
                    else:
                        features = np.array(list(pdf[alias.data]), order=array_order)
                    # experiments indicate it is faster to convert to numpy array and then to cupy array than directly
                    # invoking cupy array on the list
                    if cuda_managed_mem_enabled:
                        features = cp.array(features)

                    label = pdf[alias.label] if alias.label in pdf.columns else None
                    row_number = (
                        pdf[alias.row_number]
                        if alias.row_number in pdf.columns
                        else None
                    )
                    inputs.append((features, label, row_number))

                params[param_alias.handle] = cc.handle
                params[param_alias.part_sizes] = sizes
                params[param_alias.num_cols] = dimension
                params[param_alias.loop] = cc._loop

                logger.info("Invoking cuml fit")

                # call the cuml fit function
                # *note*: cuml_fit_func may delete components of inputs to free
                # memory.  do not rely on inputs after this call.
                result = cuml_fit_func(inputs, params)
                logger.info("Cuml fit complete")

            if use_generator:
                for row in result:
                    yield row
                return

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

    def _fit_internal(
        self, dataset: DataFrame, paramMaps: Optional[Sequence["ParamMap"]]
    ) -> List["_CumlModel"]:
        """Fit multiple models according to the parameters maps"""
        pipelined_rdd = self._call_cuml_fit_func(
            dataset=dataset,
            partially_collect=True,
            paramMaps=paramMaps,
        )
        rows = pipelined_rdd.collect()

        models: List["_CumlModel"] = [None]  # type: ignore
        if paramMaps is not None:
            models = [None] * len(paramMaps)  # type: ignore

        for index in range(len(models)):
            model = self._create_pyspark_model(rows[index])
            model._num_workers = self._num_workers

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
        self.initialize_cuml_params()

        # model_data is the native data which will be saved for model persistence
        self._model_attributes = model_attributes
        self._model_attributes["dtype"] = dtype
        self._model_attributes["n_cols"] = n_cols
        self.dtype = dtype
        self.n_cols = n_cols

    def cpu(self) -> Model:
        """Return the equivalent PySpark CPU model"""
        raise NotImplementedError()

    def get_model_attributes(self) -> Optional[Dict[str, Any]]:
        """Return model attributes as a dictionary."""
        return self._model_attributes

    @classmethod
    def from_row(cls, model_attributes: Row):  # type: ignore
        """
        Default to pass all the attributes of the model to the model constructor,
        So please make sure if the constructor can accept all of them.
        """
        attr_dict = model_attributes.asDict()
        return cls(**attr_dict)

    @abstractmethod
    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
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

            # please note that if category is transform, the evaluate function will be ignored.
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
    ) -> Tuple[DataFrame, List[str], bool]:
        """Pre-handle the dataset before transform.

        Please note that, this function just transforms the input column if necessary, and
        it will keep the unused columns.

        return (dataset, list of feature names, bool value to indicate if it is multi-columns input)
        """
        select_cols = []
        input_is_multi_cols = True

        input_col, input_cols = self._get_input_columns()

        if input_col is not None:
            if isinstance(dataset.schema[input_col].dataType, VectorUDT):
                # Vector type
                # Avoid same naming. `echo spark-rapids-ml | base64` = c3BhcmstcmFwaWRzLW1sCg==
                tmp_name = f"{alias.data}_c3BhcmstcmFwaWRzLW1sCg=="
                dataset = (
                    dataset.withColumnRenamed(input_col, tmp_name)
                    .withColumn(input_col, vector_to_array(col(tmp_name)))
                    .drop(tmp_name)
                )
            elif not isinstance(dataset.schema[input_col].dataType, ArrayType):
                # Array type
                raise ValueError("Unsupported input type.")
            select_cols.append(input_col)
            input_is_multi_cols = False
        elif input_cols is not None:
            select_cols.extend(input_cols)
        else:
            # should never get here
            raise Exception("Unable to determine input column(s).")

        return dataset, select_cols, input_is_multi_cols

    def _transform_evaluate_internal(
        self, dataset: DataFrame, schema: Union[StructType, str]
    ) -> DataFrame:
        """Internal API to support transform and evaluation in a single pass"""
        dataset, select_cols, input_is_multi_cols = self._pre_process_data(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        (
            construct_cuml_object_func,
            cuml_transform_func,
            evaluate_func,
        ) = self._get_cuml_transform_func(
            dataset, transform_evaluate.transform_evaluate
        )

        array_order = self._transform_array_order()

        def _transform_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import TaskContext

            context = TaskContext.get()

            _CumlCommon.set_gpu_device(context, is_local, True)

            # Construct the cuml counterpart object
            cuml_instance = construct_cuml_object_func()
            cuml_objects = (
                cuml_instance if isinstance(cuml_instance, list) else [cuml_instance]
            )

            # TODO try to concatenate all the data and do the transform.
            for pdf in pdf_iter:
                for index, cuml_object in enumerate(cuml_objects):
                    # Transform the dataset
                    if input_is_multi_cols:
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
        dataset, select_cols, input_is_multi_cols = self._pre_process_data(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        (
            construct_cuml_object_func,
            cuml_transform_func,
            _,
        ) = self._get_cuml_transform_func(dataset)

        array_order = self._transform_array_order()

        @pandas_udf(self._out_schema(dataset.schema))  # type: ignore
        def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            from pyspark import TaskContext

            context = TaskContext.get()
            _CumlCommon.set_gpu_device(context, is_local, True)
            cuml_objects = construct_cuml_object_func()
            cuml_object = (
                cuml_objects[0] if isinstance(cuml_objects, list) else cuml_objects
            )
            for pdf in iterator:
                if not input_is_multi_cols:
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
            return dataset.withColumn(pred_name, pred_col)
        else:
            # Avoid same naming. `echo sparkcuml | base64` = c3BhcmtjdW1sCg==
            pred_struct_col_name = "_prediction_struct_c3BhcmtjdW1sCg=="
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

            # 3. Drop the unused column
            dataset = dataset.drop(pred_struct_col_name)

            return dataset

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        assert self.dtype is not None

        pyspark_type = dtype_to_pyspark_type(self.dtype)

        schema = f"{pred.prediction} {pyspark_type}"
        if self._has_probability_col():
            schema = f"{schema}, {pred.probability} array<{pyspark_type}>"
        else:
            schema = f"{pyspark_type}"

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

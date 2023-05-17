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
from abc import abstractmethod
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from pyspark import RDD, TaskContext
from pyspark.ml import Estimator, Model
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
    from cuml.cluster.kmeans_mg import KMeansMG
    from cuml.decomposition.pca_mg import PCAMG

    CumlT = TypeVar("CumlT", PCAMG, KMeansMG)
else:
    CumlT = Any

_SinglePdDataFrameBatchType = Tuple[
    pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]
]
_SingleNpArrayBatchType = Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
# CumlInputType is type of [(feature, label), ...]
CumlInputType = Union[List[_SinglePdDataFrameBatchType], List[_SingleNpArrayBatchType]]


# Global constant for defining column alias
Alias = namedtuple("Alias", ("data", "label", "row_number"))
alias = Alias("cuml_values", "cuml_label", "unique_id")

# Global prediction names
Pred = namedtuple("Pred", ("prediction", "probability"))
pred = Pred("prediction", "probability")

# Global parameter alias used by core and subclasses.
ParamAlias = namedtuple(
    "ParamAlias", ("cuml_init", "handle", "num_cols", "part_sizes", "loop")
)
param_alias = ParamAlias("cuml_init", "handle", "num_cols", "part_sizes", "loop")


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

    @abstractmethod
    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        """
        Subclass must implement this function to return a cuml fit function that will be
        sent to executor to run.

        Eg,

        def _get_cuml_fit_func(self, dataset: DataFrame):
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
        self, dataset: DataFrame, partially_collect: bool = True
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

        params: Dict[str, Any] = {
            param_alias.cuml_init: self.cuml_params,
        }

        cuml_fit_func = self._get_cuml_fit_func(dataset)
        array_order = self._fit_array_order()

        cuml_verbose = self.cuml_params.get("verbose", False)

        (enable_nccl, require_ucx) = self._require_nccl_ucx()

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import BarrierTaskContext

            logger = get_logger(cls)
            logger.info("Initializing cuml context")

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
                        features = pdf[multi_col_names]
                    else:
                        features = np.array(list(pdf[alias.data]), order=array_order)
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


class _CumlEstimator(Estimator, _CumlCaller):
    """
    The common estimator to handle the fit callback (_fit). It should:
    1. set the default parameters
    2. validate the parameters
    3. prepare the dataset
    4. train and return CUML model
    5. create the pyspark model
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _create_pyspark_model(self, result: Row) -> "_CumlModel":
        """
        Create the model according to the collected Row
        """
        raise NotImplementedError()

    def _fit(self, dataset: DataFrame) -> "_CumlModel":
        pipelined_rdd = self._call_cuml_fit_func(
            dataset=dataset, partially_collect=True
        )
        ret = pipelined_rdd.collect()[0]

        model = self._create_pyspark_model(ret)
        model._num_workers = self._num_workers
        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore
        return model

    def write(self) -> MLWriter:
        return _CumlEstimatorWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlEstimatorReader(cls)


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
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union["cudf.DataFrame", np.ndarray]], pd.DataFrame],
    ]:
        """
        Subclass must implement this function to return two functions,
        1. a function to construct cuml counterpart instance
        2. a function to transform the dataset

        Eg,

        def _get_cuml_transform_func(self, dataset: DataFrame):
            ...
            def _construct_cuml_object() -> CumlT
                ...
            def _cuml_transform(cuml_obj: CumlT, df: Union[pd.DataFrame, np.ndarray]) ->pd.DataFrame:
                ...
            ...

            return _construct_cuml_object, _cuml_transform

        _get_cuml_transform_func itself runs on the driver side, while the returned
        _construct_cuml_object and _cuml_transform will run on the executor side.
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
        """Pre-handle the dataset before transform."""
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
        dataset, select_cols, input_is_multi_cols = self._pre_process_data(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        (
            construct_cuml_object_func,
            cuml_transform_func,
        ) = self._get_cuml_transform_func(dataset)

        array_order = self._transform_array_order()

        def _transform_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import TaskContext

            context = TaskContext.get()

            _CumlCommon.set_gpu_device(context, is_local, True)

            # Construct the cuml counterpart object
            cuml_object = construct_cuml_object_func()

            # Transform the dataset
            if input_is_multi_cols:
                for pdf in pdf_iter:
                    yield cuml_transform_func(cuml_object, pdf[select_cols])
            else:
                for pdf in pdf_iter:
                    nparray = np.array(list(pdf[select_cols[0]]), order=array_order)
                    yield cuml_transform_func(cuml_object, nparray)

        return dataset.mapInPandas(
            _transform_udf, schema=self._out_schema(dataset.schema)  # type: ignore
        )

    def write(self) -> MLWriter:
        return _CumlModelWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlModelReader(cls)


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
        ) = self._get_cuml_transform_func(dataset)

        array_order = self._transform_array_order()

        @pandas_udf(self._out_schema(dataset.schema))  # type: ignore
        def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            from pyspark import TaskContext

            context = TaskContext.get()
            _CumlCommon.set_gpu_device(context, is_local, True)
            cuml_object = construct_cuml_object_func()
            for pdf in iterator:
                if not input_is_multi_cols:
                    data = np.array(list(pdf[select_cols[0]]), order=array_order)
                else:
                    data = pdf[select_cols]
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


class _CumlModelSupervised(_CumlModelWithColumns, HasPredictionCol):
    """Cuml base model for supervised machine learning Eg, RF/LR"""

    @property  # type: ignore[misc]
    def numFeatures(self) -> int:
        """
        Returns the number of features the model was trained on. If unknown, returns -1
        """

        num_features = self.n_cols if self.n_cols else -1
        return num_features

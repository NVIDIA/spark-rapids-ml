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

import cudf
import numpy as np
import pandas as pd
from pyspark import TaskContext
from pyspark.ml import Estimator, Model
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasLabelCol,
    HasOutputCol,
    HasOutputCols,
    HasPredictionCol,
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

from spark_rapids_ml.common.cuml_context import CumlContext
from spark_rapids_ml.utils import (
    _get_class_name,
    _get_default_params_from_func,
    _get_gpu_id,
    _get_spark_session,
    _is_local,
    dtype_to_pyspark_type,
    get_logger,
)

INIT_PARAMETERS_NAME = "init"

if TYPE_CHECKING:
    from cuml.cluster.kmeans_mg import KMeansMG
    from cuml.decomposition.pca_mg import PCAMG

    CumlT = TypeVar("CumlT", PCAMG, KMeansMG)
else:
    CumlT = Any

_SinglePdDataFrameBatchType = Tuple[pd.DataFrame, Optional[pd.DataFrame]]
_SingleNpArrayBatchType = Tuple[np.ndarray, Optional[np.ndarray]]
# CumlInputType is type of [(feature, label), ...]
CumlInputType = Union[List[_SinglePdDataFrameBatchType], List[_SingleNpArrayBatchType]]


# Global constant for defining column alias
Alias = namedtuple("Alias", ("data", "label"))
alias = Alias("cuml_values", "cuml_label")


class _CumlEstimatorWriter(MLWriter):
    """
    Write the parameters of _CumlEstimator to the file
    """

    def __init__(self, instance: "_CumlEstimator") -> None:
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        DefaultParamsWriter.saveMetadata(self.instance, path, self.sc)  # type: ignore


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
        return cuml_estimator


class _CumlModelWriter(MLWriter):
    """
    Write the parameters of _CumlModel to the file
    """

    def __init__(self, instance: "_CumlModel") -> None:
        super().__init__()
        self.instance: "_CumlModel" = instance

    def saveImpl(self, path: str) -> None:
        DefaultParamsWriter.saveMetadata(self.instance, path, self.sc)  # type: ignore
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
        return instance


class _CumlCommon(Params, MLWritable, MLReadable):
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

    def set_params(self, **kwargs: Any) -> None:
        """
        Set the kwargs to model's parameters
        """
        for k, v in kwargs.items():
            if self.hasParam(k):
                self._set(**{str(k): v})  # type: ignore
            else:
                raise ValueError(f"Unsupported param '{k}'.")

    @staticmethod
    def initialize_cuml_logging(verbose: Optional[Union[bool, int]]) -> None:
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


class _CumlEstimatorParams(HasInputCols, HasInputCol, HasOutputCol, HasOutputCols):
    """
    The common parameters for all Spark CUML algorithms.
    """

    num_workers = Param(
        Params._dummy(),  # type: ignore
        "num_workers",
        "The number of Spark CUML workers. Each CUML worker corresponds to one spark task.",
        TypeConverters.toInt,
    )

    def get_num_workers(self) -> int:
        return self.getOrDefault(self.num_workers)

    @classmethod
    def _cuml_cls(cls) -> List[type]:
        """
        Return the cuml python counterpart class name, which will be used to
        auto generate pyspark parameters.
        """
        raise NotImplementedError()

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return []

    @classmethod
    def _get_cuml_params_default(cls) -> Dict[str, Any]:
        """
        Inspect the __init__ function of _cuml_cls() to get the
        parameters and default values.
        """
        params = {}

        for cls_type in cls._cuml_cls():
            params.update(
                _get_default_params_from_func(cls_type, cls._not_supported_param())
            )
        return params

    def _gen_cuml_param(self) -> Dict[str, Any]:
        """
        Generate the CUML parameters according the pyspark estimator parameters.
        """
        params = {}
        for k, _ in self._get_cuml_params_default().items():
            params[k] = self.getOrDefault(k)

        return params


class _CumlEstimator(_CumlCommon, Estimator, _CumlEstimatorParams):
    """
    The common estimator to handle the fit callback (_fit). It should handle
    1. set the default parameters
    2. validate the parameters
    3. prepare the dataset
    4. train and return CUML model
    5. create the pyspark model
    """

    def __init__(self) -> None:
        super().__init__()
        self._set_pyspark_cuml_params()
        self._setDefault(num_workers=1)  # type: ignore

    def _set_pyspark_cuml_params(self) -> None:
        # Auto set the parameters into the estimator
        params = self._get_cuml_params_default()
        self._setDefault(**params)  # type: ignore

    @abstractmethod
    def _out_schema(self) -> Union[StructType, str]:
        """
        The output schema of the estimator, which will be used to
        construct the returning pandas dataframe
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_pyspark_model(self, result: Row) -> "_CumlModel":
        """
        Create the model according to the collected Row
        """
        raise NotImplementedError()

    def _repartition_dataset(self, dataset: DataFrame) -> DataFrame:
        """
        Repartition the dataset to the desired number of workers.
        """
        return dataset.repartition(self.getOrDefault(self.num_workers))

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
                    especially, the parameters of __init__ is stored in params[INIT_PARAMETERS_NAME]
                "" "
                ...
            ...

            return _cuml_fit

        _get_cuml_fit_func itself runs on the driver side, while the returned _cuml_fit will
        run on the executor side.
        """
        raise NotImplementedError()

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[
        List[Column], Optional[List[str]], int, Union[Type[FloatType], Type[DoubleType]]
    ]:
        select_cols = []
        multi_col_names = None

        # label column will be cast to feature type if needed.
        feature_type: Union[Type[FloatType], Type[DoubleType]] = FloatType
        if self.isDefined(self.inputCol):
            # Single Column
            input_name = self.getInputCol()
            input_datatype = dataset.schema[input_name].dataType

            if isinstance(input_datatype, ArrayType):
                # Array type
                select_cols.append(col(input_name).alias(alias.data))
                if isinstance(input_datatype.elementType, DoubleType):
                    feature_type = DoubleType
            elif isinstance(input_datatype, VectorUDT):
                # Vector type
                select_cols.append(
                    vector_to_array(col(input_name)).alias(alias.data)  # type: ignore
                )
                feature_type = DoubleType
            else:
                raise ValueError("Unsupported input type.")

            dimension = len(dataset.first()[self.getInputCol()])  # type: ignore

        elif self.isDefined(self.inputCols):
            # Multi columns
            multi_col_names = self.getInputCols()
            dimension = len(multi_col_names)
            for c in multi_col_names:
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
            raise ValueError("Please set inputCol or inputCols")

        return select_cols, multi_col_names, dimension, feature_type

    def _fit(self, dataset: DataFrame) -> "_CumlModel":
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

        dataset = dataset.select(*select_cols)

        if dataset.rdd.getNumPartitions() != self.get_num_workers():
            dataset = self._repartition_dataset(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        params: Dict[str, Any] = {
            INIT_PARAMETERS_NAME: self._gen_cuml_param(),
        }

        num_workers = self.get_num_workers()

        cuml_fit_func = self._get_cuml_fit_func(dataset)

        cuml_verbose = None
        if self.hasParam("verbose"):
            cuml_verbose = self.getOrDefault("verbose")

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import BarrierTaskContext

            logger = get_logger(cls)
            logger.info("Initializing cuml context")

            self.initialize_cuml_logging(cuml_verbose)

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()

            # set gpu device
            self.set_gpu_device(context, is_local)

            with CumlContext(partition_id, num_workers, context) as cc:
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
                        features = np.array(list(pdf[alias.data]))
                    label = pdf[alias.label] if alias.label in pdf.columns else None
                    inputs.append((features, label))

                params["handle"] = cc.handle
                params["part_sizes"] = sizes
                params["n"] = dimension

                logger.info("Invoking cuml fit")

                # call the cuml fit function
                result = cuml_fit_func(inputs, params)
                logger.info("Cuml fit complete")

            context.barrier()
            if context.partitionId() == 0:
                yield pd.DataFrame(data=result)

        ret = (
            dataset.mapInPandas(_train_udf, schema=self._out_schema())  # type: ignore
            .rdd.barrier()
            .mapPartitions(lambda x: x)
            .collect()[0]
        )

        return self._copyValues(self._create_pyspark_model(ret))  # type: ignore

    def write(self) -> MLWriter:
        return _CumlEstimatorWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlEstimatorReader(cls)


class _CumlEstimatorSupervised(_CumlEstimator, HasLabelCol):
    """
    Base class for Cuml Supervised machine learning
    """

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

        select_cols.append(label_col)

        return select_cols, multi_col_names, dimension, feature_type


class _CumlModel(
    _CumlCommon, Model, HasInputCol, HasInputCols, HasOutputCol, HasOutputCols
):
    """
    Abstract class for spark cuml models that are fitted by spark cuml estimators.
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

        # model_data is the native data which will be saved for model persistence
        self._model_attributes = model_attributes
        self._model_attributes["dtype"] = dtype
        self._model_attributes["n_cols"] = n_cols
        self.dtype = dtype
        self.n_cols = n_cols

    def get_model_attributes(self) -> Optional[Dict[str, Any]]:
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
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
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
        if self.isDefined(self.inputCol):
            # Single Column
            col_name = self.getInputCol()
            if isinstance(dataset.schema[col_name].dataType, VectorUDT):
                # Vector type
                # Avoid same naming. `echo sparkcuml | base64` = c3BhcmtjdW1sCg==
                tmp_name = f"{alias.data}_c3BhcmtjdW1sCg=="
                dataset = (
                    dataset.withColumnRenamed(col_name, tmp_name)
                    .withColumn(col_name, vector_to_array(col(tmp_name)))
                    .drop(tmp_name)
                )
            elif not isinstance(dataset.schema[col_name].dataType, ArrayType):
                # Array type
                raise ValueError("Unsupported input type.")
            select_cols.append(col_name)
            input_is_multi_cols = False
        elif self.isDefined(self.inputCols):
            select_cols.extend(self.getInputCols())
        else:
            raise ValueError("Please set inputCol or inputCols")

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
        construct_cuml_object_func, cuml_transform_func = self._get_cuml_transform_func(
            dataset
        )

        def _transform_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import TaskContext

            context = TaskContext.get()

            self.set_gpu_device(context, is_local, True)

            # Construct the cuml counterpart object
            cuml_object = construct_cuml_object_func()

            # Transform the dataset
            if input_is_multi_cols:
                for pdf in pdf_iter:
                    yield cuml_transform_func(cuml_object, pdf[select_cols])
            else:
                for pdf in pdf_iter:
                    nparray = np.array(list(pdf[select_cols[0]]))
                    yield cuml_transform_func(cuml_object, nparray)

        return dataset.mapInPandas(
            _transform_udf, schema=self._out_schema(dataset.schema)  # type: ignore
        )

    def write(self) -> MLWriter:
        return _CumlModelWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlModelReader(cls)


class _CumlModelSupervised(_CumlModel, HasPredictionCol):
    """Cuml base model for supervised machine learning"""

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """This version of transform is directly adding extra columns to the dataset"""
        dataset, select_cols, input_is_multi_cols = self._pre_process_data(dataset)

        pred_name = self.getOrDefault(self.predictionCol)

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        construct_cuml_object_func, cuml_transform_func = self._get_cuml_transform_func(
            dataset
        )

        @pandas_udf(self._out_schema(dataset.schema))  # type: ignore
        def predict_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            from pyspark import TaskContext

            context = TaskContext.get()
            self.set_gpu_device(context, is_local, True)
            cuml_object = construct_cuml_object_func()
            for pdf in iterator:
                if not input_is_multi_cols:
                    data = np.array(list(pdf[select_cols[0]]))
                else:
                    data = pdf[select_cols]

                yield cuml_transform_func(cuml_object, data)

        return dataset.withColumn(pred_name, predict_udf(struct(*select_cols)))

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        assert self.dtype is not None
        return dtype_to_pyspark_type(self.dtype)


def _set_pyspark_cuml_cls_param_attrs(
    pyspark_estimator_class: Type[_CumlEstimator], pyspark_model_class: Type[_CumlModel]
) -> None:
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
    cuml_estimator_class_name = []
    for cls_type in pyspark_estimator_class._cuml_cls():
        cuml_estimator_class_name.append(_get_class_name(cls_type))

    params_dict = pyspark_estimator_class._get_cuml_params_default()

    def param_value_converter(v: Any) -> Any:
        if isinstance(v, np.generic):
            # convert numpy scalar values to corresponding python scalar values
            return np.array(v).item()
        if isinstance(v, dict):
            return {k: param_value_converter(nv) for k, nv in v.items()}
        if isinstance(v, list):
            return [param_value_converter(nv) for nv in v]
        return v

    def set_param_attrs(attr_name: str, param_obj_: Param) -> None:
        param_obj_.typeConverter = param_value_converter  # type: ignore
        setattr(pyspark_estimator_class, attr_name, param_obj_)
        setattr(pyspark_model_class, attr_name, param_obj_)

    for name in params_dict.keys():
        doc = f"Refer to CUML doc of {', '.join(cuml_estimator_class_name)} for this param {name}"

        param_obj = Param(Params._dummy(), name=name, doc=doc)  # type: ignore
        set_param_attrs(name, param_obj)

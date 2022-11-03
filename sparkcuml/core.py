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

from abc import abstractmethod
from typing import Any, Callable, Iterator, Optional, Type, Union

import cudf
import numpy as np
import pandas as pd
from pyspark import SparkContext, TaskContext
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)
from pyspark.sql import DataFrame
from pyspark.sql.types import Row, StructType

from sparkcuml.common.nccl import NcclComm
from sparkcuml.utils import (
    _get_class_name,
    _get_default_params_from_func,
    _get_gpu_id,
    _get_spark_session,
    _is_local,
)

INIT_PARAMETERS_NAME = "init"


class _CumlSharedReadWrite:
    @staticmethod
    def save_meta_data(
        instance: "_CumlEstimator",
        path: str,
        sc: SparkContext,
        extra_metadata: Optional[dict] = None,
    ) -> None:
        instance.validate_params()
        skip_params: list[str] = []
        json_params: dict[str, Any] = {}
        for p, v in instance._paramMap.items():  # type: ignore
            if p.name not in skip_params:
                json_params[p.name] = v
        extra_metadata = extra_metadata or {}
        DefaultParamsWriter.saveMetadata(
            instance, path, sc, extraMetadata=extra_metadata, paramMap=json_params  # type: ignore
        )

    @staticmethod
    def load_instance(
        cuml_estimator_cls: Type, path: str, sc: SparkContext
    ) -> "_CumlEstimator":
        metadata = DefaultParamsReader.loadMetadata(path, sc)
        cuml_estimator = cuml_estimator_cls()
        DefaultParamsReader.getAndSetParams(cuml_estimator, metadata)
        return cuml_estimator


class _CumlEstimatorWriter(MLWriter):
    """
    Write the parameters of _CumlEstimator to the file
    """

    def __init__(self, instance: "_CumlEstimator") -> None:
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        _CumlSharedReadWrite.save_meta_data(self.instance, path, self.sc)


class _CumlEstimatorReader(MLReader):
    """
    Instantiate the _CumlEstimator from the file.
    """

    def __init__(self, cls: Type) -> None:
        super().__init__()
        self.cls = cls

    def load(self, path: str) -> "_CumlEstimator":
        return _CumlSharedReadWrite.load_instance(self.cls, path, self.sc)


class _CumlCommon(MLWritable, MLReadable):
    @staticmethod
    def set_gpu_device(context: Optional[TaskContext], is_local: bool) -> None:
        """
        Set gpu device according to the spark task resources.

        If it is local mode, we use partition id as gpu id.
        """
        # Get the GPU ID from resources
        assert context is not None
        gpu_id = context.partitionId() if is_local else _get_gpu_id(context)

        import cupy

        cupy.cuda.Device(gpu_id).use()


class _CumlEstimatorParams(HasInputCols, HasInputCol, HasOutputCol):
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
    def _cuml_cls(cls) -> type:
        """
        Return the cuml python counterpart class name, which will be used to
        auto generate pyspark parameters.
        """
        raise NotImplementedError()

    @classmethod
    def _not_supported_param(cls) -> list[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return []

    @classmethod
    def _get_cuml_params_default(cls) -> dict[str, Any]:
        """
        Inspect the __init__ function of _cuml_cls() to get the
        parameters and default values.
        """
        return _get_default_params_from_func(
            cls._cuml_cls(), cls._not_supported_param()
        )

    def _gen_cuml_param(self) -> dict[str, Any]:
        """
        Generate the CUML parameters according the pyspark estimator parameters.
        """
        params = {}
        for k, _ in self._get_cuml_params_default().items():
            if self.getOrDefault(k):
                params[k] = self.getOrDefault(k)

        return params

    def validate_params(self) -> None:
        pass


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

    def set_params(self, **kwargs: Any) -> None:
        """
        Set the kwargs to estimator's parameters
        """
        for k, v in kwargs.items():
            if self.hasParam(k):
                self._set(**{str(k): v})  # type: ignore
            else:
                raise ValueError(f"Unsupported param '{k}'.")

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
    ) -> Callable[[list[cudf.DataFrame], dict[str, Any]], dict[str, Any]]:
        """
        Subclass must implement this function to return a cuml fit function that will be
        sent to executor to run.

        Eg,

        def _get_cuml_fit_func(self, dataset: DataFrame):
            ...
            def _cuml_fit(df: list[cudf.DataFrame], params: dict[str, Any]) -> dict[str, Any]:
                "" "
                df:  a sequence of cudf DataFrame
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
        select_cols = []
        input_is_multi_cols = True
        if self.isDefined(self.inputCol):
            select_cols = [self.getInputCol()]
            dimension = len(dataset.first()[self.getInputCol()])  # type: ignore
            input_is_multi_cols = False
        elif self.isDefined(self.inputCols):
            select_cols.extend(self.getInputCols())
            dimension = len(self.getInputCols())
        else:
            raise ValueError("Please set inputCol or inputCols")

        dataset = dataset.select(*select_cols)
        if dataset.rdd.getNumPartitions() != self.get_num_workers():
            dataset = self._repartition_dataset(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        params: dict[str, Any] = {
            INIT_PARAMETERS_NAME: self._gen_cuml_param(),
            "dimension": dimension,
        }

        num_workers = self.get_num_workers()

        cuml_fit_func = self._get_cuml_fit_func(dataset)

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()

            # set gpu device
            self.set_gpu_device(context, is_local)

            # initialize nccl comm
            comm = NcclComm(num_workers, context)
            handle = comm.init_worker(partition_id, init_nccl=True)

            # handle the input
            inputs = []
            size = 0
            if input_is_multi_cols:
                for pdf in pdf_iter:
                    gdf = cudf.DataFrame(pdf[select_cols])
                    size += gdf.shape[0]
                    inputs.append(gdf)
            else:
                for pdf in pdf_iter:
                    flatten = pdf.apply(
                        lambda x: x[select_cols[0]],  # type: ignore
                        axis=1,
                        result_type="expand",
                    )
                    gdf = cudf.from_pandas(flatten)
                    size += gdf[0].size
                    inputs.append(gdf)

            # prepare (parts, rank)
            import json

            rank_size = (partition_id, size)
            messages = context.allGather(message=json.dumps(rank_size))
            parts_rank_size = [json.loads(pair) for pair in messages]
            num_cols = sum(pair[1] for pair in parts_rank_size)

            params["partsToRanks"] = parts_rank_size
            params["rank"] = partition_id
            params["handle"] = handle
            params["numVec"] = num_cols

            # call the cuml fit function
            result = cuml_fit_func(inputs, params)

            context.barrier()
            comm.destroy()

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


class _CumlModel(_CumlCommon, Model, HasInputCol, HasInputCols, HasOutputCol):
    """
    Abstract class for spark cuml models that are fitted by spark cuml estimators.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Callable[[cudf.DataFrame], pd.DataFrame]:
        """
        Subclass must implement this function to return a cuml transform function that will be
        sent to executor to run.

        Eg,

        def _get_cuml_transform_func(self, dataset: DataFrame):
            ...
            def _cuml_transform(df: cudf.DataFrame) ->pd.DataFrame:
                ...
            ...

            return _cuml_transform

        _get_cuml_transform_func itself runs on the driver side, while the returned _cuml_transform will
        run on the executor side.
        """
        raise NotImplementedError()

    @abstractmethod
    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        """
        The output schema of the model, which will be used to
        construct the returning pandas dataframe
        """
        raise NotImplementedError()

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

        select_cols = []
        input_is_multi_cols = True
        if self.isDefined(self.inputCol):
            select_cols.append(self.getInputCol())
            input_is_multi_cols = False
        elif self.isDefined(self.inputCols):
            select_cols.extend(self.getInputCols())
        else:
            raise ValueError("Please set inputCol or inputCols")

        is_local = _is_local(_get_spark_session().sparkContext)

        cuml_transform_func = self._get_cuml_transform_func(dataset)

        def _transform_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import TaskContext

            context = TaskContext.get()

            self.set_gpu_device(context, is_local)

            if input_is_multi_cols:
                for pdf in pdf_iter:
                    gdf = cudf.DataFrame(pdf[select_cols])
                    yield cuml_transform_func(gdf)
            else:
                for pdf in pdf_iter:
                    flatten = pdf.apply(
                        lambda x: x[select_cols[0]],  # type: ignore
                        axis=1,
                        result_type="expand",
                    )
                    gdf = cudf.from_pandas(flatten)
                    yield cuml_transform_func(gdf)

        return dataset.mapInPandas(
            _transform_udf, schema=self._out_schema(dataset.schema)  # type: ignore
        )


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
    cuml_estimator_class_name = _get_class_name(pyspark_estimator_class._cuml_cls())
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
        doc = f"Refer to CUML doc of {cuml_estimator_class_name} for this param {name}"

        param_obj = Param(Params._dummy(), name=name, doc=doc)  # type: ignore
        set_param_attrs(name, param_obj)

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
from typing import Any, Union, Iterator

import cudf
import pandas as pd
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasInputCol
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, Row

from sparkcuml.utils import _is_local, _get_spark_session, _get_gpu_id, _get_default_params_from_func
from sparkcuml.common.nccl import NcclComm

INIT_PARAMETERS_NAME = "init"


class _CumlEstimatorParams(HasInputCols, HasInputCol):
    """
    The common parameters for all Spark CUML algorithms.
    """
    num_workers = Param(
        Params._dummy(),
        "num_workers",
        "The number of Spark CUML workers. Each CUML worker corresponds to one spark task.",
        TypeConverters.toInt,
    )

    def get_num_workers(self):
        return self.getOrDefault(self.num_workers)

    @classmethod
    def _cuml_cls(cls):
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
            cls._cuml_cls(),
            cls._not_supported_param(),
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


class _CumlEstimator(Estimator, _CumlEstimatorParams):
    """
    The common estimator to handle the fit callback (_fit). It should handle
    1. set the default parameters
    2. validate the parameters
    3. prepare the dataset
    4. train and return CUML model
    5. create the pyspark model
    """

    def __init__(self):
        super().__init__()
        self._set_pyspark_cuml_params()
        self._setDefault(
            num_workers=1,
        )

    def _set_pyspark_cuml_params(self):
        # Auto set the parameters into the estimator
        params = self._get_cuml_params_default()
        self._setDefault(**params)

    def set_params(self, **kwargs):
        """
        Set the kwargs to estimator's parameters
        """
        for k, v in kwargs.items():
            if self.hasParam(k):
                self._set(**{str(k): v})
            else:
                raise ValueError(f"Unsupported param '{k}'.")

    @abstractmethod
    def _fit_internal(self, df: list[cudf.DataFrame], **kwargs) -> dict[str, Any]:
        """
        Subclass must implement its own logic to fit a model to the input dataset.
        Please note that, this function is called on the executor side.
        """
        raise NotImplementedError()

    @abstractmethod
    def _out_schema(self) -> Union[StructType, str]:
        """
        The output schema of the estimator, which will be used to
        construct the returning pandas dataframe
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_pyspark_model(self, result: Row):
        """
        Create the model according to the collected Row
        """
        raise NotImplementedError()

    def _repartition_dataset(self, dataset: DataFrame) -> DataFrame:
        """
        Repartition the dataset to the desired number of workers.
        """
        return dataset.repartition(self.getOrDefault(self.num_workers))

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
            dimension = len(dataset.first())
            input_is_multi_cols = False
        elif self.isDefined(self.inputCols):
            select_cols.extend(self.getInputCols())
            dimension = len(self.getInputCols())
        else:
            raise ValueError("Please set inputCol or inputCols")

        dataset = dataset.select(*select_cols)
        dataset = self._repartition_dataset(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        params = {}
        params[INIT_PARAMETERS_NAME] = self._gen_cuml_param()
        params['dimension'] = dimension

        comm = NcclComm(self.get_num_workers())

        def _cuml_fit(pdf_iter: Iterator[pd.DataFrame]):
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()

            # Get the GPU ID from resources
            gpu_id = context.partitionId() if is_local else _get_gpu_id(context)

            import cupy
            cupy.cuda.Device(gpu_id).use()
            params['rank'] = context.partitionId()

            context.barrier()

            handle = comm.init_worker(params['rank'], init_nccl=True)
            params['handle'] = handle

            inputs = []
            size = 0

            if input_is_multi_cols:
                for pdf in pdf_iter:
                    gdf = cudf.DataFrame(pdf[select_cols])
                    size += gdf.shape[0]
                    inputs.append(gdf)
            else:
                for pdf in pdf_iter:
                    flatten = pdf.apply(lambda x: x[input_is_multi_cols[0]], axis=1, result_type='expand')
                    gdf = cudf.from_pandas(flatten)
                    size += gdf[0].size
                    inputs.append(gdf)

            import json
            rank2size = (params['rank'], size)
            messages = context.allGather(message=json.dumps(rank2size))
            parts_to_ranks = [json.loads(pair) for pair in messages]
            parts_to_ranks = sorted(parts_to_ranks, key=lambda p: p[0])
            params['partsToRanks'] = parts_to_ranks

            num_vec = sum(pair[1] for pair in parts_to_ranks)
            params['numVec'] = num_vec

            result = self._fit_internal(inputs, **params)

            context.barrier()
            if context.partitionId() == 0:
                yield pd.DataFrame(data=result)

        ret = (
            dataset.mapInPandas(
                _cuml_fit,
                schema=self._out_schema(),
            )
            .rdd.barrier()
            .mapPartitions(lambda x: x)
            .collect()[0]
        )

        return self._copyValues(self._create_pyspark_model(ret))


class _CumlModel(Model):
    """
    Abstract class for spark cuml models that are fitted by spark cuml estimators.
    """

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, dataset) -> DataFrame:
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
        pass

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
from pyspark.ml import Estimator
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, Row


class _CumlEstimatorParams(HasInputCols):
    """
    The common parameters for all Spark CUML algorithms.
    """
    num_workers = Param(
        Params._dummy(),
        "num_workers",
        "The number of Spark CUML workers. Each CUML worker corresponds to one spark task.",
        TypeConverters.toInt,
    )


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
        self._setDefault(
            num_workers=1,
        )

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

    def _fit(self, dataset: DataFrame):
        input_col_names = self.getInputCols()
        dataset = dataset.select(*input_col_names)
        dataset = self._repartition_dataset(dataset)
        params = {}

        def _cuml_fit(pdf_iter: Iterator[pd.DataFrame]):
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            context.barrier()

            inputs = []

            if input_col_names:
                for pdf in pdf_iter:
                    inputs.append(cudf.DataFrame(pdf[input_col_names]))
            else:
                # TODO do we need to support features (vector or array type) ??
                for pdf in pdf_iter:
                    flatten = pdf.apply(lambda x: x['features'], axis=1, result_type='expand')
                    gdf = cudf.from_pandas(flatten)
                    inputs.append(gdf)

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

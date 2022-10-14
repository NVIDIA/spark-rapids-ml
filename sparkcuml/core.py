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
from pyspark.ml import Estimator
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql import DataFrame


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

    def _fit(self, dataset: DataFrame):
        pass

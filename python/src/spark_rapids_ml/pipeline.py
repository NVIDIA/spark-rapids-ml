#
# Copyright (c) 2025, NVIDIA CORPORATION.
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

from typing import TYPE_CHECKING, Any, List, Optional

from pyspark import keyword_only
from pyspark.ml.base import Estimator
from pyspark.ml.base import Transformer as CPUTransformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline as SparkPipeline
from pyspark.ml.pipeline import PipelineModel as SparkPipelineModel
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.sql import DataFrame

from spark_rapids_ml.core import _CumlEstimator
from spark_rapids_ml.utils import get_logger

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap, PipelineStage


class NoOpTransformer(VectorAssembler):
    @keyword_only
    def __init__(
        self,
        inputCols: Optional[List[str]] = None,
        outputCol: Optional[str] = None,
        handleInvalid: str = "error",
    ):
        kwargs = self._input_kwargs
        super().__init__(**kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class Pipeline(SparkPipeline):
    @keyword_only
    def __init__(self, *, stages: Optional[List["PipelineStage"]] = None):
        kwargs = self._input_kwargs
        super(Pipeline, self).__init__(**kwargs)

    def _fit(self, dataset: DataFrame) -> "SparkPipelineModel":
        stages = self.getStages()

        for i in range(len(stages)):

            if not self._isGPUEstimator(stages[i]):
                continue

            if (
                i > 0
                and isinstance(stages[i - 1], VectorAssembler)
                and self._allScalar(dataset, stages[i - 1].getInputCols())  # type: ignore
            ):

                self._setInputCols(stages[i], stages[i - 1].getInputCols())  # type: ignore
                stages[i - 1] = NoOpTransformer()
                logger = get_logger(stages[i].__class__)
                logger.info(
                    "Spark Rapids ML pipeline bypasses VectorAssembler for GPU-based estimators to achieve optimal performance."
                )

        return super(Pipeline, self)._fit(dataset)

    @staticmethod
    def _setInputCols(gpu_est: _CumlEstimator, input_cols: List[str]) -> None:
        assert Pipeline._isGPUEstimator(gpu_est)
        if isinstance(gpu_est, SparkCrossValidator):
            gpu_est = gpu_est.getOrDefault(gpu_est.estimator)

        from spark_rapids_ml.utils import setInputOrFeaturesCol

        setInputOrFeaturesCol(gpu_est, input_cols)

    @staticmethod
    def _allScalar(dataset: DataFrame, inputCols: List[str]) -> bool:
        from pyspark.sql.types import NumericType

        for col_name in inputCols:
            if not isinstance(dataset.schema[col_name].dataType, NumericType):
                return False
        return True

    @staticmethod
    def _isGPUEstimator(est: Any) -> bool:

        if isinstance(est, _CumlEstimator):
            return True

        if isinstance(est, SparkCrossValidator):
            actual_est = est.getOrDefault(est.estimator)
            return isinstance(actual_est, _CumlEstimator)

        return False

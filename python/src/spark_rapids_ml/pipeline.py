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

from typing import TYPE_CHECKING, Any, List, Optional, Union

from pyspark import keyword_only
from pyspark.ml.base import Estimator
from pyspark.ml.base import Transformer
from pyspark.ml.base import Transformer as CPUTransformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline as SparkPipeline
from pyspark.ml.pipeline import PipelineModel as SparkPipelineModel
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.ml.tuning import CrossValidatorModel as SparkCrossValidatorModel
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
    """
    A customized Spark ML Pipeline optimized for GPU-based estimators.

    This subclass of `SparkPipeline` bypasses the `VectorAssembler` stage when the following conditions are met:
    - The pipeline contains exactly two stages.
    - The first stage is a `VectorAssembler`.
    - The second stage is a GPU-based estimator (checked via `_isGPUEstimator`).
    - The input columns to the assembler exist in the dataset and are all scalar.

    When these conditions are met, the pipeline skips the `VectorAssembler` transformation during fitting,
    and instead injects the scalar columns directly into the GPU estimator to improve performance
    on the Spark Rapids ML runtime. After fitting, compatibility is restored by reassigning the original
    `VectorAssembler` to the pipeline and pipeline model.

    Parameters
    ----------
    stages : Optional[List[PipelineStage]], default=None
        A list of pipeline stages. If provided, should contain a `VectorAssembler` followed by a GPU estimator
        to trigger the GPU-specific optimization.

    Notes
    -----
    - If the optimization conditions are not met, the pipeline behaves exactly like a standard `SparkPipeline`.
    - This class is primarily intended for use with Spark Rapids ML, where GPU-based estimators benefit
      from bypassing unnecessary CPU preprocessing.
    """

    @keyword_only
    def __init__(self, *, stages: Optional[List["PipelineStage"]] = None):
        kwargs = self._input_kwargs
        super(Pipeline, self).__init__(**kwargs)

    def _fit(self, dataset: DataFrame) -> "SparkPipelineModel":
        stages = self.getStages()

        if (
            len(stages) != 2
            or not isinstance(stages[0], VectorAssembler)
            or stages[0].getHandleInvalid() != "error"
            or not self._isGPUEstimator(stages[1])
            or not self._colsValid(dataset, stages[0].getInputCols())
        ):
            return super(Pipeline, self)._fit(dataset)

        # revise pipeline
        va_stage = stages[0]
        est_stage = stages[1]

        stages[0] = NoOpTransformer()
        self._setEitherColsOrCol(est_stage, va_stage.getInputCols())  # type: ignore

        logger = get_logger(est_stage.__class__)
        logger.info(
            "Spark Rapids ML pipeline bypasses VectorAssembler for GPU-based estimators to achieve optimal performance."
        )

        # get fit model
        p_model = super(Pipeline, self)._fit(dataset)

        # ensure compatibility
        stages[0] = va_stage
        self._setEitherColsOrCol(est_stage, va_stage.getOutputCol())

        p_model.stages[0] = va_stage
        self._setEitherColsOrCol(p_model.stages[1], va_stage.getOutputCol())

        return p_model

    @staticmethod
    def _setEitherColsOrCol(
        pstage: Union[Estimator, Transformer], input_col: Union[List[str], str]
    ) -> None:
        if isinstance(pstage, SparkCrossValidator) or isinstance(
            pstage, SparkCrossValidatorModel
        ):
            pstage = pstage.getOrDefault(pstage.estimator)

        from spark_rapids_ml.utils import setInputOrFeaturesCol

        setInputOrFeaturesCol(pstage, input_col)

    @staticmethod
    def _colsValid(dataset: DataFrame, inputCols: List[str]) -> bool:
        from pyspark.sql.types import NumericType

        for col_name in inputCols:
            if col_name not in dataset.columns:
                return False
            if not isinstance(dataset.schema[col_name].dataType, NumericType):
                return False
        return True

    @staticmethod
    def _isGPUEstimator(est: Any) -> bool:

        if isinstance(est, SparkCrossValidator):
            est = est.getOrDefault(est.estimator)

        # CPU estimator
        if not isinstance(est, _CumlEstimator):
            return False

        # GPU estimator but will fallback
        if est._pyspark_class() and est._fallback_enabled and est._use_cpu_fallback():
            return False

        return True

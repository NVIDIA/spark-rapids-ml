#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from pyspark import inheritable_thread_target
from pyspark.ml import Model
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.util import DefaultParamsReader
from pyspark.sql import DataFrame

from .core import _CumlEstimator, _CumlModel


def _gen_avg_and_std_metrics_(
    metrics_all: List[List[float]],
) -> Tuple[List[float], List[float]]:
    avg_metrics = np.mean(metrics_all, axis=0)
    std_metrics = np.std(metrics_all, axis=0)
    return list(avg_metrics), list(std_metrics)


class CrossValidator(SparkCrossValidator):
    """K-fold cross validation performs model selection by splitting the dataset into a set of
    non-overlapping randomly partitioned folds which are used as separate training and test datasets
    e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs,
    each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the
    test set exactly once.

    It is the gpu version CrossValidator which fits multiple models in a single pass for a single
    training dataset and transforms/evaluates in a single pass for multiple models.

    Examples
    --------

    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.tuning import ParamGridBuilder, CrossValidatorModel
    >>> from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    >>> from spark_rapids_ml.tuning import CrossValidator
    >>> from spark_rapids_ml.classification import RandomForestClassifier
    >>> import tempfile
    >>> dataset = spark.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 2.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"])
    >>> rfc = RandomForestClassifier()
    >>> grid = ParamGridBuilder().addGrid(rfc.maxBins, [8, 16]).build()
    >>> evaluator = MulticlassClassificationEvaluator()
    >>> cv = CrossValidator(estimator=rfc, estimatorParamMaps=grid, evaluator=evaluator,
    ...                     parallelism=2)
    >>> cvModel = cv.fit(dataset)
    ...
    >>> cvModel.getNumFolds()
    3
    >>> cvModel.avgMetrics[0]
    1.0
    >>> evaluator.evaluate(cvModel.transform(dataset))
    1.0
    >>> path = tempfile.mkdtemp()
    >>> model_path = path + "/model"
    >>> cvModel.write().save(model_path)
    >>> cvModelRead = CrossValidatorModel.read().load(model_path)
    >>> cvModelRead.avgMetrics
    [1.0, 1.0]
    >>> evaluator.evaluate(cvModel.transform(dataset))
    1.0
    >>> evaluator.evaluate(cvModelRead.transform(dataset))
    1.0

    """

    def _fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        est = self.getOrDefault(self.estimator)
        eva = self.getOrDefault(self.evaluator)

        # fallback at very early time.
        if not (
            isinstance(est, _CumlEstimator) and est._supportsTransformEvaluate(eva)
        ):
            return super()._fit(dataset)

        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        nFolds = self.getOrDefault(self.numFolds)
        metrics_all = [[0.0] * numModels for i in range(nFolds)]

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        datasets = self._kFold(dataset)

        def singePassTask(
            fold: int,
        ) -> Tuple[int, List[float], Optional[List[_CumlModel]]]:
            index_models = list(est.fitMultiple(datasets[fold][0], epm))
            models = [model for _, model in index_models]
            model = models[0]._combine(models)
            metrics = model._transformEvaluate(datasets[fold][1], eva)
            return fold, metrics, models if collectSubModelsParam else None

        for fold, metrics, subModels in pool.imap_unordered(
            inheritable_thread_target(singePassTask), range(nFolds)
        ):
            metrics_all[fold] = metrics
            if collectSubModelsParam:
                assert subModels is not None
                subModels[fold] = subModels

        metrics, std_metrics = _gen_avg_and_std_metrics_(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])

        try:
            model = CrossValidatorModel(
                bestModel, metrics, cast(List[List[Model]], subModels), std_metrics
            )
        except:
            model = CrossValidatorModel(
                bestModel, metrics, cast(List[List[Model]], subModels)
            )

        return self._copyValues(model)

    @staticmethod
    def _is_python_params_instance(metadata: Dict[str, Any]) -> bool:
        # If it's not python module, pyspark will load spark_rapids_ml.tuning.CrossValidator
        # from JVM package. So we need to hack here
        return metadata["class"].startswith(("pyspark.ml.", "spark_rapids_ml."))

    @classmethod
    def load(cls, path: str) -> "CrossValidator":
        orig_is_python_params_instance = DefaultParamsReader.isPythonParamsInstance
        try:
            # Replace isPythonParamsInstance
            setattr(
                DefaultParamsReader,
                "isPythonParamsInstance",
                CrossValidator._is_python_params_instance,
            )
            cv_pyspark = super().load(path)
            cv = cls()
            cv_pyspark._copyValues(cv)
        finally:
            # Must restore to the original isPythonParamsInstance
            setattr(
                DefaultParamsReader,
                "isPythonParamsInstance",
                orig_is_python_params_instance,
            )

        return cv

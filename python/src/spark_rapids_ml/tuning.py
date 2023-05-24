#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, cast

import numpy as np
from pyspark import inheritable_thread_target
from pyspark.ml import Estimator, Model, Transformer
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame

from spark_rapids_ml.classification import (
    RandomForestClassificationModel,
    RandomForestClassifier,
)

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


# Copied from pyspark.ml.tuning
def _parallelFitTasks(
    est: Estimator,
    train: DataFrame,
    eva: Evaluator,
    validation: DataFrame,
    epm: Sequence["ParamMap"],
    collectSubModel: bool,
) -> List[Callable[[], Tuple[int, float, Transformer]]]:
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.

    Parameters
    ----------
    est : :py:class:`pyspark.ml.baseEstimator`
        he estimator to be fit.
    train : :py:class:`pyspark.sql.DataFrame`
        DataFrame, training data set, used for fitting.
    eva : :py:class:`pyspark.ml.evaluation.Evaluator`
        used to compute `metric`
    validation : :py:class:`pyspark.sql.DataFrame`
        DataFrame, validation data set, used for evaluation.
    epm : :py:class:`collections.abc.Sequence`
        Sequence of ParamMap, params maps to be used during fitting & evaluation.
    collectSubModel : bool
        Whether to collect sub model.

    Returns
    -------
    tuple
        (int, float, subModel), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    def singleTask() -> Tuple[int, float, Transformer]:
        index, model = next(modelIter)
        if isinstance(model, RandomForestClassificationModel) and model.numClasses > 2:
            metric = model._transformEvaluate(validation, epm[index])
        else:
            # TODO: duplicate evaluator to take extra params from input
            #  Note: Supporting tuning params in evaluator need update method
            #  `MetaAlgorithmReadWrite.getAllNestedStages`, make it return
            #  all nested stages and evaluators
            metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric, model if collectSubModel else None

    return [singleTask] * len(epm)


# Copied from pyspark.ml.tuning
def _gen_avg_and_std_metrics(
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
    training dataset and transforms/evaluates in a single pass for a single model.
    """

    def _fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        est = self.getOrDefault(self.estimator)
        eva = self.getOrDefault(self.evaluator)

        if not (
            isinstance(est, RandomForestClassifier)
            and isinstance(eva, MulticlassClassificationEvaluator)
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
        for i in range(nFolds):
            # TODO, Need to get rid of cache when supporting transform/evaluate
            # in a single pass for multiple models.
            validation = datasets[i][1].cache()
            train = datasets[i][0]

            tasks = map(
                inheritable_thread_target,
                _parallelFitTasks(
                    est, train, eva, validation, epm, collectSubModelsParam
                ),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics_all[i][j] = metric
                if collectSubModelsParam:
                    assert subModels is not None
                    subModels[i][j] = subModel

            validation.unpersist()

        metrics, std_metrics = _gen_avg_and_std_metrics(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])

        try:
            # since spark3.3.0+
            model = CrossValidatorModel(
                bestModel, metrics, cast(List[List[Model]], subModels), std_metrics
            )
        except:
            model = CrossValidatorModel(
                bestModel, metrics, cast(List[List[Model]], subModels)
            )

        return self._copyValues(model)

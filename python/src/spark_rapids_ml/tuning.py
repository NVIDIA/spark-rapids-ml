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

from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from pyspark import inheritable_thread_target
from pyspark.ml import Model
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.util import DefaultParamsReader
from pyspark.sql import DataFrame
from pyspark.sql.connect.plan import LogicalPlan

from .core import _CumlEstimator, _CumlModel
from .utils import get_logger

from pyspark.sql.connect import proto as spark_pb
from .proto import relations_pb2 as rapids_pb
import json

def _gen_avg_and_std_metrics_(
    metrics_all: List[List[float]],
) -> Tuple[List[float], List[float]]:
    avg_metrics = np.mean(metrics_all, axis=0)
    std_metrics = np.std(metrics_all, axis=0)
    return list(avg_metrics), list(std_metrics)

class _CrossValidatorPlan(LogicalPlan):

    def __init__(self, cv_relation: rapids_pb.CrossValidatorRelation):
        super().__init__(None)
        self._cv_relation = cv_relation

    def plan(self, session: "SparkConnectClient") -> spark_pb.Relation:
        plan = self._create_proto_relation()
        plan.extension.Pack(self._cv_relation)
        return plan


def _extractParams(instance: "Params") -> str:
    params = {}
    # TODO: support vector/matrix
    for k, v in instance._paramMap.items():
        if instance.isSet(k) and isinstance(v, int | float | str | bool):
            params[k.name] = v

    import json
    return json.dumps(params)


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

    def __remote_fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        estimator = self.getEstimator()
        evaluator = self.getEvaluator()
        est_param_list = []
        for param_group in self.getEstimatorParamMaps():
            est_param_items = []
            for p, v in param_group.items():
                tmp_map = {"parent": p.parent, "name": p.name, "value": v}
                est_param_items.append(tmp_map)
            est_param_list.append(est_param_items)

        est_param_map_json = json.dumps(est_param_list)

        estimator_name = type(estimator).__name__
        cv_rel = rapids_pb.CrossValidatorRelation(
            uid=self.uid,
            estimator=rapids_pb.MlOperator(
                name=estimator_name,
                uid=estimator.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_ESTIMATOR,
                params=_extractParams(estimator),
            ),
            estimator_param_maps=est_param_map_json,
            evaluator=rapids_pb.MlOperator(
                name=type(evaluator).__name__,
                uid=evaluator.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_EVALUATOR,
                params=_extractParams(evaluator),
            ),
            dataset=dataset._plan.to_proto(dataset.sparkSession.client).SerializeToString(),
            params=_extractParams(self),
        )
        from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
        df = ConnectDataFrame(_CrossValidatorPlan(cv_relation=cv_rel), dataset.sparkSession)
        row = df.collect()

        best_model = None
        model_id = row[0].best_model_id
        # TODO support other estimators
        if estimator_name == "LogisticRegression":
            from pyspark.ml.classification import LogisticRegressionModel
            best_model = LogisticRegressionModel(model_id)

        return CrossValidatorModel(best_model)

    def _fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        from pyspark.sql import is_remote
        if is_remote():
            return self.__remote_fit(dataset)

        est = self.getOrDefault(self.estimator)
        eva = self.getOrDefault(self.evaluator)

        # fallback at very early time.
        if not (
            isinstance(est, _CumlEstimator) and est._supportsTransformEvaluate(eva)
        ):
            return super()._fit(dataset)

        epm = self.getOrDefault(self.estimatorParamMaps)
        # fallback if any params are not gpu supported
        for param_map in epm:
            est_tmp = est.copy(param_map)
            if est_tmp._fallback_enabled and est_tmp._use_cpu_fallback():
                logger = get_logger(self.__class__)
                logger.warning("Falling back to CPU CrossValidator fit().")
                return super()._fit(dataset)

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

        for fold, metrics, _subModels in pool.imap_unordered(
            inheritable_thread_target(singePassTask), range(nFolds)
        ):
            metrics_all[fold] = metrics
            if collectSubModelsParam:
                assert subModels is not None
                subModels[fold] = _subModels

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

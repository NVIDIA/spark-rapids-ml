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
from abc import ABCMeta
from collections import Counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import pyspark
from pyspark.ml.common import _py2java
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator

from .metrics import EvalMetricInfo, transform_evaluate_metric
from .metrics.MulticlassMetrics import MulticlassMetrics

if TYPE_CHECKING:
    import cupy as cp
    from pyspark.ml._typing import ParamMap

import numpy as np
import pandas as pd
import scipy
from pyspark import Row, TaskContext, keyword_only
from pyspark.ml.classification import (
    BinaryRandomForestClassificationSummary,
)
from pyspark.ml.classification import (
    LogisticRegressionModel as SparkLogisticRegressionModel,
)
from pyspark.ml.classification import (
    LogisticRegressionSummary,
    LogisticRegressionTrainingSummary,
)
from pyspark.ml.classification import (
    RandomForestClassificationModel as SparkRandomForestClassificationModel,
)
from pyspark.ml.classification import (
    RandomForestClassificationSummary,
    _LogisticRegressionParams,
    _RandomForestClassifierParams,
)
from pyspark.ml.linalg import DenseMatrix, Matrix, Vector, Vectors
from pyspark.ml.param.shared import HasLabelCol, HasProbabilityCol, HasRawPredictionCol
from pyspark.sql import Column, DataFrame, Row
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    IntegralType,
    StringType,
    StructField,
    StructType,
)

from .core import (
    CumlT,
    FitInputType,
    TransformInputType,
    _ConstructFunc,
    _CumlEstimatorSupervised,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
    pred,
)
from .params import HasEnableSparseDataOptim, HasFeaturesCols, _CumlClass, _CumlParams
from .tree import (
    _RandomForestClass,
    _RandomForestCumlParams,
    _RandomForestEstimator,
    _RandomForestModel,
)
from .utils import (
    PartitionDescriptor,
    _ArrayOrder,
    _concat_and_free,
    _get_spark_session,
    get_logger,
    java_uid,
)

T = TypeVar("T")


class _ClassificationModelEvaluationMixIn:
    # https://github.com/python/mypy/issues/5868#issuecomment-437690894 to bypass mypy checking
    _this_model: Union["RandomForestClassificationModel", "LogisticRegressionModel"]

    def _get_evaluate_fn(self, eval_metric_info: EvalMetricInfo) -> _EvaluateFunc:
        if eval_metric_info.eval_metric == transform_evaluate_metric.accuracy_like:

            def _evaluate(
                input: TransformInputType,
                transformed: "cp.ndarray",
            ) -> pd.DataFrame:
                # calculate the count of (label, prediction)
                import cudf

                comb = cudf.DataFrame(
                    {
                        "label": input[alias.label],
                        "prediction": transformed,
                    }
                )
                confusion = (
                    comb.groupby(["label", "prediction"])
                    .size()
                    .reset_index(name="total")
                )

                confusion = confusion.to_pandas()

                return confusion

        else:

            def _evaluate(
                input: TransformInputType,
                transformed: "cp.ndarray",
            ) -> pd.DataFrame:
                from cuml.metrics import log_loss

                _log_loss = log_loss(input[alias.label], transformed, normalize=False)

                _log_loss_pdf = pd.DataFrame(
                    {"total": [len(input[alias.label])], "log_loss": [_log_loss]}
                )

                return _log_loss_pdf

        return _evaluate

    def _transformEvaluate(
        self,
        dataset: DataFrame,
        evaluator: Evaluator,
        params: Optional["ParamMap"] = None,
    ) -> List[float]:
        """
        Transforms and evaluates the input dataset with optional parameters in a single pass.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            a dataset that contains labels/observations and predictions
        evaluator: :py:class:`pyspark.ml.evaluation.Evaluator`
            an evaluator user intends to use
        params : dict, optional
            an optional param map that overrides embedded params

        Returns
        -------
        list of float
            metrics
        """

        if not isinstance(evaluator, MulticlassClassificationEvaluator):
            raise NotImplementedError(f"{evaluator} is unsupported yet.")

        if (
            evaluator.getMetricName()
            not in MulticlassMetrics.SUPPORTED_MULTI_CLASS_METRIC_NAMES
        ):
            raise NotImplementedError(
                f"{evaluator.getMetricName()} is not supported yet."
            )

        if self._this_model.getLabelCol() not in dataset.schema.names:
            raise RuntimeError("Label column is not existing.")

        dataset = dataset.withColumnRenamed(self._this_model.getLabelCol(), alias.label)

        if evaluator.getMetricName() == "logLoss":
            schema = StructType(
                [
                    StructField(pred.model_index, IntegerType()),
                    StructField("total", FloatType()),
                    StructField("log_loss", FloatType()),
                ]
            )

            eval_metric_info = EvalMetricInfo(
                eval_metric=transform_evaluate_metric.log_loss, eps=evaluator.getEps()
            )
        else:
            schema = StructType(
                [
                    StructField(pred.model_index, IntegerType()),
                    StructField("label", FloatType()),
                    StructField("prediction", FloatType()),
                    StructField("total", FloatType()),
                ]
            )
            eval_metric_info = EvalMetricInfo(
                eval_metric=transform_evaluate_metric.accuracy_like
            )
        # TBD: use toPandas and pandas df operations below
        rows = self._this_model._transform_evaluate_internal(
            dataset, schema, eval_metric_info
        ).collect()

        num_models = self._this_model._get_num_models()

        if eval_metric_info.eval_metric == transform_evaluate_metric.accuracy_like:
            # if we ever implement weights, Counter supports float values, but
            # type checking might fail https://github.com/python/typeshed/issues/3438
            tp_by_class: List[Counter[float]] = [Counter() for _ in range(num_models)]
            fp_by_class: List[Counter[float]] = [Counter() for _ in range(num_models)]
            label_count_by_class: List[Counter[float]] = [
                Counter() for _ in range(num_models)
            ]
            label_count = [0 for _ in range(num_models)]

            for row in rows:
                label_count[row.model_index] += row.total
                label_count_by_class[row.model_index][row.label] += row.total

                if row.label == row.prediction:
                    tp_by_class[row.model_index][row.label] += row.total
                else:
                    fp_by_class[row.model_index][row.prediction] += row.total

            scores = []
            for i in range(num_models):
                # match spark mllib behavior in the below cases
                for l in label_count_by_class[i]:
                    if l not in tp_by_class[i]:
                        tp_by_class[i][l] = 0
                    if l not in fp_by_class[i]:
                        fp_by_class[i][l] = 0
                metrics = MulticlassMetrics(
                    tp=dict(tp_by_class[i]),
                    fp=dict(fp_by_class[i]),
                    label=dict(label_count_by_class[i]),
                    label_count=label_count[i],
                )
                scores.append(metrics.evaluate(evaluator))
        else:
            # logLoss metric
            label_count = [0 for _ in range(num_models)]
            log_loss = [0.0 for _ in range(num_models)]
            for row in rows:
                label_count[row.model_index] += row.total
                log_loss[row.model_index] += row.log_loss

            scores = []
            for i in range(num_models):
                metrics = MulticlassMetrics(
                    label_count=label_count[i],
                    log_loss=log_loss[i],
                )
                scores.append(metrics.evaluate(evaluator))

        return scores


class _RFClassifierParams(
    _RandomForestClassifierParams, HasProbabilityCol, HasRawPredictionCol
):
    def __init__(self, *args: Any):
        super().__init__(*args)

    def setProbabilityCol(
        self: "_RFClassifierParams", value: str
    ) -> "_RFClassifierParams":
        """
        Sets the value of :py:attr:`probabilityCol`.
        """
        return self._set(probabilityCol=value)

    def setRawPredictionCol(
        self: "_RFClassifierParams", value: str
    ) -> "_RFClassifierParams":
        """
        Sets the value of :py:attr:`rawPredictionCol`.
        """
        return self._set(rawPredictionCol=value)


class _RandomForestClassifierClass(_RandomForestClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        mapping = super()._param_mapping()
        return mapping

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return pyspark.ml.classification.RandomForestClassifier


class RandomForestClassifier(
    _RandomForestClassifierClass,
    _RandomForestEstimator,
    _RandomForestCumlParams,
    _RFClassifierParams,
):
    """RandomForestClassifier implements a Random Forest classifier model which
    fits multiple decision tree classifiers in an ensemble. It supports both
    binary and multiclass labels. It implements cuML's GPU accelerated
    RandomForestClassifier algorithm based on cuML python library,
    and it can be used in PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`,
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`,
    :py:class:`~pyspark.ml.classification.OneVsRest`.

    The distributed algorithm uses an *embarrassingly-parallel* approach. For a
    forest with `N` trees being built on `w` workers, each worker simply builds `N/w`
    trees on the data it has available locally. In many cases, partitioning the
    data so that each worker builds trees on a subset of the total dataset works
    well, but it generally requires the data to be well-shuffled in advance.

    RandomForestClassifier automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.classification.RandomForestClassifier`
    and :py:class:`cuml.ensemble.RandomForestClassifier`. And it can automatically
    map pyspark parameters to cuML parameters.


    Parameters
    ----------

    featuresCol: str or List[str] (default = "features")
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.
    labelCol: str (default = "label")
        The label column name.
    predictionCol: str (default = "prediction")
        The prediction column name.
    probabilityCol: str (default = "probability")
        The column name for predicted class conditional probabilities.
    rawPredictionCol: str (default = "rawPrediction")
        The column name for class raw predictions - this is currently set equal to probabilityCol values.
    maxDepth: int (default = 5)
        Maximum tree depth. Must be greater than 0.
    maxBins: int (default = 32)
        Maximum number of bins used by the split algorithm per feature.
    minInstancesPerNode: int (default = 1)
        The minimum number of samples (rows) in each leaf node.
    impurity: str (default = "gini")
        The criterion used to split nodes.\n
            * ``'gini'`` for gini impurity
            * ``'entropy'`` for information gain (entropy)
    numTrees: int (default = 20)
        Total number of trees in the forest.
    featureSubsetStrategy: str (default = "auto")
        Ratio of number of features (columns) to consider per node split.\n
        The supported options:\n
            ``'auto'``:  If numTrees == 1, set to 'all', If numTrees > 1 (forest), set to 'sqrt'\n
            ``'all'``: use all features\n
            ``'onethird'``: use 1/3 of the features\n
            ``'sqrt'``: use sqrt(number of features)\n
            ``'log2'``: log2(number of features)\n
            ``'n'``: when n is in the range (0, 1.0], use n * number of features. When n
            is in the range (1, number of features), use n features.
    seed: int (default = None)
        Seed for the random number generator.
    bootstrap: bool (default = True)
        Control bootstrapping.\n
            * If ``True``, each tree in the forest is built on a bootstrapped
              sample with replacement.
            * If ``False``, the whole dataset is used to build each tree.
    num_workers:
        Number of cuML workers, where each cuML worker corresponds to one Spark task
        running on one GPU. If not set, spark-rapids-ml tries to infer the number of
        cuML workers (i.e. GPUs in cluster) from the Spark environment.
    verbose:
        Logging level.
            * ``0`` - Disables all log messages.
            * ``1`` - Enables only critical messages.
            * ``2`` - Enables all messages up to and including errors.
            * ``3`` - Enables all messages up to and including warnings.
            * ``4 or False`` - Enables all messages up to and including information messages.
            * ``5 or True`` - Enables all messages up to and including debug messages.
            * ``6`` - Enables all messages up to and including trace messages.
    n_streams: int (default = 4)
        Number of parallel streams used for forest building.
        Please note that there could be a bug running spark-rapids-ml on a node with multi-gpus
        when n_streams > 1. See https://github.com/rapidsai/cuml/issues/5402.
    min_samples_split: int or float (default = 2)
        The minimum number of samples required to split an internal node.\n
         * If type ``int``, then ``min_samples_split`` represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``ceil(min_samples_split * n_rows)`` is the minimum number of
           samples for each split.    max_samples:
        Ratio of dataset rows used while fitting each tree.
    max_leaves: int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
    min_impurity_decrease: float (default = 0.0)
        Minimum decrease in impurity required for node to be split.
    max_batch_size: int (default = 4096)
        Maximum number of nodes that can be processed in a given batch.

    Examples
    --------
    >>> import numpy
    >>> from numpy import allclose
    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.feature import StringIndexer
    >>> df = spark.createDataFrame([
    ...     (1.0, Vectors.dense(1.0)),
    ...     (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    >>> stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
    >>> si_model = stringIndexer.fit(df)
    >>> td = si_model.transform(df)
    >>> from spark_rapids_ml.classification import RandomForestClassifier, RandomForestClassificationModel
    >>> rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="indexed", seed=42)
    >>> model = rf.fit(td)
    >>> model.getLabelCol()
    'indexed'
    >>> model.setFeaturesCol("features")
    RandomForestClassificationModel_...
    >>> model.getBootstrap()
    True
    >>> test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    >>> result = model.transform(test0).head()
    >>> result.prediction
    0.0
    >>> test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    >>> model.transform(test1).head().prediction
    1.0
    >>>
    >>> rfc_path = temp_path + "/rfc"
    >>> rf.save(rfc_path)
    >>> rf2 = RandomForestClassifier.load(rfc_path)
    >>> rf2.getNumTrees()
    3
    >>> model_path = temp_path + "/rfc_model"
    >>> model.save(model_path)
    >>> model2 = RandomForestClassificationModel.load(model_path)
    >>> model2.getNumTrees
    3
    >>> model.transform(test0).take(1) == model2.transform(test0).take(1)
    True

    """

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        probabilityCol: str = "probability",
        rawPredictionCol: str = "rawPrediction",
        maxDepth: int = 5,
        maxBins: int = 32,
        minInstancesPerNode: int = 1,
        impurity: str = "gini",
        numTrees: int = 20,
        featureSubsetStrategy: str = "auto",
        seed: Optional[int] = None,
        bootstrap: Optional[bool] = True,
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        n_streams: int = 1,
        min_samples_split: Union[int, float] = 2,
        max_samples: float = 1.0,
        max_leaves: int = -1,
        min_impurity_decrease: float = 0.0,
        max_batch_size: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(**self._input_kwargs)

    def _pre_process_label(
        self, dataset: DataFrame, feature_type: Union[Type[FloatType], Type[DoubleType]]
    ) -> Column:
        """Cuml RandomForestClassifier requires the int32 type of label column"""
        label_name = self.getLabelCol()
        label_datatype = dataset.schema[label_name].dataType
        if isinstance(label_datatype, (IntegralType, FloatType, DoubleType)):
            label_col = col(label_name).cast(IntegerType()).alias(alias.label)
        else:
            raise ValueError(
                "Label column must be integral types or float/double types."
            )

        return label_col

    def _create_pyspark_model(self, result: Row) -> "RandomForestClassificationModel":
        return RandomForestClassificationModel._from_row(result)

    def _is_classification(self) -> bool:
        return True

    def _supportsTransformEvaluate(self, evaluator: Evaluator) -> bool:
        if (
            isinstance(evaluator, MulticlassClassificationEvaluator)
            and evaluator.getMetricName()
            in MulticlassMetrics.SUPPORTED_MULTI_CLASS_METRIC_NAMES
        ):
            return True

        return False


class RandomForestClassificationModel(
    _ClassificationModelEvaluationMixIn,
    _RandomForestClassifierClass,
    _RandomForestModel,
    _RandomForestCumlParams,
    _RFClassifierParams,
):
    """
    Model fitted by :class:`RandomForestClassifier`.
    """

    def __init__(
        self,
        n_cols: int,
        dtype: str,
        treelite_model: Union[str, List[str]],
        model_json: Union[List[str], List[List[str]]],
        num_classes: int,
    ):
        super().__init__(
            dtype=dtype,
            n_cols=n_cols,
            treelite_model=treelite_model,
            model_json=model_json,
            num_classes=num_classes,
        )
        self._num_classes = num_classes
        self._model_json = model_json
        self._rf_spark_model: Optional[SparkRandomForestClassificationModel] = None
        self._this_model = self

    def cpu(self) -> SparkRandomForestClassificationModel:
        """Return the PySpark ML RandomForestClassificationModel"""

        if self._rf_spark_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            uid, java_trees = self._convert_to_java_trees(self.getImpurity())

            # Create the Spark RandomForestClassificationModel
            java_rf_model = sc._jvm.org.apache.spark.ml.classification.RandomForestClassificationModel(
                uid,
                java_trees,
                self.numFeatures,
                self._num_classes,
            )
            self._rf_spark_model = SparkRandomForestClassificationModel(java_rf_model)
            self._copyValues(self._rf_spark_model)
        return self._rf_spark_model

    def _get_num_models(self) -> int:
        return (
            len(self._treelite_model) if isinstance(self._treelite_model, list) else 1
        )

    def _is_classification(self) -> bool:
        return True

    def _use_prob_as_raw_pred_col(self) -> bool:
        return True

    @property
    def hasSummary(self) -> bool:
        """Indicates whether a training summary exists for this model instance."""
        return False

    @property
    def numClasses(self) -> int:
        """Number of classes (values which the label can take)."""
        return self._num_classes

    def predictRaw(self, value: Vector) -> Vector:
        """
        Raw prediction for each possible label.
        """
        return self.cpu().predictRaw(value)

    def predictProbability(self, value: Vector) -> Vector:
        """
        Predict the probability of each class given the features.
        """
        return self.cpu().predictProbability(value)

    def evaluate(
        self, dataset: DataFrame
    ) -> Union[
        BinaryRandomForestClassificationSummary, RandomForestClassificationSummary
    ]:
        """
        Evaluates the model on a test dataset.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            Test dataset to evaluate model on.
        """
        return self.cpu().evaluate(dataset)

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        _construct_rf, _, _ = super()._get_cuml_transform_func(dataset)

        if eval_metric_info:
            if eval_metric_info.eval_metric == transform_evaluate_metric.log_loss:

                def _predict(rf: CumlT, pdf: TransformInputType) -> "cp.ndarray":
                    rf.update_labels = False
                    return rf.predict_proba(pdf)

            else:

                def _predict(rf: CumlT, pdf: TransformInputType) -> "cp.ndarray":
                    rf.update_labels = False
                    return rf.predict(pdf)

        else:

            def _predict(rf: CumlT, pdf: TransformInputType) -> pd.Series:
                data = {}
                rf.update_labels = False
                data[pred.prediction] = rf.predict(pdf)

                probs = rf.predict_proba(pdf)
                if isinstance(probs, pd.DataFrame):
                    # For 2302, when input is multi-cols, the output will be DataFrame
                    data[pred.probability] = pd.Series(probs.values.tolist())
                else:
                    # should be np.ndarray
                    data[pred.probability] = pd.Series(list(probs))

                return pd.DataFrame(data)

        _evaluate = (
            self._get_evaluate_fn(eval_metric_info) if eval_metric_info else None
        )

        return _construct_rf, _predict, _evaluate


class LogisticRegressionClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "maxIter": "max_iter",
            "regParam": "C",
            "elasticNetParam": "l1_ratio",
            "tol": "tol",
            "fitIntercept": "fit_intercept",
            "threshold": None,
            "thresholds": None,
            "standardization": "standardization",
            "weightCol": None,
            "aggregationDepth": None,
            "family": "",  # family can be 'auto', 'binomial' or 'multinomial', cuml automatically detects num_classes
            "lowerBoundsOnCoefficients": None,
            "upperBoundsOnCoefficients": None,
            "lowerBoundsOnIntercepts": None,
            "upperBoundsOnIntercepts": None,
            "maxBlockSizeInMB": None,
        }

    @classmethod
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[Any], Union[None, str, float, int]]]:
        return {"C": lambda x: 1 / x if x > 0.0 else (0.0 if x == 0.0 else None)}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "fit_intercept": True,
            "standardization": False,
            "verbose": False,
            "C": 1.0,
            "penalty": "l2",
            "l1_ratio": None,
            "max_iter": 1000,
            "tol": 0.0001,
        }

    # Given Spark params: regParam, elasticNetParam,
    # return cuml params: penalty, C, l1_ratio
    @classmethod
    def _reg_params_value_mapping(
        cls, reg_param: float, elasticNet_param: float
    ) -> Tuple[Optional[str], float, float]:
        # Note cuml ignores l1_ratio when penalty is None, "l2", and "l1"
        # Spark Rapids ML sets it to elasticNet_param to be compatible with Spark
        if reg_param == 0.0:
            penalty = None
            C = 0.0
            l1_ratio = elasticNet_param
        elif elasticNet_param == 0.0:
            penalty = "l2"
            C = 1.0 / reg_param
            l1_ratio = elasticNet_param
        elif elasticNet_param == 1.0:
            penalty = "l1"
            C = 1.0 / reg_param
            l1_ratio = elasticNet_param
        else:
            penalty = "elasticnet"
            C = 1.0 / reg_param
            l1_ratio = elasticNet_param

        return (penalty, C, l1_ratio)

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return pyspark.ml.classification.LogisticRegression


class _LogisticRegressionCumlParams(
    _CumlParams,
    _LogisticRegressionParams,
    HasEnableSparseDataOptim,
    HasFeaturesCols,
    HasProbabilityCol,
    HasRawPredictionCol,
):
    def getFeaturesCol(self) -> Union[str, List[str]]:  # type:ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(
        self: "_LogisticRegressionCumlParams", value: Union[str, List[str]]
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featureCols`.
        """
        if isinstance(value, str):
            self._set_params(featuresCol=value)
        else:
            self._set_params(featuresCols=value)
        return self

    def setFeaturesCols(
        self: "_LogisticRegressionCumlParams", value: List[str]
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self._set_params(featuresCols=value)

    def setLabelCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self._set_params(labelCol=value)

    def setPredictionCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set_params(predictionCol=value)

    def setProbabilityCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`probabilityCol`.
        """
        return self._set_params(probabilityCol=value)

    def setRawPredictionCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`rawPredictionCol`.
        """
        return self._set(rawPredictionCol=value)


class LogisticRegression(
    LogisticRegressionClass,
    _CumlEstimatorSupervised,
    _LogisticRegressionCumlParams,
):
    """LogisticRegression is a machine learning model where the response y is modeled
    by the sigmoid (or softmax for more than 2 classes) function applied to a linear
    combination of the features in X. It implements cuML's GPU accelerated
    LogisticRegression algorithm based on cuML python library, and it can be used in
    PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    This supports multiple types of regularization:

    * none
    * L2 (ridge regression)
    * L1 (lasso)
    * L2 + L1 (elastic net)

    LogisticRegression automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.classification.LogisticRegression`.
    And it will automatically map pyspark parameters
    to cuML parameters.

    In the case of applying LogisticRegression on sparse vectors, Spark 3.4 or above is required.

    Parameters
    ----------
    featuresCol: str or List[str] (default = "features")
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.
    labelCol: (default = "label")
        The label column name.
    predictionCol: (default = "prediction")
        The class prediction column name.
    probabilityCol: (default = "probability")
        The probability prediction column name.
    rawPredictionCol: (default = "rawPrediction")
        The column name for class raw predictions - this is currently set equal to probabilityCol values.
    maxIter: (default = 100)
        The maximum number of iterations of the underlying L-BFGS algorithm.
    regParam: (default = 0.0)
        The regularization parameter.
    elasticNetParam: (default = 0.0)
        The ElasticNet mixing parameter, in range [0, 1]. For alpha = 0,
        the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
    tol: (default = 1e-6)
        The convergence tolerance.
    enable_sparse_data_optim: None or boolean, optional (default=None)
        If features column is VectorUDT type, Spark rapids ml relies on this parameter to decide whether to use dense array or sparse array in cuml.
        If None, use dense array if the first VectorUDT of a dataframe is DenseVector. Use sparse array if it is SparseVector.
        If False, always uses dense array. This is favorable if the majority of VectorUDT vectors are DenseVector.
        If True, always uses sparse array. This is favorable if the majority of the VectorUDT vectors are SparseVector.
        Note this is only supported in spark >= 3.4.
    fitIntercept: (default = True)
        Whether to fit an intercept term.
    standardization: (default = True)
        Whether to standardize the training data before fit.
    num_workers:
        Number of cuML workers, where each cuML worker corresponds to one Spark task
        running on one GPU. If not set, spark-rapids-ml tries to infer the number of
        cuML workers (i.e. GPUs in cluster) from the Spark environment.
    verbose:
    Logging level.
            * ``0`` - Disables all log messages.
            * ``1`` - Enables only critical messages.
            * ``2`` - Enables all messages up to and including errors.
            * ``3`` - Enables all messages up to and including warnings.
            * ``4 or False`` - Enables all messages up to and including information messages.
            * ``5 or True`` - Enables all messages up to and including debug messages.
            * ``6`` - Enables all messages up to and including trace messages.

    Examples
    --------
    >>> from spark_rapids_ml.classification import LogisticRegression
    >>> data = [
    ...     ([1.0, 2.0], 1.0),
    ...     ([1.0, 3.0], 1.0),
    ...     ([2.0, 1.0], 0.0),
    ...     ([3.0, 1.0], 0.0),
    ... ]
    >>> schema = "features array<float>, label float"
    >>> df = spark.createDataFrame(data, schema=schema)
    >>> df.show()
    +----------+-----+
    |  features|label|
    +----------+-----+
    |[1.0, 2.0]|  1.0|
    |[1.0, 3.0]|  1.0|
    |[2.0, 1.0]|  0.0|
    |[3.0, 1.0]|  0.0|
    +----------+-----+

    >>> lr_estimator = LogisticRegression()
    >>> lr_estimator.setFeaturesCol("features")
    LogisticRegression_a757215437b0
    >>> lr_estimator.setLabelCol("label")
    LogisticRegression_a757215437b0
    >>> lr_model = lr_estimator.fit(df)
    >>> lr_model.coefficients
    DenseVector([-0.7148, 0.7148])
    >>> lr_model.intercept
    -8.543887375367376e-09
    """

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        probabilityCol: str = "probability",
        rawPredictionCol: str = "rawPrediction",
        maxIter: int = 100,
        regParam: float = 0.0,
        elasticNetParam: float = 0.0,
        tol: float = 1e-6,
        fitIntercept: bool = True,
        standardization: bool = True,
        enable_sparse_data_optim: Optional[bool] = None,
        float32_inputs: bool = True,
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ):
        super().__init__()
        self._set_cuml_reg_params()
        self._set_params(**self._input_kwargs)

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        array_order = self._fit_array_order()
        standardization = self.getStandardization()
        fit_intercept = self.getFitIntercept()

        logger = get_logger(self.__class__)
        float32_input = self._float32_inputs

        def _logistic_regression_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            import cupyx
            from cuml.linear_model.logistic_regression_mg import LogisticRegressionMG

            X_list = [x for (x, _, _) in dfs]
            y_list = [y for (_, y, _) in dfs]

            if isinstance(X_list[0], pd.DataFrame):
                concated = pd.concat(X_list)
                concated_y = pd.concat(y_list)
            else:
                # features are either cp, np, scipy csr or cupyx csr arrays here
                concated = _concat_and_free(X_list, order=array_order)
                concated_y = _concat_and_free(y_list, order=array_order)

            is_sparse = isinstance(concated, scipy.sparse.csr_matrix) or isinstance(
                concated, cupyx.scipy.sparse.csr_matrix
            )

            pdesc = PartitionDescriptor.build(
                [concated.shape[0]],
                params[param_alias.num_cols],
            )

            def _single_fit(init_parameters: Dict[str, Any]) -> Dict[str, Any]:

                if init_parameters["C"] == 0.0:
                    init_parameters["penalty"] = None

                elif init_parameters["l1_ratio"] == 0.0:
                    init_parameters["penalty"] = "l2"

                elif init_parameters["l1_ratio"] == 1.0:
                    init_parameters["penalty"] = "l1"

                else:
                    init_parameters["penalty"] = "elasticnet"

                logistic_regression = LogisticRegressionMG(
                    handle=params[param_alias.handle],
                    **init_parameters,
                )

                logistic_regression.penalty_normalized = False
                logistic_regression.lbfgs_memory = 10

                logistic_regression.fit(
                    [(concated, concated_y)],
                    pdesc.m,
                    pdesc.n,
                    pdesc.parts_rank_size,
                    pdesc.rank,
                )

                coef_ = logistic_regression.coef_
                intercept_ = logistic_regression.intercept_
                intercept_array = intercept_
                # follow Spark to center the intercepts for multinomial classification
                if (
                    init_parameters["fit_intercept"] is True
                    and len(intercept_array) > 1
                ):
                    import cupy as cp

                    intercept_mean = (
                        np.mean(intercept_array)
                        if isinstance(intercept_array, np.ndarray)
                        else cp.mean(intercept_array)
                    )
                    intercept_array -= intercept_mean

                n_cols = logistic_regression.n_cols

                # index_dtype is only available in sparse logistic regression. It records the dtype of indices array and indptr array that were used in C++ computation layer. Its value can be 'int32' or 'int64'.
                index_dtype = (
                    str(logistic_regression.index_dtype)
                    if hasattr(logistic_regression, "index_dtype")
                    else "None"
                )

                model = {
                    "coef_": coef_[:, :n_cols].tolist(),
                    "intercept_": intercept_.tolist(),
                    "classes_": logistic_regression.classes_.tolist(),
                    "n_cols": n_cols,
                    "dtype": logistic_regression.dtype.name,
                    "num_iters": logistic_regression.solver_model.num_iters,
                    "objective": logistic_regression.solver_model.objective,
                    "index_dtype": index_dtype,
                }

                # check if invalid label exists
                for class_val in model["classes_"]:
                    if class_val < 0:
                        raise RuntimeError(
                            f"Labels MUST be in [0, 2147483647), but got {class_val}"
                        )
                    elif not class_val.is_integer():
                        raise RuntimeError(
                            f"Labels MUST be Integers, but got {class_val}"
                        )

                if len(logistic_regression.classes_) == 1:
                    class_val = logistic_regression.classes_[0]
                    # TODO: match Spark to use max(class_list) to calculate the number of classes
                    # Cuml currently uses unique(class_list)
                    if class_val != 1.0 and class_val != 0.0:
                        raise RuntimeError(
                            "class value must be either 1. or 0. when dataset has one label"
                        )

                    if init_parameters["fit_intercept"] is True:
                        model["coef_"] = [[0.0] * n_cols]
                        model["intercept_"] = [
                            float("inf") if class_val == 1.0 else float("-inf")
                        ]

                del logistic_regression
                return model

            init_parameters = params[param_alias.cuml_init]
            fit_multiple_params = params[param_alias.fit_multiple_params]
            if len(fit_multiple_params) == 0:
                fit_multiple_params.append({})

            models = []
            for i in range(len(fit_multiple_params)):
                tmp_params = init_parameters.copy()
                tmp_params.update(fit_multiple_params[i])
                models.append(_single_fit(tmp_params))

            models_dict = {}
            tc = TaskContext.get()
            assert tc is not None
            if tc.partitionId() == 0:
                for k in models[0].keys():
                    models_dict[k] = [m[k] for m in models]
            return models_dict

        return _logistic_regression_fit

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

        return select_cols, multi_col_names, dimension, feature_type

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("coef_", ArrayType(ArrayType(DoubleType()), False), False),
                StructField("intercept_", ArrayType(DoubleType()), False),
                StructField("classes_", ArrayType(DoubleType()), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
                StructField("num_iters", IntegerType(), False),
                StructField("objective", DoubleType(), False),
                StructField("index_dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "LogisticRegressionModel":
        logger = get_logger(self.__class__)
        if len(result["classes_"]) == 1:
            if self.getFitIntercept() is False:
                logger.warning(
                    "All labels belong to a single class and fitIntercept=false. It's a dangerous ground, so the algorithm may not converge."
                )
            else:
                logger.warning(
                    "All labels are the same value and fitIntercept=true, so the coefficients will be zeros. Training is not needed."
                )

        d = result.asDict()
        self._index_dtype = d.pop("index_dtype")

        return LogisticRegressionModel._from_row(Row(**d))

    def _set_cuml_reg_params(self) -> "LogisticRegression":
        penalty, C, l1_ratio = self._reg_params_value_mapping(
            self.getRegParam(), self.getElasticNetParam()
        )
        self._cuml_params["penalty"] = penalty
        self._cuml_params["C"] = C
        self._cuml_params["l1_ratio"] = l1_ratio
        return self

    def _set_params(self, **kwargs: Any) -> "LogisticRegression":
        super()._set_params(**kwargs)
        if "regParam" in kwargs or "elasticNetParam" in kwargs:
            self._set_cuml_reg_params()
        return self

    def setMaxIter(self, value: int) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`maxIter`.
        """
        return self._set_params(maxIter=value)

    def setRegParam(self, value: float) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`regParam`.
        """
        return self._set_params(regParam=value)

    def setElasticNetParam(self, value: float) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`regParam`.
        """
        return self._set_params(elasticNetParam=value)

    def setTol(self, value: float) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`tol`.
        """
        return self._set_params(tol=value)

    def setFitIntercept(self, value: bool) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`fitIntercept`.
        """
        return self._set_params(fitIntercept=value)

    def setStandardization(self, value: bool) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`standardization`.
        """
        return self._set_params(standardization=value)

    def _enable_fit_multiple_in_single_pass(self) -> bool:
        return True

    def _supportsTransformEvaluate(self, evaluator: Evaluator) -> bool:
        if (
            isinstance(evaluator, MulticlassClassificationEvaluator)
            and evaluator.getMetricName()
            in MulticlassMetrics.SUPPORTED_MULTI_CLASS_METRIC_NAMES
        ):
            return True

        return False


class LogisticRegressionModel(
    LogisticRegressionClass,
    _ClassificationModelEvaluationMixIn,
    _CumlModelWithPredictionCol,
    _LogisticRegressionCumlParams,
):
    """Model fitted by :class:`LogisticRegression`."""

    def __init__(
        self,
        coef_: Union[List[List[float]], List[List[List[float]]]],
        intercept_: Union[List[float], List[List[float]]],
        classes_: List[float],
        n_cols: int,
        dtype: str,
        num_iters: int,
        objective: float,
    ) -> None:
        super().__init__(
            dtype=dtype,
            n_cols=n_cols,
            coef_=coef_,
            intercept_=intercept_,
            classes_=classes_,
            num_iters=num_iters,
            objective=objective,
        )
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.classes_ = classes_
        self._lr_spark_model: Optional[SparkLogisticRegressionModel] = None
        self._num_classes = len(self.classes_)
        self.num_iters = num_iters
        self.objective = objective
        self._this_model = self

    def cpu(self) -> SparkLogisticRegressionModel:
        """Return the PySpark ML LogisticRegressionModel"""
        if self._lr_spark_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            is_multinomial = False if len(self.classes_) == 2 else True

            assert self.n_cols is not None

            java_model = (
                sc._jvm.org.apache.spark.ml.classification.LogisticRegressionModel(
                    java_uid(sc, "logreg"),
                    _py2java(sc, self.coefficientMatrix),
                    _py2java(sc, self.interceptVector),
                    self._num_classes,
                    is_multinomial,
                )
            )
            self._lr_spark_model = SparkLogisticRegressionModel(java_model)
            self._copyValues(self._lr_spark_model)

        return self._lr_spark_model

    def _get_num_models(self) -> int:
        return 1 if isinstance(self.intercept_[0], float) else len(self.intercept_)

    @property
    def coefficients(self) -> Vector:
        """
        Model coefficients.
        """
        if isinstance(self.coef_[0][0], float):
            if len(self.coef_) == 1:
                return Vectors.dense(cast(list, self.coef_[0]))
            else:
                raise Exception(
                    "Multinomial models contain a matrix of coefficients, use coefficientMatrix instead."
                )
        else:
            raise Exception("coefficients not defined for multi-model instance")

    @property
    def intercept(self) -> float:
        """
        Model intercept.
        """
        if isinstance(self.intercept_[0], float):
            if len(self.intercept_) == 1:
                return self.intercept_[0]
            else:
                raise Exception(
                    "Multinomial models contain a vector of intercepts, use interceptVector instead."
                )
        else:
            raise Exception("intercept not defined for multi-model instance")

    @property
    def coefficientMatrix(self) -> Matrix:
        """
        Model coefficients.
        Note Spark CPU uses denseCoefficientMatrix.compressed that may return a sparse vector
        if there are many zero values. Since the compressed function is not available in pyspark,
        Spark Rapids ML always returns a dense vector.
        """

        if isinstance(self.coef_[0][0], float):
            n_rows = len(self.coef_)
            n_cols = len(self.coef_[0])
            flat_coef = [cast(float, c) for row in self.coef_ for c in row]
            return DenseMatrix(
                numRows=n_rows, numCols=n_cols, values=flat_coef, isTransposed=True
            )
        else:
            raise Exception("coefficientMatrix not defined for multi-model instance")

    @property
    def interceptVector(self) -> Vector:
        """
        Model intercept.
        """

        if isinstance(self.intercept_[0], float):
            nnz = np.count_nonzero(self.intercept_)

            # spark returns interceptVec.compressed
            # According spark doc, a dense vector needs 8 * size + 8 bytes, while a sparse vector needs 12 * nnz + 20 bytes.
            if 1.5 * (nnz + 1.0) < len(self.intercept_):
                size = len(self.intercept_)
                data_m = {p[0]: cast(float, p[1]) for p in enumerate(self.intercept_)}
                return Vectors.sparse(size, data_m)
            else:
                return Vectors.dense(cast(list, self.intercept_))
        else:
            raise Exception("interceptVector not defined for multi-model instance")

    @property
    def numClasses(self) -> int:
        return self._num_classes

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        coef_ = self.coef_
        intercept_ = self.intercept_
        classes_ = self.classes_
        n_cols = self.n_cols
        dtype = self.dtype

        num_models = self._get_num_models()

        # from cuml logistic_regression.pyx
        def _predict_proba(scores: "cp.ndarray", _num_classes: int) -> "cp.ndarray":
            import cupy as cp

            if _num_classes == 2:
                proba = cp.zeros((scores.shape[0], 2))
                proba[:, 1] = 1 / (1 + cp.exp(-scores.ravel()))
                proba[:, 0] = 1 - proba[:, 1]
            elif _num_classes > 2:
                max_scores = cp.max(scores, axis=1).reshape((-1, 1))
                scores_shifted = scores - max_scores
                proba = cp.exp(scores_shifted)
                row_sum = cp.sum(proba, axis=1).reshape((-1, 1))
                proba /= row_sum
            return proba

        def _predict_labels(scores: "cp.ndarray", _num_classes: int) -> "cp.ndarray":
            import cupy as cp

            _num_classes = max(scores.shape[1] if len(scores.shape) == 2 else 2, 2)
            if _num_classes == 2:
                predictions = (scores.ravel() > 0).astype("float32")
            else:
                predictions = cp.argmax(scores, axis=1)
            return predictions

        def _construct_lr() -> CumlT:
            import cupy as cp
            import numpy as np
            from cuml.internals.input_utils import input_to_cuml_array
            from cuml.linear_model.logistic_regression_mg import LogisticRegressionMG

            _intercepts, _coefs = (
                (intercept_, coef_) if num_models > 1 else ([intercept_], [coef_])
            )
            lrs = []

            for i in range(num_models):
                lr = LogisticRegressionMG(output_type="cupy")
                lr.n_cols = n_cols
                lr.dtype = np.dtype(dtype)

                gpu_intercept_ = cp.array(_intercepts[i], order="C", dtype=dtype)

                gpu_coef_ = cp.array(_coefs[i], order="F", dtype=dtype).T
                gpu_stacked = cp.vstack([gpu_coef_, gpu_intercept_])
                lr.solver_model._coef_ = input_to_cuml_array(
                    gpu_stacked, order="C"
                ).array

                lr.classes_ = input_to_cuml_array(
                    np.array(classes_, order="F").astype(dtype)
                ).array
                lr._num_classes = len(lr.classes_)

                lr.loss = "sigmoid" if lr._num_classes <= 2 else "softmax"
                lr.solver_model.qnparams = lr.create_qnparams()
                lrs.append(lr)

            return lrs

        _evaluate = (
            self._get_evaluate_fn(eval_metric_info) if eval_metric_info else None
        )

        if eval_metric_info:
            if eval_metric_info.eval_metric == transform_evaluate_metric.log_loss:

                def _predict(lr: CumlT, pdf: TransformInputType) -> "cp.ndarray":

                    return lr.predict_proba(pdf)

            else:

                def _predict(lr: CumlT, pdf: TransformInputType) -> "cp.ndarray":

                    return lr.predict(pdf)

        else:

            def _predict(lr: CumlT, pdf: TransformInputType) -> pd.DataFrame:
                import cupy as cp

                data = {}

                scores = lr.decision_function(pdf)
                assert isinstance(scores, cp.ndarray)
                _num_classes = max(scores.shape[1] if len(scores.shape) == 2 else 2, 2)

                data[pred.prediction] = pd.Series(
                    list(_predict_labels(scores, _num_classes).get())
                )

                data[pred.probability] = pd.Series(
                    list(_predict_proba(scores, _num_classes).get())
                )
                if _num_classes == 2:
                    raw_prediction = cp.zeros((scores.shape[0], 2))
                    raw_prediction[:, 1] = scores.ravel()
                    raw_prediction[:, 0] = -raw_prediction[:, 1]
                elif _num_classes > 2:
                    raw_prediction = scores
                data[pred.raw_prediction] = pd.Series(list(cp.asnumpy(raw_prediction)))

                return pd.DataFrame(data)

        return _construct_lr, _predict, _evaluate

    @classmethod
    def _combine(
        cls: Type["LogisticRegressionModel"], models: List["LogisticRegressionModel"]  # type: ignore
    ) -> "LogisticRegressionModel":
        assert len(models) > 0 and all(isinstance(model, cls) for model in models)
        first_model = models[0]
        intercepts = [model.intercept_ for model in models]
        coefs = [model.coef_ for model in models]
        attrs = first_model._get_model_attributes()
        assert attrs is not None
        attrs["coef_"] = coefs
        attrs["intercept_"] = intercepts
        lr_model = cls(**attrs)
        first_model._copyValues(lr_model)
        first_model._copy_cuml_params(lr_model)
        return lr_model

    @property
    def hasSummary(self) -> bool:
        """
        Indicates whether a training summary exists for this model
        instance.
        """
        return False

    @property
    def summary(self) -> "LogisticRegressionTrainingSummary":
        """
        Gets summary (accuracy/precision/recall, objective history, total iterations) of model
        trained on the training set. An exception is thrown if `trainingSummary is None`.
        """
        raise RuntimeError(
            "No training summary available for this %s" % self.__class__.__name__
        )

    def predict(self, value: Vector) -> float:
        """cuML doesn't support predicting 1 single sample.
        Fall back to PySpark ML LogisticRegressionModel"""
        return self.cpu().predict(value)

    def evaluate(self, dataset: DataFrame) -> LogisticRegressionSummary:
        """cuML doesn't support evaluating.
        Fall back to PySpark ML LogisticRegressionModel"""
        return self.cpu().evaluate(dataset)

    def predictRaw(self, value: Vector) -> Vector:
        """
        Raw prediction for each possible label.
        Fall back to PySpark ML LogisticRegressionModel
        """
        return self.cpu().predictRaw(value)

    def predictProbability(self, value: Vector) -> Vector:
        """
        Predict the probability of each class given the features.
        Fall back to PySpark ML LogisticRegressionModel
        """
        return self.cpu().predictProbability(value)

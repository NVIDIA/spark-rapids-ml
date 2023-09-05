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

from pyspark.ml.common import _py2java
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator

from .metrics.MulticlassMetrics import MulticlassMetrics

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap

import numpy as np
import pandas as pd
from pyspark import Row, keyword_only
from pyspark.ml.classification import BinaryRandomForestClassificationSummary
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
from pyspark.ml.linalg import DenseMatrix, Vector, Vectors
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from pyspark.sql import Column, DataFrame
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
    transform_evaluate,
)
from .params import HasFeaturesCols, _CumlClass, _CumlParams
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
        mapping["rawPredictionCol"] = ""
        return mapping


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

    featuresCol:
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.
    labelCol:
        The label column name.
    predictionCol:
        The prediction column name.
    probabilityCol
        The column name for predicted class conditional probabilities.
    maxDepth:
        Maximum tree depth. Must be greater than 0.
    maxBins:
        Maximum number of bins used by the split algorithm per feature.
    minInstancesPerNode:
        The minimum number of samples (rows) in each leaf node.
    impurity: str = "gini",
        The criterion used to split nodes.\n
            * ``'gini'`` for gini impurity
            * ``'entropy'`` for information gain (entropy)
    numTrees:
        Total number of trees in the forest.
    featureSubsetStrategy:
        Ratio of number of features (columns) to consider per node split.\n
        The supported options:\n
            ``'auto'``:  If numTrees == 1, set to 'all', If numTrees > 1 (forest), set to 'sqrt'\n
            ``'all'``: use all features\n
            ``'onethird'``: use 1/3 of the features\n
            ``'sqrt'``: use sqrt(number of features)\n
            ``'log2'``: log2(number of features)\n
            ``'n'``: when n is in the range (0, 1.0], use n * number of features. When n
            is in the range (1, number of features), use n features.
    seed:
        Seed for the random number generator.
    bootstrap:
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
    n_streams:
        Number of parallel streams used for forest building.
        Please note that there is a bug running spark-rapids-ml on a node with multi-gpus
        when n_streams > 1. See https://github.com/rapidsai/cuml/issues/5402.
    min_samples_split:
        The minimum number of samples required to split an internal node.\n
         * If type ``int``, then ``min_samples_split`` represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``ceil(min_samples_split * n_rows)`` is the minimum number of
           samples for each split.    max_samples:
        Ratio of dataset rows used while fitting each tree.
    max_leaves:
        Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
    min_impurity_decrease:
        Minimum decrease in impurity required for node to be split.
    max_batch_size:
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
        return RandomForestClassificationModel.from_row(result)

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

    def _is_classification(self) -> bool:
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
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        _construct_rf, _, _ = super()._get_cuml_transform_func(dataset)

        def _predict(rf: CumlT, pdf: TransformInputType) -> pd.Series:
            data = {}
            rf.update_labels = False
            data[pred.prediction] = rf.predict(pdf)

            if category == transform_evaluate.transform:
                # transform_evaluate doesn't need probs for f1 score.
                probs = rf.predict_proba(pdf)
                if isinstance(probs, pd.DataFrame):
                    # For 2302, when input is multi-cols, the output will be DataFrame
                    data[pred.probability] = pd.Series(probs.values.tolist())
                else:
                    # should be np.ndarray
                    data[pred.probability] = pd.Series(list(probs))

            return pd.DataFrame(data)

        def _evaluate(
            input: TransformInputType,
            transformed: TransformInputType,
        ) -> pd.DataFrame:
            # calculate the count of (label, prediction)
            comb = pd.DataFrame(
                {
                    "label": input[alias.label],
                    "prediction": transformed[pred.prediction],
                }
            )
            confusion = (
                comb.groupby(["label", "prediction"]).size().reset_index(name="total")
            )
            return confusion

        return _construct_rf, _predict, _evaluate

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

        if self.getLabelCol() not in dataset.schema.names:
            raise RuntimeError("Label column is not existing.")

        dataset = dataset.withColumnRenamed(self.getLabelCol(), alias.label)

        schema = StructType(
            [
                StructField(pred.model_index, IntegerType()),
                StructField("label", FloatType()),
                StructField("prediction", FloatType()),
                StructField("total", FloatType()),
            ]
        )

        rows = super()._transform_evaluate_internal(dataset, schema).collect()

        num_models = (
            len(self._treelite_model) if isinstance(self._treelite_model, list) else 1
        )

        tp_by_class: List[Dict[float, float]] = [{} for _ in range(num_models)]
        fp_by_class: List[Dict[float, float]] = [{} for _ in range(num_models)]
        label_count_by_class: List[Dict[float, float]] = [{} for _ in range(num_models)]
        label_count = [0 for _ in range(num_models)]

        for i in range(num_models):
            for j in range(self._num_classes):
                tp_by_class[i][float(j)] = 0.0
                label_count_by_class[i][float(j)] = 0.0
                fp_by_class[i][float(j)] = 0.0

        for row in rows:
            label_count[row.model_index] += row.total
            label_count_by_class[row.model_index][row.label] += row.total

            if row.label == row.prediction:
                tp_by_class[row.model_index][row.label] += row.total
            else:
                fp_by_class[row.model_index][row.prediction] += row.total

        scores = []
        for i in range(num_models):
            metrics = MulticlassMetrics(
                tp=tp_by_class[i],
                fp=fp_by_class[i],
                label=label_count_by_class[i],
                label_count=label_count[i],
            )
            scores.append(metrics.evaluate(evaluator))
        return scores


class LogisticRegressionClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "maxIter": "max_iter",
            "regParam": "C",  # regParam = 1/C
            "tol": "tol",
            "fitIntercept": "fit_intercept",
            "elasticNetParam": None,
            "threshold": None,
            "thresholds": None,
            "standardization": "",  # Set to "" instead of None because cuml defaults to standardization = False
            "weightCol": None,
            "aggregationDepth": None,
            "family": "",  # family can be 'auto', 'binomial' or 'multinomial', cuml automatically detects num_classes
            "lowerBoundsOnCoefficients": None,
            "upperBoundsOnCoefficients": None,
            "lowerBoundsOnIntercepts": None,
            "upperBoundsOnIntercepts": None,
            "maxBlockSizeInMB": None,
            "rawPredictionCol": "",
        }

    @classmethod
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[Any], Union[None, str, float, int]]]:
        def regParam_value_mapper(x: float) -> float:
            # TODO: remove this checking and set regParam to 0.0 once no regularization is supported
            if x == 0.0:
                logger = get_logger(cls)
                logger.warning(
                    "no regularization is not supported yet. if regParam is set to 0,"
                    + "it will be mapped to smallest positive float, i.e. numpy.finfo('float32').tiny"
                )

                return 1.0 / np.finfo("float32").tiny.item()
            else:
                return 1.0 / x

        return {"C": lambda x: regParam_value_mapper(x)}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "fit_intercept": True,
            "verbose": False,
            "C": 1.0,
            "max_iter": 1000,
            "tol": 0.0001,
        }


class _LogisticRegressionCumlParams(
    _CumlParams,
    _LogisticRegressionParams,
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
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(
        self: "_LogisticRegressionCumlParams", value: List[str]
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self.set_params(featuresCols=value)

    def setLabelCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self.set_params(labelCol=value)

    def setPredictionCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self.set_params(predictionCol=value)

    def setProbabilityCol(
        self: "_LogisticRegressionCumlParams", value: str
    ) -> "_LogisticRegressionCumlParams":
        """
        Sets the value of :py:attr:`probabilityCol`.
        """
        return self.set_params(probabilityCol=value)

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

    This currently supports the regularization options:

    * none
    * L2 (ridge regression)

    and two classes.

    LogisticRegression automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.classification.LogisticRegression`.
    And it will automatically map pyspark parameters
    to cuML parameters.

    Parameters
    ----------
    featuresCol:
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.
    labelCol:
        The label column name.
    predictionCol:
        The class prediction column name.
    probabilityCol:
        The probability prediction column name.
    maxIter:
        The maximum number of iterations of the underlying L-BFGS algorithm.
    regParam:
        The regularization parameter.
    tol:
        The convergence tolerance.
    fitIntercept:
        Whether to fit an intercept term.
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
        maxIter: int = 100,
        regParam: float = 0.0,  # NOTE: the default value of regParam is actually mapped to sys.float_info.min on GPU
        tol: float = 1e-6,
        fitIntercept: bool = True,
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ):
        if not self._input_kwargs.get("float32_inputs", True):
            get_logger(self.__class__).warning(
                "This estimator does not support double precision inputs. Setting float32_inputs to False will be ignored."
            )
            self._input_kwargs.pop("float32_inputs")
        super().__init__()
        self.set_params(**self._input_kwargs)

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
        array_order = self._fit_array_order()

        def _logistic_regression_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            init_parameters = params[param_alias.cuml_init]

            from cuml.linear_model.logistic_regression_mg import LogisticRegressionMG

            logistic_regression = LogisticRegressionMG(
                handle=params[param_alias.handle],
                **init_parameters,
            )

            logistic_regression.penalty_normalized = False
            logistic_regression.lbfgs_memory = 10

            X_list = [x for (x, _, _) in dfs]
            y_list = [y for (_, y, _) in dfs]
            if isinstance(X_list[0], pd.DataFrame):
                concated = pd.concat(X_list)
                concated_y = pd.concat(y_list)
            else:
                # features are either cp or np arrays here
                concated = _concat_and_free(X_list, order=array_order)
                concated_y = _concat_and_free(y_list, order=array_order)

            pdesc = PartitionDescriptor.build(
                [concated.shape[0]], params[param_alias.num_cols]
            )

            logistic_regression.fit(
                [(concated, concated_y)],
                pdesc.m,
                pdesc.n,
                pdesc.parts_rank_size,
                pdesc.rank,
            )

            return {
                "coef_": [logistic_regression.coef_.tolist()],
                "intercept_": [logistic_regression.intercept_.tolist()],
                "n_cols": [logistic_regression.n_cols],
                "dtype": [logistic_regression.dtype.name],
            }

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
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "LogisticRegressionModel":
        return LogisticRegressionModel.from_row(result)

    def setMaxIter(self, value: int) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`maxIter`.
        """
        return self.set_params(maxIter=value)

    def setRegParam(self, value: float) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`regParam`.
        """
        return self.set_params(regParam=value)

    def setTol(self, value: float) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`tol`.
        """
        return self.set_params(tol=value)

    def setFitIntercept(self, value: bool) -> "LogisticRegression":
        """
        Sets the value of :py:attr:`fitIntercept`.
        """
        return self.set_params(fitIntercept=value)


class LogisticRegressionModel(
    LogisticRegressionClass,
    _CumlModelWithPredictionCol,
    _LogisticRegressionCumlParams,
):
    """Model fitted by :class:`LogisticRegression`."""

    def __init__(
        self,
        coef_: List[List[float]],
        intercept_: List[float],
        n_cols: int,
        dtype: str,
    ) -> None:
        super().__init__(dtype=dtype, n_cols=n_cols, coef_=coef_, intercept_=intercept_)
        self.coef_ = coef_
        self.intercept_ = intercept_
        self._lr_spark_model: Optional[SparkLogisticRegressionModel] = None

    def cpu(self) -> SparkLogisticRegressionModel:
        """Return the PySpark ML LogisticRegressionModel"""
        if self._lr_spark_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            # TODO Multinomial is not supported yet.
            num_classes = 2
            is_multinomial = False
            num_coefficient_sets = 1
            coefficients = self.coef_[0]

            assert self.n_cols is not None
            coefficients_dmatrix = DenseMatrix(
                num_coefficient_sets, self.n_cols, list(coefficients), True
            )
            intercepts = Vectors.dense(self.intercept)

            java_model = (
                sc._jvm.org.apache.spark.ml.classification.LogisticRegressionModel(
                    java_uid(sc, "logreg"),
                    _py2java(sc, coefficients_dmatrix),
                    _py2java(sc, intercepts),
                    num_classes,
                    is_multinomial,
                )
            )
            self._lr_spark_model = SparkLogisticRegressionModel(java_model)
            self._copyValues(self._lr_spark_model)

        return self._lr_spark_model

    @property
    def coefficients(self) -> Vector:
        """
        Model coefficients.
        """
        assert len(self.coef_) == 1, "multi classes not supported yet"
        return Vectors.dense(cast(list, self.coef_[0]))

    @property
    def intercept(self) -> float:
        """
        Model intercept.
        """
        assert len(self.intercept_) == 1, "multi classes not supported yet"
        return self.intercept_[0]

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        coef_ = self.coef_
        intercept_ = self.intercept_
        n_cols = self.n_cols
        dtype = self.dtype

        def _construct_lr() -> CumlT:
            import numpy as np
            from cuml.internals.input_utils import input_to_cuml_array
            from cuml.linear_model.logistic_regression_mg import LogisticRegressionMG

            lr = LogisticRegressionMG(output_type="numpy")
            lr.n_cols = n_cols
            lr.dtype = np.dtype(dtype)
            lr.intercept_ = input_to_cuml_array(
                np.array(intercept_, order="C").astype(dtype)
            ).array
            lr.coef_ = input_to_cuml_array(
                np.array(coef_, order="C").astype(dtype)
            ).array
            # TBD: infer class indices from data for > 2 classes
            # needed for predict_proba
            lr.classes_ = input_to_cuml_array(
                np.array([0, 1], order="F").astype(dtype)
            ).array
            return lr

        def _predict(lr: CumlT, pdf: TransformInputType) -> pd.DataFrame:
            data = {}
            data[pred.prediction] = lr.predict(pdf)
            probs = lr.predict_proba(pdf)
            if isinstance(probs, pd.DataFrame):
                data[pred.probability] = pd.Series(probs.values.tolist())
            else:
                # should be np.ndarray
                data[pred.probability] = pd.Series(list(probs))
            return pd.DataFrame(data)

        return _construct_lr, _predict, None

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

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
import json
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import cudf
import numpy as np
import pandas as pd
from pyspark import Row
from pyspark.ml.classification import (
    BinaryRandomForestClassificationSummary,
    DecisionTreeClassificationModel,
)
from pyspark.ml.classification import (
    RandomForestClassificationModel as SparkRandomForestClassificationModel,
)
from pyspark.ml.classification import (
    RandomForestClassificationSummary,
    _RandomForestClassifierParams,
)
from pyspark.ml.linalg import Vector
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, FloatType, IntegerType, IntegralType

from .core import CumlT, alias, pred
from .tree import (
    _RandomForestClass,
    _RandomForestCumlParams,
    _RandomForestEstimator,
    _RandomForestModel,
)
from .utils import _get_spark_session, java_uid, translate_trees


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


class RandomForestClassifier(
    _RandomForestClass,
    _RandomForestEstimator,
    _RandomForestCumlParams,
    _RFClassifierParams,
):
    """RandomForestClassifier implements a Random Forest classifier model which
    fits multiple decision tree classifiers in an ensemble. It supports both
    binary and multiclass labels. It implements cuML's GPU accelerated
    RandomForestClassifier algorithm based on cuML python library,
    and it can be used in PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    The distributed algorithm uses an *embarrassingly-parallel* approach. For a
    forest with `N` trees being built on `w` workers, each worker simply builds `N/w`
    trees on the data it has available locally. In many cases, partitioning the
    data so that each worker builds trees on a subset of the total dataset works
    well, but it generally requires the data to be well-shuffled in advance.

    RandomForestClassifier automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.classification.RandomForestClassifier` and in the constructors of
    :py:class:`cuml.ensemble.RandomForestClassifier`. And it can automatically map pyspark
    parameters to cuML parameters.

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

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.set_params(**kwargs)

    def setNumTrees(self, value: int) -> "RandomForestClassifier":
        """
        Sets the value of :py:attr:`numTrees`.
        """
        return self._set(numTrees=value)

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


class RandomForestClassificationModel(
    _RandomForestClass,
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
        treelite_model: str,
        model_json: List[str],
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
        if self.getImpurity() != "gini":
            # TODO, support entropy impurity
            raise ValueError(
                "Can't convert to Spark RandomForestClassificationModel"
                " when impurity is not gini"
            )

        if self._rf_spark_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None
            assert sc._gateway is not None

            uid = java_uid(sc, "rfc")

            # Convert cuml trees to Spark trees
            trees = [
                translate_trees(sc, trees)
                for trees_json in self._model_json
                for trees in json.loads(trees_json)
            ]

            # Wrap the trees into Spark DecisionTreeClassificationModel
            decision_trees = [
                sc._jvm.org.apache.spark.ml.classification.DecisionTreeClassificationModel(
                    uid, tree, self.numFeatures, self._num_classes
                )
                for tree in trees
            ]
            object_class = (
                sc._jvm.org.apache.spark.ml.classification.DecisionTreeClassificationModel
            )
            java_trees = sc._gateway.new_array(object_class, len(decision_trees))
            for i in range(len(decision_trees)):
                java_trees[i] = decision_trees[i]

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

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        _construct_rf, _ = super()._get_cuml_transform_func(dataset)

        def _predict(rf: CumlT, pdf: Union[cudf.DataFrame, np.ndarray]) -> pd.Series:
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

        return _construct_rf, _predict

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

    def predict(self, value: Vector) -> float:
        """
        Predict label for the given features.
        """
        return self.cpu().predict(value)

    def predictLeaf(self, value: Vector) -> float:
        """
        Predict the indices of the leaves corresponding to the feature vector.
        """
        return self.cpu().predictLeaf(value)

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

    @property
    def featureImportances(self) -> Vector:
        """
        Estimate of the importance of each feature.

        Each feature's importance is the average of its importance across all trees in the ensemble
        The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
        (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
        and follows the implementation from scikit-learn.

        See Also
        --------
        DecisionTreeClassificationModel.featureImportances
        """
        return self.cpu().featureImportances

    @property
    def treeWeights(self) -> List[float]:
        """Return the weights for each tree"""
        return self.cpu().treeWeights

    @property
    def trees(self) -> List[DecisionTreeClassificationModel]:  # type: ignore
        """Trees in this ensemble. Warning: These have null parent Estimators."""
        return self.cpu().trees

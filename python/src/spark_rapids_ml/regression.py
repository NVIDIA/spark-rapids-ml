# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
)

if TYPE_CHECKING:
    import cudf

import numpy as np
import pandas as pd
from pyspark import Row
from pyspark.ml.common import _py2java
from pyspark.ml.linalg import Vector, Vectors, _convert_to_vector
from pyspark.ml.regression import LinearRegressionModel as SparkLinearRegressionModel
from pyspark.ml.regression import LinearRegressionSummary
from pyspark.ml.regression import (
    RandomForestRegressionModel as SparkRandomForestRegressionModel,
)
from pyspark.ml.regression import _LinearRegressionParams, _RandomForestRegressorParams
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from .core import (
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModelSupervised,
    param_alias,
)
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .tree import (
    _RandomForestClass,
    _RandomForestCumlParams,
    _RandomForestEstimator,
    _RandomForestModel,
)
from .utils import PartitionDescriptor, _get_spark_session, cudf_to_cuml_array, java_uid

T = TypeVar("T")


class LinearRegressionClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "aggregationDepth": "",
            "elasticNetParam": "l1_ratio",
            "epsilon": "",
            "fitIntercept": "fit_intercept",
            "loss": "loss",
            "maxBlockSizeInMB": "",
            "maxIter": "max_iter",
            "regParam": "alpha",
            "solver": "solver",
            "standardization": "normalize",
            "tol": "tol",
            "weightCol": None,
        }

    @classmethod
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[str], Union[None, str, float, int]]]:
        return {
            "loss": lambda x: {
                "squaredError": "squared_loss",
                "huber": None,
                "squared_loss": "squared_loss",
            }.get(x, None),
            "solver": lambda x: {
                "auto": "eig",
                "normal": "eig",
                "l-bfgs": None,
                "eig": "eig",
            }.get(x, None),
        }

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "algorithm": "eig",
            "fit_intercept": True,
            "normalize": False,
            "verbose": False,
            "alpha": 0.0001,
            "solver": "eig",
            "loss": "squared_loss",
            "l1_ratio": 0.15,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
        }


class _LinearRegressionCumlParams(
    _CumlParams, _LinearRegressionParams, HasFeaturesCols
):
    """
    Shared Spark Params for LinearRegression and LinearRegressionModel.
    """

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

    def setFeaturesCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featureCols`.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self.set_params(featuresCols=value)

    def setLabelCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self.set_params(labelCol=value)

    def setPredictionCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self.set_params(predictionCol=value)


class LinearRegression(
    LinearRegressionClass,
    _CumlEstimatorSupervised,
    _LinearRegressionCumlParams,
):
    """LinearRegression is a machine learning model where the response y is modeled
    by a linear combination of the predictors in X. It implements cuML's GPU accelerated
    LinearRegression algorithm based on cuML python library, and it can be used in
    PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    This supports multiple types of regularization:

    * none (a.k.a. ordinary least squares)
    * L2 (ridge regression)
    * L1 (Lasso)
    * L2 + L1 (elastic net)

    LinearRegression automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.regression.LinearRegression` and in the constructors of
    :py:class:`cuml.LinearRegression`, :py:class:`cuml.Ridge`, :py:class:`cuml.Lasso`
    and :py:class:`cuml.ElasticNet`. And it will automatically map pyspark parameters
    to cuML parameters.

    Note
    ----
        Results for spark ML and spark rapids ml fit() will currently match in all regularization
        cases only if features and labels are standardized in the input dataframe.  Otherwise,
        they will match only if regParam = 0 or elastNetParam = 1.0 (aka Lasso).

    Examples
    --------
    >>> from spark_rapids_ml.regression import LinearRegression, LinearRegressionModel
    >>> from pyspark.ml.linalg import Vectors
    >>>
    >>> df = spark.createDataFrame([
    ...     (6.5, Vectors.dense(1.0, 2.0)),
    ...     (3.5, Vectors.sparse(2, {1: 2}))], ["label", "features"])
    >>>
    >>> lr = LinearRegression(regParam=0.0, solver="normal")
    >>> lr.setMaxIter(5)
    LinearRegression...
    >>> model = lr.fit(df)
    >>> model.setFeaturesCol("features")
    LinearRegressionModel...
    >>> model.setPredictionCol("newPrediction")
    LinearRegressionModel...
    >>> model.getMaxIter()
    5
    >>> model.coefficients
    [3.000000000000001, 0.0]
    >>> model.intercept
    3.4999999999999996
    >>> model.transform(df).show()
    +-----+----------+------------------+
    |label|  features|     newPrediction|
    +-----+----------+------------------+
    |  6.5|[1.0, 2.0]|               6.5|
    |  3.5|[0.0, 2.0]|3.4999999999999996|
    +-----+----------+------------------+

    >>> lr_path = temp_path + "/rl"
    >>> lr.save(lr_path)
    >>> lr2 = LinearRegression.load(lr_path)
    >>> lr2.getMaxIter()
    5
    >>> model_path = temp_path + "/lr_model"
    >>> model.save(model_path)
    >>> model2 = LinearRegressionModel.load(model_path)
    >>> model.coefficients[0] == model2.coefficients[0]
    True
    >>> model.intercept == model2.intercept
    True
    >>> model.numFeatures
    2
    >>> model2.transform(df).show()
    +-----+----------+------------------+
    |label|  features|     newPrediction|
    +-----+----------+------------------+
    |  6.5|[1.0, 2.0]|               6.5|
    |  3.5|[0.0, 2.0]|3.4999999999999996|
    +-----+----------+------------------+

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setMaxIter(self, value: int) -> "LinearRegression":
        """
        Sets the value of :py:attr:`maxIter`.
        """
        return self.set_params(maxIter=value)

    def setRegParam(self, value: float) -> "LinearRegression":
        """
        Sets the value of :py:attr:`regParam`.
        """
        return self.set_params(regParam=value)

    def setElasticNetParam(self, value: float) -> "LinearRegression":
        """
        Sets the value of :py:attr:`elasticNetParam`.
        """
        return self.set_params(elasticNetParam=value)

    def setLoss(self, value: str) -> "LinearRegression":
        """
        Sets the value of :py:attr:`loss`.
        """
        return self.set_params(loss=value)

    def setStandardization(self, value: bool) -> "LinearRegression":
        """
        Sets the value of :py:attr:`standardization`.
        """
        return self.set_params(standardization=value)

    def setTol(self, value: float) -> "LinearRegression":
        """
        Sets the value of :py:attr:`tol`.
        """
        return self.set_params(tol=value)

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

        # Ridge and LinearRegression can't train on the dataset which only has 1 feature
        if dimension == 1 and (
            self.cuml_params["alpha"] == 0 or self.cuml_params["l1_ratio"] == 0
        ):
            raise RuntimeError(
                "LinearRegression doesn't support training data with 1 column"
            )

        return select_cols, multi_col_names, dimension, feature_type

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        def _linear_regression_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            init_parameters = params[param_alias.cuml_init]

            pdesc = PartitionDescriptor.build(
                params[param_alias.part_sizes], params[param_alias.num_cols]
            )

            if init_parameters["alpha"] == 0:
                # LR
                from cuml.linear_model.linear_regression_mg import (
                    LinearRegressionMG as CumlLinearRegression,
                )

                supported_params = [
                    "algorithm",
                    "fit_intercept",
                    "normalize",
                    "verbose",
                ]
            else:
                if init_parameters["l1_ratio"] == 0:
                    # LR + L2
                    from cuml.linear_model.ridge_mg import (
                        RidgeMG as CumlLinearRegression,
                    )

                    supported_params = [
                        "alpha",
                        "solver",
                        "fit_intercept",
                        "normalize",
                        "verbose",
                    ]
                    # spark ML normalizes sample portion of objective by the number of examples
                    # but cuml does not for RidgeRegression (l1_ratio=0).   Induce similar behavior
                    # to spark ml by scaling up the reg parameter by the number of examples.
                    # With this, spark ML and spark rapids ML results match closely when features
                    # and label columns are all standardized.
                    init_parameters = init_parameters.copy()
                    if "alpha" in init_parameters.keys():
                        print(f"pdesc.m {pdesc.m}")
                        init_parameters["alpha"] *= (float)(pdesc.m)

                else:
                    # LR + L1, or LR + L1 + L2
                    # Cuml uses Coordinate Descent algorithm to implement Lasso and ElasticNet
                    # So combine Lasso and ElasticNet here.
                    from cuml.solvers.cd_mg import CDMG as CumlLinearRegression

                    # in this case, both spark ML and cuml CD normalize sample portion of
                    # objective by the number of training examples, so no need to adjust
                    # reg params

                    supported_params = [
                        "loss",
                        "alpha",
                        "l1_ratio",
                        "fit_intercept",
                        "max_iter",
                        "normalize",
                        "tol",
                        "shuffle",
                        "verbose",
                    ]

            # filter only supported params
            init_parameters = {
                k: v for k, v in init_parameters.items() if k in supported_params
            }

            linear_regression = CumlLinearRegression(
                handle=params[param_alias.handle],
                output_type="cudf",
                **init_parameters,
            )

            linear_regression.fit(
                dfs,
                pdesc.m,
                pdesc.n,
                pdesc.parts_rank_size,
                pdesc.rank,
            )

            return {
                "coef_": [linear_regression.coef_.to_numpy().tolist()],
                "intercept_": linear_regression.intercept_,
                "dtype": linear_regression.dtype.name,
                "n_cols": linear_regression.n_cols,
            }

        return _linear_regression_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("coef_", ArrayType(DoubleType(), False), False),
                StructField("intercept_", DoubleType(), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "LinearRegressionModel":
        return LinearRegressionModel.from_row(result)


class LinearRegressionModel(
    LinearRegressionClass,
    _CumlModelSupervised,
    _LinearRegressionCumlParams,
):
    """Model fitted by :class:`LinearRegression`."""

    def __init__(
        self,
        coef_: List[float],
        intercept_: float,
        n_cols: int,
        dtype: str,
    ) -> None:
        super().__init__(dtype=dtype, n_cols=n_cols, coef_=coef_, intercept_=intercept_)
        self.coef_ = coef_
        self.intercept_ = intercept_
        self._lr_ml_model: Optional[SparkLinearRegressionModel] = None

    def cpu(self) -> SparkLinearRegressionModel:
        """Return the PySpark ML LinearRegressionModel"""
        if self._lr_ml_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            coef = _convert_to_vector(self.coefficients)

            java_model = sc._jvm.org.apache.spark.ml.regression.LinearRegressionModel(
                java_uid(sc, "linReg"), _py2java(sc, coef), self.intercept, self.scale
            )
            self._lr_ml_model = SparkLinearRegressionModel(java_model)
            self._copyValues(self._lr_ml_model)

        return self._lr_ml_model

    @property
    def coefficients(self) -> Vector:
        """
        Model coefficients.
        """
        # TBD: for large enough dimension, SparseVector is returned. Need to find out how to match
        return Vectors.dense(self.coef_)

    @property
    def hasSummary(self) -> bool:
        """
        Indicates whether a training summary exists for this model instance.
        """
        return False

    @property
    def intercept(self) -> float:
        """
        Model intercept.
        """
        return self.intercept_

    @property
    def scale(self) -> float:
        """
        Since "huber" loss is not supported by cuML, just returns the value 1.0 for API compatibility.
        """
        return 1.0

    def predict(self, value: T) -> float:
        """cuML doesn't support predicting 1 single sample.
        Fall back to PySpark ML LinearRegressionModel"""
        return self.cpu().predict(value)

    def evaluate(self, dataset: DataFrame) -> LinearRegressionSummary:
        """cuML doesn't support evaluating.
        Fall back to PySpark ML LinearRegressionModel"""
        return self.cpu().evaluate(dataset)

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union["cudf.DataFrame", np.ndarray]], pd.DataFrame],
    ]:
        coef_ = self.coef_
        intercept_ = self.intercept_
        n_cols = self.n_cols
        dtype = self.dtype

        def _construct_lr() -> CumlT:
            from cuml.linear_model.linear_regression_mg import LinearRegressionMG

            lr = LinearRegressionMG(output_type="numpy")
            lr.coef_ = cudf_to_cuml_array(np.array(coef_, order="F").astype(dtype))
            lr.intercept_ = intercept_
            lr.n_cols = n_cols
            lr.dtype = np.dtype(dtype)

            return lr

        def _predict(lr: CumlT, pdf: Union["cudf.DataFrame", np.ndarray]) -> pd.Series:
            ret = lr.predict(pdf)
            return pd.Series(ret)

        return _construct_lr, _predict


class _RandomForestRegressorClass(_RandomForestClass):
    @classmethod
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[str], Union[None, str, float, int]]]:
        mapping = super()._param_value_mapping()
        mapping["split_criterion"] = lambda x: {"variance": "mse", "mse": "mse"}.get(
            x, None
        )
        return mapping


class RandomForestRegressor(
    _RandomForestRegressorClass,
    _RandomForestEstimator,
    _RandomForestCumlParams,
    _RandomForestRegressorParams,
):
    """RandomForestRegressor implements a Random Forest regressor model which
    fits multiple decision tree in an ensemble. It implements cuML's
    GPU accelerated RandomForestRegressor algorithm based on cuML python library,\
    and it can be used in PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`/
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`/
    :py:class:`~pyspark.ml.classification.OneVsRest`

    The distributed algorithm uses an *embarrassingly-parallel* approach. For a
    forest with `N` trees being built on `w` workers, each worker simply builds `N/w`
    trees on the data it has available locally. In many cases, partitioning the
    data so that each worker builds trees on a subset of the total dataset works
    well, but it generally requires the data to be well-shuffled in advance.

    RandomForestRegressor automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.regression.RandomForestRegressor` and in the constructors of
    :py:class:`cuml.ensemble.RandomForestRegressor`. And it can automatically map
    pyspark parameters to cuML parameters.

    Examples
    --------
    >>> from spark_rapids_ml.regression import RandomForestRegressor, RandomForestRegressionModel
    >>> from numpy import allclose
    >>> from pyspark.ml.linalg import Vectors
    >>> df = spark.createDataFrame([
    ...     (1.0, Vectors.dense(1.0)),
    ...     (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    >>> rf = RandomForestRegressor(numTrees=2, maxDepth=2)
    >>> rf.setSeed(42)
    RandomForestRegressor_...
    >>> model = rf.fit(df)
    >>> model.getBootstrap()
    True
    >>> model.getSeed()
    42
    >>> test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    >>> result = model.transform(test0).head()
    >>> result.prediction
    0.0
    >>> model.numFeatures
    1
    >>> model.getNumTrees
    2
    >>> test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    >>> model.transform(test1).head().prediction
    1.0
    >>> rfr_path = temp_path + "/rfr"
    >>> rf.save(rfr_path)
    >>> rf2 = RandomForestRegressor.load(rfr_path)
    >>> rf2.getNumTrees()
    2
    >>> model_path = temp_path + "/rfr_model"
    >>> model.save(model_path)
    >>> model2 = RandomForestRegressionModel.load(model_path)
    >>> model.transform(test0).take(1) == model2.transform(test0).take(1)
    True

    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.set_params(**kwargs)

    def _is_classification(self) -> bool:
        return False

    def _create_pyspark_model(self, result: Row) -> "RandomForestRegressionModel":
        return RandomForestRegressionModel.from_row(result)


class RandomForestRegressionModel(
    _RandomForestRegressorClass,
    _RandomForestModel,
    _RandomForestCumlParams,
    _RandomForestRegressorParams,
):
    """
    Model fitted by :class:`RandomForestRegressor`.
    """

    def __init__(
        self,
        n_cols: int,
        dtype: str,
        treelite_model: str,
        model_json: List[str],
    ):
        super().__init__(
            dtype=dtype,
            n_cols=n_cols,
            treelite_model=treelite_model,
            model_json=model_json,
        )

        self._rf_spark_model: Optional[SparkRandomForestRegressionModel] = None

    def cpu(self) -> SparkRandomForestRegressionModel:
        """Return the PySpark ML RandomForestRegressionModel"""

        if self._rf_spark_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            uid, java_trees = self._convert_to_java_trees(self.getImpurity())
            # Create the Spark RandomForestClassificationModel
            java_rf_model = (
                sc._jvm.org.apache.spark.ml.regression.RandomForestRegressionModel(
                    uid,
                    java_trees,
                    self.numFeatures,
                )
            )
            self._rf_spark_model = SparkRandomForestRegressionModel(java_rf_model)
            self._copyValues(self._rf_spark_model)
        return self._rf_spark_model

    def _is_classification(self) -> bool:
        return False

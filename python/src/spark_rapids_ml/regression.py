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
    cast,
)

import numpy as np
import pandas as pd
from pyspark import Row, TaskContext, keyword_only
from pyspark.ml.common import _py2java
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator
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
    CumlT,
    FitInputType,
    TransformInputType,
    _ConstructFunc,
    _CumlEstimatorSupervised,
    _CumlModel,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
    pred,
    transform_evaluate,
)
from .metrics.RegressionMetrics import RegressionMetrics, reg_metrics
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .tree import (
    _RandomForestClass,
    _RandomForestCumlParams,
    _RandomForestEstimator,
    _RandomForestModel,
)
from .utils import PartitionDescriptor, _get_spark_session, cudf_to_cuml_array, java_uid

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap

T = TypeVar("T")


class _RegressionModelEvaluationMixIn:
    # https://github.com/python/mypy/issues/5868#issuecomment-437690894 to bypass mypy checking
    _this_model: Union["RandomForestRegressionModel", "LinearRegressionModel"]

    def _transform_evaluate(
        self,
        dataset: DataFrame,
        evaluator: Evaluator,
        num_models: int,
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
        num_models: how many models are used to perform transform and evaluation in a single pass
        params : dict, optional
            an optional param map that overrides embedded params

        Returns
        -------
        list of float
            metrics
        """

        if not isinstance(evaluator, RegressionEvaluator):
            raise NotImplementedError(f"{evaluator} is unsupported yet.")

        if self._this_model.getLabelCol() not in dataset.schema.names:
            raise RuntimeError("Label column is not existing.")

        dataset = dataset.withColumnRenamed(self._this_model.getLabelCol(), alias.label)

        schema = StructType(
            [
                StructField(pred.model_index, IntegerType()),
                StructField(reg_metrics.mean, ArrayType(FloatType())),
                StructField(reg_metrics.m2n, ArrayType(FloatType())),
                StructField(reg_metrics.m2, ArrayType(FloatType())),
                StructField(reg_metrics.l1, ArrayType(FloatType())),
                StructField(reg_metrics.total_count, IntegerType()),
            ]
        )

        rows = self._this_model._transform_evaluate_internal(dataset, schema).collect()

        metrics = RegressionMetrics.from_rows(num_models, rows)
        return [metric.evaluate(evaluator) for metric in metrics]

    @staticmethod
    def calculate_regression_metrics(
        input: TransformInputType,
        transformed: TransformInputType,
    ) -> pd.DataFrame:
        """calculate the metrics: mean/m2n/m2/l1 ...

        input must have `alias.label` column"""

        comb = pd.DataFrame(
            {
                "label": input[alias.label],
                "prediction": transformed,
            }
        )
        comb.insert(1, "label-prediction", comb["label"] - comb["prediction"])
        total_cnt = comb.shape[0]
        return pd.DataFrame(
            data={
                reg_metrics.mean: [comb.mean().to_list()],
                reg_metrics.m2n: [(comb.var(ddof=0) * total_cnt).to_list()],
                reg_metrics.m2: [comb.pow(2).sum().to_list()],
                reg_metrics.l1: [comb.abs().sum().to_list()],
                reg_metrics.total_count: total_cnt,
            }
        )


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
    :py:class:`~pyspark.ml.regression.LinearRegression`,
    :py:class:`cuml.LinearRegression`, :py:class:`cuml.Ridge`, :py:class:`cuml.Lasso`
    and :py:class:`cuml.ElasticNet`. And it will automatically map pyspark parameters
    to cuML parameters.

    Notes
    -----
        Results for spark ML and spark rapids ml fit() will currently match in all regularization
        cases only if features and labels are standardized in the input dataframe.  Otherwise,
        they will match only if regParam = 0 or elastNetParam = 1.0 (aka Lasso).

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
    maxIter:
        Max number of iterations (>= 0).
    regParam:
        Regularization parameter (>= 0)
    elasticNetParam:
        The ElasticNet mixing parameter, in range [0, 1]. For alpha = 0,
        the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
    tol:
        The convergence tolerance for iterative algorithms (>= 0).
    fitIntercept:
        whether to fit an intercept term.
    standardization:
        Whether to standardize the training features before fitting the model.
    solver:
        The solver algorithm for optimization. If this is not set or empty, default value is 'auto'.\n
        The supported options: 'auto', 'normal' and 'eig', all of them will be mapped to 'eig' in cuML.
    loss:
        The loss function to be optimized.
        The supported options: 'squaredError'
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

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        maxIter: int = 100,
        regParam: float = 0.0,
        elasticNetParam: float = 0.0,
        tol: float = 1e-6,
        fitIntercept: bool = True,
        standardization: bool = True,
        solver: str = "auto",
        loss: str = "squaredError",
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.set_params(**self._input_kwargs)

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
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
        def _linear_regression_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            # Step 1, get the PartitionDescriptor
            pdesc = PartitionDescriptor.build(
                params[param_alias.part_sizes], params[param_alias.num_cols]
            )

            def _single_fit(init_parameters: Dict[str, Any]) -> Dict[str, Any]:
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
                final_init_parameters = {
                    k: v for k, v in init_parameters.items() if k in supported_params
                }

                linear_regression = CumlLinearRegression(
                    handle=params[param_alias.handle],
                    output_type="cudf",
                    **final_init_parameters,
                )

                linear_regression.fit(
                    dfs,
                    pdesc.m,
                    pdesc.n,
                    pdesc.parts_rank_size,
                    pdesc.rank,
                )

                return {
                    "coef_": linear_regression.coef_.to_numpy().tolist(),
                    "intercept_": linear_regression.intercept_,
                    "dtype": linear_regression.dtype.name,
                    "n_cols": linear_regression.n_cols,
                }

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

    def _enable_fit_multiple_in_single_pass(self) -> bool:
        return True

    def _supportsTransformEvaluate(self, evaluator: Evaluator) -> bool:
        return True if isinstance(evaluator, RegressionEvaluator) else False


class LinearRegressionModel(
    LinearRegressionClass,
    _CumlModelWithPredictionCol,
    _LinearRegressionCumlParams,
    _RegressionModelEvaluationMixIn,
):
    """Model fitted by :class:`LinearRegression`."""

    def __init__(
        self,
        coef_: Union[List[float], List[List[float]]],
        intercept_: Union[float, List[float]],
        n_cols: int,
        dtype: str,
    ) -> None:
        super().__init__(dtype=dtype, n_cols=n_cols, coef_=coef_, intercept_=intercept_)
        self.coef_ = coef_
        self.intercept_ = intercept_
        self._lr_ml_model: Optional[SparkLinearRegressionModel] = None
        self._this_model = self

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
        assert not isinstance(self.coef_[0], list)
        return Vectors.dense(cast(list, self.coef_))

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
        assert not isinstance(self.intercept_, list)
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
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        coef_ = self.coef_
        intercept_ = self.intercept_
        n_cols = self.n_cols
        dtype = self.dtype

        def _construct_lr() -> CumlT:
            from cuml.linear_model.linear_regression_mg import LinearRegressionMG

            lrs = []

            coefs = coef_ if isinstance(intercept_, list) else [coef_]
            intercepts = intercept_ if isinstance(intercept_, list) else [intercept_]

            for i in range(len(coefs)):
                lr = LinearRegressionMG(output_type="numpy")
                lr.coef_ = cudf_to_cuml_array(
                    np.array(coefs[i], order="F").astype(dtype)
                )
                lr.intercept_ = intercepts[i]
                lr.n_cols = n_cols
                lr.dtype = np.dtype(dtype)
                lrs.append(lr)

            return lrs

        def _predict(lr: CumlT, pdf: TransformInputType) -> pd.Series:
            ret = lr.predict(pdf)
            return pd.Series(ret)

        return _construct_lr, _predict, self.calculate_regression_metrics

    def _transform(self, dataset: DataFrame) -> DataFrame:
        df = super()._transform(dataset)
        return df.withColumn(
            self.getPredictionCol(), df[self.getPredictionCol()].cast("double")
        )

    @classmethod
    def _combine(
        cls: Type["LinearRegressionModel"], models: List["LinearRegressionModel"]  # type: ignore
    ) -> "LinearRegressionModel":
        assert len(models) > 0 and all(isinstance(model, cls) for model in models)
        first_model = models[0]

        # Combine coef and intercepts
        coefs = cast(list, [model.coef_ for model in models])
        intercepts = cast(list, [model.intercept_ for model in models])

        assert first_model.n_cols is not None
        assert first_model.dtype is not None

        lr_model = cls(
            n_cols=first_model.n_cols,
            dtype=first_model.dtype,
            coef_=coefs,
            intercept_=intercepts,
        )
        first_model._copyValues(lr_model)
        first_model._copy_cuml_params(lr_model)

        return lr_model

    def _transformEvaluate(
        self,
        dataset: DataFrame,
        evaluator: Evaluator,
        params: Optional["ParamMap"] = None,
    ) -> List[float]:
        num_models = len(self.intercept_) if isinstance(self.intercept_, list) else 1
        return self._transform_evaluate(
            dataset=dataset, evaluator=evaluator, num_models=num_models, params=params
        )


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
    GPU accelerated RandomForestRegressor algorithm based on cuML python library,
    and it can be used in PySpark Pipeline and PySpark ML meta algorithms like
    :py:class:`~pyspark.ml.tuning.CrossValidator`,
    :py:class:`~pyspark.ml.tuning.TrainValidationSplit`,
    :py:class:`~pyspark.ml.classification.OneVsRest`

    The distributed algorithm uses an *embarrassingly-parallel* approach. For a
    forest with `N` trees being built on `w` workers, each worker simply builds `N/w`
    trees on the data it has available locally. In many cases, partitioning the
    data so that each worker builds trees on a subset of the total dataset works
    well, but it generally requires the data to be well-shuffled in advance.

    RandomForestRegressor automatically supports most of the parameters from both
    :py:class:`~pyspark.ml.regression.RandomForestRegressor` and
    :py:class:`cuml.ensemble.RandomForestRegressor`. And it can automatically map
    pyspark parameters to cuML parameters.


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
    maxDepth:
        Maximum tree depth. Must be greater than 0.
    maxBins:
        Maximum number of bins used by the split algorithm per feature.
    minInstancesPerNode:
        The minimum number of samples (rows) in each leaf node.
    impurity: str = "variance",
        The criterion used to split nodes.
    numTrees:
        Total number of trees in the forest.
    featureSubsetStrategy:
        Ratio of number of features (columns) to consider per node split.\n
        The supported options:\n
            ``'auto'``:  If numTrees == 1, set to 'all', If numTrees > 1 (forest), set to 'onethird'\n
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

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        labelCol: str = "label",
        predictionCol: str = "prediction",
        maxDepth: int = 5,
        maxBins: int = 32,
        minInstancesPerNode: int = 1,
        impurity: str = "variance",
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

    def _is_classification(self) -> bool:
        return False

    def _create_pyspark_model(self, result: Row) -> "RandomForestRegressionModel":
        return RandomForestRegressionModel.from_row(result)

    def _supportsTransformEvaluate(self, evaluator: Evaluator) -> bool:
        return True if isinstance(evaluator, RegressionEvaluator) else False


class RandomForestRegressionModel(
    _RandomForestRegressorClass,
    _RandomForestModel,
    _RandomForestCumlParams,
    _RandomForestRegressorParams,
    _RegressionModelEvaluationMixIn,
):
    """
    Model fitted by :class:`RandomForestRegressor`.
    """

    def __init__(
        self,
        n_cols: int,
        dtype: str,
        treelite_model: Union[str, List[str]],
        model_json: Union[List[str], List[List[str]]] = [],  # type: ignore
    ):
        super().__init__(
            dtype=dtype,
            n_cols=n_cols,
            treelite_model=treelite_model,
            model_json=model_json,
        )

        self._rf_spark_model: Optional[SparkRandomForestRegressionModel] = None
        self._this_model = self

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

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        _construct_rf, _predict, _ = super()._get_cuml_transform_func(dataset, category)
        return _construct_rf, _predict, self.calculate_regression_metrics

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

        num_models = (
            len(self._treelite_model) if isinstance(self._treelite_model, list) else 1
        )
        return self._transform_evaluate(
            dataset=dataset, evaluator=evaluator, num_models=num_models, params=params
        )

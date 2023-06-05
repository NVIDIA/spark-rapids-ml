#
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
from typing import Any, Dict, Tuple, Type, TypeVar

import numpy as np
import pytest
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression as SparkLinearRegression
from pyspark.ml.regression import LinearRegressionModel as SparkLinearRegressionModel
from pyspark.sql.functions import array, col
from pyspark.sql.types import DoubleType

from spark_rapids_ml.regression import LinearRegression, LinearRegressionModel

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types,
    get_default_cuml_parameters,
    idfn,
    make_regression_dataset,
    pyspark_supported_feature_types,
)

LinearRegressionType = TypeVar(
    "LinearRegressionType", Type[LinearRegression], Type[SparkLinearRegression]
)
LinearRegressionModelType = TypeVar(
    "LinearRegressionModelType",
    Type[LinearRegressionModel],
    Type[SparkLinearRegressionModel],
)


# @lru_cache(4) TODO fixme: TypeError: Unhashable Type” Numpy.Ndarray
def train_with_cuml_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    other_params: Dict[str, Any] = {},
) -> Any:
    if alpha == 0:
        from cuml import LinearRegression as cuLinearRegression

        lr = cuLinearRegression(output_type="numpy")
    else:
        if l1_ratio == 0.0:
            from cuml import Ridge

            lr = Ridge(output_type="numpy", alpha=alpha * len(y))
        elif l1_ratio == 1.0:
            from cuml import Lasso

            lr = Lasso(output_type="numpy", alpha=alpha)
        else:
            from cuml import ElasticNet

            lr = ElasticNet(
                output_type="numpy", alpha=alpha, l1_ratio=l1_ratio, **other_params
            )

    lr.fit(X, y)
    return lr


def test_default_cuml_params() -> None:
    from cuml.linear_model.linear_regression import (
        LinearRegression as CumlLinearRegression,
    )
    from cuml.linear_model.ridge import Ridge
    from cuml.solvers import CD

    cuml_params = get_default_cuml_parameters(
        [CumlLinearRegression, Ridge, CD], ["handle", "output_type"]
    )
    spark_params = LinearRegression()._get_cuml_params_default()
    assert cuml_params == spark_params


@pytest.mark.parametrize("reg", [0.0, 0.7])
def test_linear_regression_params(tmp_path: str, reg: float) -> None:
    # Default params
    default_spark_params = {
        "elasticNetParam": 0.0,
        "fitIntercept": True,
        "loss": "squaredError",
        "maxIter": 100,
        "regParam": 0.0,
        "solver": "auto",
        "standardization": True,
        "tol": 1e-06,
    }
    default_cuml_params = {
        "algorithm": "eig",
        "alpha": 0.0,
        "fit_intercept": True,
        "l1_ratio": 0.0,
        "max_iter": 100,
        "normalize": True,
        "solver": "eig",
    }
    default_lr = LinearRegression()
    assert_params(default_lr, default_spark_params, default_cuml_params)

    # Spark ML Params
    spark_params: Dict[str, Any] = {
        "fitIntercept": False,
        "standardization": False,
        "regParam": reg,
        "solver": "normal",
    }
    spark_lr = LinearRegression(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(
        {
            "alpha": reg,
            "fit_intercept": False,
            "normalize": False,
            "solver": "eig",
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/linear_regression_tests"
    estimator_path = f"{path}/linear_regression"
    spark_lr.write().overwrite().save(estimator_path)
    loaded_lr = LinearRegression.load(estimator_path)
    assert_params(loaded_lr, expected_spark_params, expected_cuml_params)

    # Unsupported value
    spark_params = {"solver": "l-bfgs"}
    with pytest.raises(
        ValueError, match="Value 'l-bfgs' for 'solver' param is unsupported"
    ):
        unsupported_lr = LinearRegression(**spark_params)


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_linear_regression_numeric_type(gpu_number: int, data_type: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [
        [1, 4, 4, 4, 0],
        [2, 2, 2, 2, 1],
        [3, 3, 3, 2, 2],
        [3, 3, 3, 2, 3],
        [5, 2, 1, 3, 4],
    ]

    with CleanSparkSession() as spark:
        feature_cols = ["c1", "c2", "c3", "c4"]
        schema = (
            ", ".join([f"{c} {data_type}" for c in feature_cols])
            + f", label {data_type}"
        )
        df = spark.createDataFrame(data, schema=schema)
        lr = LinearRegression(num_workers=gpu_number)
        lr.setFeaturesCol(feature_cols)
        lr.fit(df)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("data_shape", [(10, 2)], ids=idfn)
@pytest.mark.parametrize("reg", [0.0, 0.7])
def test_linear_regression_basic(
    gpu_number: int,
    tmp_path: str,
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
    reg: float,
) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

    # Train a toy model
    X, _, y, _ = make_regression_dataset(data_type, data_shape[0], data_shape[1])
    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )

        lr = LinearRegression(num_workers=gpu_number)
        lr.setRegParam(reg)

        lr.setFeaturesCol(features_col)
        assert lr.getFeaturesCol() == features_col

        assert label_col is not None
        lr.setLabelCol(label_col)
        assert lr.getLabelCol() == label_col

        def assert_cuml_pyspark_model(
            lhs: LinearRegressionModel, rhs: SparkLinearRegressionModel
        ) -> None:
            assert lhs.coefficients == rhs.coefficients
            assert lhs.intercept == rhs.intercept
            assert lhs.getRegParam() == rhs.getRegParam()
            assert lhs.getRegParam() == reg

        def assert_cuml_model(
            lhs: LinearRegressionModel, rhs: LinearRegressionModel
        ) -> None:
            assert lhs.coef_ == rhs.coef_
            assert lhs.intercept_ == rhs.intercept_
            assert lhs.coefficients == rhs.coefficients
            assert lhs.intercept == rhs.intercept

            # Vector type will be cast to array(double)
            if feature_type == "vector":
                assert lhs.dtype == np.dtype(np.float64).name
            else:
                assert lhs.dtype == np.dtype(data_type).name

            assert lhs.dtype == rhs.dtype
            assert lhs.n_cols == rhs.n_cols
            assert lhs.n_cols == data_shape[1]

        # train a model
        lr_model = lr.fit(df)
        assert (
            lr_model.transform(df).schema[lr.getPredictionCol()].dataType
            == DoubleType()
        )

        assert isinstance(lr_model.cpu(), SparkLinearRegressionModel)
        assert_cuml_pyspark_model(lr_model, lr_model.cpu())

        # Convert input to vector dataframe to fit in the Spark LinearRegressionModel
        if feature_type == feature_types.array:
            vector_df = df.select(array_to_vector(col(features_col)).alias("features"))  # type: ignore
        elif feature_type == feature_types.multi_cols:
            assembler = (
                VectorAssembler().setInputCols(features_col).setOutputCol("features")  # type: ignore
            )
            vector_df = assembler.transform(df).drop(*features_col)
        else:
            vector_df = df

        # transform without throwing exception
        lr_model.cpu().transform(vector_df).collect()

        # model persistence
        path = tmp_path + "/linear_regression_tests"
        model_path = f"{path}/linear_regression_model"
        lr_model.write().overwrite().save(model_path)

        lr_model_loaded = LinearRegressionModel.load(model_path)
        assert isinstance(lr_model_loaded.cpu(), SparkLinearRegressionModel)
        assert_cuml_pyspark_model(lr_model_loaded, lr_model_loaded.cpu())

        assert_cuml_model(lr_model, lr_model_loaded)

        # transform without throwing exception
        lr_model_loaded.cpu().transform(vector_df).collect()


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize("alpha", [0.0, 0.7])  # equal to reg parameter
@pytest.mark.parametrize(
    "l1_ratio_and_other_params",
    [
        (0.0, {}),  # LinearRegression
        (0.5, {"tol": 1e-5}),  # ElasticNet
        (1.0, {"tol": 1e-5}),  # Lasso
    ],
)
@pytest.mark.slow
def test_linear_regression(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    alpha: float,
    l1_ratio_and_other_params: Tuple[float, Dict[str, Any]],
) -> None:
    X_train, X_test, y_train, _ = make_regression_dataset(
        data_type, data_shape[0], data_shape[1]
    )

    l1_ratio, other_params = l1_ratio_and_other_params
    cu_lr = train_with_cuml_linear_regression(
        X_train, y_train, alpha, l1_ratio, other_params
    )
    cu_expected = cu_lr.predict(X_test)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )
        assert label_col is not None

        slr = LinearRegression(num_workers=gpu_number, verbose=7, **other_params)
        slr.setRegParam(alpha)
        slr.setStandardization(
            False
        )  # Spark default is True, but Cuml default is False
        slr.setElasticNetParam(l1_ratio)
        slr.setFeaturesCol(features_col)
        slr.setLabelCol(label_col)
        slr_model: LinearRegressionModel = slr.fit(train_df)

        assert slr_model.cpu().getElasticNetParam() == l1_ratio
        assert slr_model.cpu().getRegParam() == alpha
        assert not slr_model.cpu().getStandardization()
        assert slr_model.cpu().getLabelCol() == label_col

        assert array_equal(cu_lr.coef_, slr_model.coef_, 1e-3)
        assert array_equal(cu_lr.coef_, slr_model.coefficients.toArray(), 1e-3)

        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = slr_model.transform(test_df).collect()
        pred_result = [row.prediction for row in result]
        assert array_equal(cu_expected, pred_result, 1e-3)


params_exception = [
    # params, if throwing exception
    ({"alpha": 0}, True),  # LinearRegression throws exception
    ({"alpha": 0.5, "l1_ratio": 0}, True),  # Ridge throws exception
    ({"alpha": 0.5, "l1_ratio": 0.5}, False),  # ElasticNet and Lasso can work
]


@pytest.mark.compat
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLinearRegression, SparkLinearRegressionModel),
        (LinearRegression, LinearRegressionModel),
    ],
)
def test_linear_regression_spark_compat(
    lr_types: Tuple[LinearRegressionType, LinearRegressionModelType],
    tmp_path: str,
) -> None:
    _LinearRegression, _LinearRegressionModel = lr_types

    X = np.array(
        [
            [-0.20515826, 1.4940791],
            [0.12167501, 0.7610377],
            [1.4542735, 0.14404356],
            [-0.85409576, 0.3130677],
            [2.2408931, 0.978738],
            [-0.1513572, 0.95008844],
            [-0.9772779, 1.867558],
            [0.41059852, -0.10321885],
        ]
    )
    weight = np.ones([8])
    y = np.array(
        [
            2.0374513,
            22.403986,
            139.4456,
            -76.19584,
            225.72075,
            -0.6784152,
            -65.54835,
            37.30829,
        ]
    )

    feature_cols = ["c0", "c1"]
    schema = ["c0 float, c1 float, weight float, label float"]

    with CleanSparkSession() as spark:
        df = spark.createDataFrame(
            np.concatenate((X, weight.reshape(8, 1), y.reshape(8, 1)), axis=1).tolist(),
            ",".join(schema),
        )
        df = df.withColumn("features", array_to_vector(array(*feature_cols))).drop(
            *feature_cols
        )

        lr = _LinearRegression(regParam=0.1, solver="normal")
        assert lr.getRegParam() == 0.1

        lr.setFeaturesCol("features")
        lr.setMaxIter(5)
        lr.setRegParam(0.0)
        lr.setLabelCol("label")
        if isinstance(lr, SparkLinearRegression):
            lr.setWeightCol("weight")

        assert lr.getFeaturesCol() == "features"
        assert lr.getMaxIter() == 5
        assert lr.getRegParam() == 0.0
        assert lr.getLabelCol() == "label"

        model = lr.fit(df)
        coefficients = model.coefficients.toArray()
        expected_coefficients = [94.46689350900762, 14.33532962562045]
        assert array_equal(coefficients, expected_coefficients)

        intercept = model.intercept
        assert np.isclose(intercept, -3.3089753423400734e-07)

        example = df.head()
        if example:
            model.predict(example.features)

        model.setPredictionCol("prediction")
        output = model.transform(df).head()
        # Row(weight=1.0, label=2.0374512672424316, features=DenseVector([-0.2052, 1.4941]), prediction=2.037452415464224)
        assert np.isclose(output.prediction, 2.037452415464224)

        lr_path = tmp_path + "/lr"
        lr.save(lr_path)

        lr2 = _LinearRegression.load(lr_path)
        assert lr2.getMaxIter() == 5

        model_path = tmp_path + "/lr_model"
        model.save(model_path)

        model2 = _LinearRegressionModel.load(model_path)
        assert model.coefficients.toArray()[0] == model2.coefficients.toArray()[0]
        assert model.intercept == model2.intercept
        assert model.transform(df).take(1) == model2.transform(df).take(1)
        assert model.numFeatures == 2


@pytest.mark.parametrize("params_exception", params_exception)
def test_fail_run_on_1_col(
    gpu_number: int, params_exception: Tuple[Dict[str, Any], bool]
) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

    params, exception = params_exception
    with CleanSparkSession() as spark:
        df = spark.createDataFrame(
            [
                (1.0, Vectors.dense(1.0)),
                (0.0, Vectors.sparse(1, [], [])),
                (1.0, Vectors.dense(1.0)),
                (0.0, Vectors.sparse(1, [], [])),
            ],
            ["label", "features"],
        )
        lr = LinearRegression(num_workers=gpu_number, **params)

        if exception:
            with pytest.raises(
                RuntimeError,
                match="LinearRegression doesn't support training data with 1 column",
            ):
                lr.fit(df)
        else:
            lr.fit(df)

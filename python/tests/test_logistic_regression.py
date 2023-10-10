from typing import Any, Dict, List, Tuple, Type, TypeVar

import cuml
import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from packaging import version
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import (
    LogisticRegressionModel as SparkLogisticRegressionModel,
)
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.param import Param
from pyspark.sql import Row
from pyspark.sql.functions import array, col

if version.parse(cuml.__version__) < version.parse("23.08.00"):
    raise ValueError(
        "Logistic Regression requires cuml 23.08.00 or above. Try upgrading cuml or ignoring this file in testing"
    )

import warnings

from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    feature_types,
    idfn,
    make_classification_dataset,
)


def test_toy_example(gpu_number: int, caplog: LogCaptureFixture) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [
        ([1.0, 2.0], 1.0),
        ([1.0, 3.0], 1.0),
        ([2.0, 1.0], 0.0),
        ([3.0, 1.0], 0.0),
    ]

    with CleanSparkSession() as spark:
        features_col = "features"
        label_col = "label"
        probability_col = "probs"
        schema = features_col + " array<float>, " + label_col + " float"
        df = spark.createDataFrame(data, schema=schema)

        lr_estimator = LogisticRegression(regParam=1.0, num_workers=gpu_number)
        lr_estimator.setFeaturesCol(features_col)
        lr_estimator.setLabelCol(label_col)
        lr_estimator.setProbabilityCol(probability_col)
        lr_model = lr_estimator.fit(df)

        assert lr_model.n_cols == 2
        assert lr_model.dtype == "float32"

        assert len(lr_model.coef_) == 1
        assert lr_model.coef_[0] == pytest.approx([-0.287264, 0.287264], abs=1e-6)
        assert lr_model.intercept_ == pytest.approx([0], abs=1e-6)

        assert lr_model.coefficients.toArray() == pytest.approx(
            [-0.287264, 0.287264], abs=1e-6
        )
        assert lr_model.intercept == pytest.approx(0, abs=1e-6)

        def assert_transform(model: LogisticRegressionModel) -> None:
            preds_df_local = model.transform(df).collect()
            preds = [row["prediction"] for row in preds_df_local]
            assert preds == [1.0, 1.0, 0.0, 0.0]
            probs = [row["probs"] for row in preds_df_local]
            assert len(probs) == len(preds)
            assert [p[1] > 0.5 for p in probs] == [True, True, False, False]

        assert_transform(lr_model)

        # test with regParam set to 0
        caplog.clear()
        lr_regParam_zero = LogisticRegression(
            regParam=0.0,
        )
        assert "no regularization is not supported yet" in caplog.text

        lr_regParam_zero.setProbabilityCol(probability_col)

        assert lr_regParam_zero.getRegParam() == 0
        assert (
            lr_regParam_zero.cuml_params["C"] == 1.0 / np.finfo("float32").tiny.item()
        )
        model_regParam_zero = lr_regParam_zero.fit(df)
        assert_transform(model_regParam_zero)

        lr_regParam_zero.setRegParam(0.1)
        assert lr_regParam_zero.getRegParam() == 0.1
        assert lr_regParam_zero.cuml_params["C"] == 1.0 / 0.1

        caplog.clear()
        lr_regParam_zero.setRegParam(0.0)
        assert "no regularization is not supported yet" in caplog.text

        assert lr_regParam_zero.getRegParam() == 0.0
        assert (
            lr_regParam_zero.cuml_params["C"] == 1.0 / np.finfo("float32").tiny.item()
        )


def test_params(tmp_path: str, caplog: LogCaptureFixture) -> None:
    # Default params
    default_spark_params = {
        "maxIter": 100,
        "regParam": 0.0,  # will be mapped to numpy.finfo('float32').tiny
        "tol": 1e-06,
        "fitIntercept": True,
    }

    default_cuml_params = {
        "max_iter": 100,
        "C": 1.0
        / np.finfo(
            "float32"
        ).tiny,  # TODO: support default value 0.0, i.e. no regularization
        "tol": 1e-6,
        "fit_intercept": True,
    }

    default_lr = LogisticRegression()

    assert_params(default_lr, default_spark_params, default_cuml_params)

    # Spark ML Params
    spark_params: Dict[str, Any] = {
        "maxIter": 30,
        "regParam": 0.5,
        "tol": 1e-2,
        "fitIntercept": False,
    }

    spark_lr = LogisticRegression(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(
        {
            "max_iter": 30,
            "C": 2.0,  # C should be equal to 1 / regParam
            "tol": 1e-2,
            "fit_intercept": False,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/logistic_regression_tests"
    estimator_path = f"{path}/logistic_regression"
    spark_lr.write().overwrite().save(estimator_path)
    loaded_lr = LogisticRegression.load(estimator_path)
    assert_params(loaded_lr, expected_spark_params, expected_cuml_params)

    # float32_inputs warn, logistic only accepts float32
    lr_float32 = LogisticRegression(float32_inputs=False)
    assert "float32_inputs to False" in caplog.text
    assert lr_float32._float32_inputs


# TODO support float64
# 'vector' will be converted to float32
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("feature_type", ["array", "multi_cols", "vector"])
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.slow
def test_classifier(
    fit_intercept: bool,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    gpu_number: int,
    n_classes: int = 2,
    reg_param: float = 0.0,
    tolerance: float = 0.001,
) -> None:
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    from cuml import LogisticRegression as cuLR

    penalty = "l2" if reg_param != 0.0 else "none"
    C = 1.0 / reg_param if reg_param != 0.0 else 0.0
    cu_lr = cuLR(fit_intercept=fit_intercept, penalty=penalty, C=C)
    cu_lr.solver_model.penalty_normalized = False
    cu_lr.solver_model.lbfgs_memory = 10
    cu_lr.fit(X_train, y_train)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_lr = LogisticRegression(
            fitIntercept=fit_intercept,
            regParam=reg_param,
            num_workers=gpu_number,
        )
        spark_lr.setFeaturesCol(features_col)
        spark_lr.setLabelCol(label_col)
        spark_lr_model: LogisticRegressionModel = spark_lr.fit(train_df)

        # test coefficients and intercepts
        assert spark_lr_model.n_cols == cu_lr.n_cols
        assert spark_lr_model.dtype == "float32"

        assert array_equal(np.array(spark_lr_model.coef_), cu_lr.coef_, tolerance)
        assert array_equal(spark_lr_model.intercept_, cu_lr.intercept_, tolerance)

        if n_classes == 2:
            assert len(spark_lr_model.coef_) == 1
            assert len(cu_lr.coef_) == 1
            assert array_equal(
                spark_lr_model.coefficients.toArray(), cu_lr.coef_[0], tolerance
            )
            assert spark_lr_model.intercept == pytest.approx(
                cu_lr.intercept_[0], tolerance
            )

        assert array_equal(
            spark_lr_model.coefficientMatrix.toArray(), cu_lr.coef_, tolerance
        )
        assert array_equal(
            spark_lr_model.interceptVector.toArray(), cu_lr.intercept_, tolerance
        )

        # test transform
        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = spark_lr_model.transform(test_df).collect()
        spark_preds = [row["prediction"] for row in result]
        cu_preds = cu_lr.predict(X_test)
        assert array_equal(cu_preds, spark_preds)

        spark_probs = np.array([row["probability"].toArray() for row in result])
        cu_probs = cu_lr.predict_proba(X_test)
        assert array_equal(spark_probs, cu_probs, tolerance)


LogisticRegressionType = TypeVar(
    "LogisticRegressionType", Type[LogisticRegression], Type[SparkLogisticRegression]
)
LogisticRegressionModelType = TypeVar(
    "LogisticRegressionModelType",
    Type[LogisticRegressionModel],
    Type[SparkLogisticRegressionModel],
)


@pytest.mark.compat
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat(
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    tmp_path: str,
) -> None:
    _LogisticRegression, _LogisticRegressionModel = lr_types

    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )
    y = np.array(
        [
            1.0,
            1.0,
            0.0,
            0.0,
        ]
    )
    num_rows = len(X)

    weight = np.ones([num_rows])
    feature_cols = ["c0", "c1"]
    schema = ["c0 float, c1 float, weight float, label float"]

    with CleanSparkSession() as spark:
        np_array = np.concatenate(
            (X, weight.reshape(num_rows, 1), y.reshape(num_rows, 1)), axis=1
        )

        bdf = spark.createDataFrame(
            np_array.tolist(),
            ",".join(schema),
        )

        bdf = bdf.withColumn("features", array_to_vector(array(*feature_cols))).drop(
            *feature_cols
        )

        assert _LogisticRegression().getRegParam() == 0.0
        if lr_types[0] is SparkLogisticRegression:
            blor = _LogisticRegression(regParam=0.1, standardization=False)
        else:
            warnings.warn("spark rapids ml does not accept standardization")
            blor = _LogisticRegression(regParam=0.1)

        assert blor.getRegParam() == 0.1

        blor.setFeaturesCol("features")
        blor.setMaxIter(10)
        blor.setRegParam(0.01)
        blor.setLabelCol("label")

        if isinstance(blor, SparkLogisticRegression):
            blor.setWeightCol("weight")

        assert blor.getFeaturesCol() == "features"
        assert blor.getMaxIter() == 10
        assert blor.getRegParam() == 0.01
        assert blor.getLabelCol() == "label"

        blor.clear(blor.maxIter)
        assert blor.getMaxIter() == 100

        blor_model = blor.fit(bdf)

        blor_model.setFeaturesCol("features")
        blor_model.setProbabilityCol("newProbability")
        blor_model.setRawPredictionCol("newRawPrediction")
        assert blor_model.getRawPredictionCol() == "newRawPrediction"
        assert blor_model.getProbabilityCol() == "newProbability"

        coefficients = blor_model.coefficients.toArray()
        intercept = blor_model.intercept

        assert array_equal(coefficients, [-2.42377087, 2.42377087])
        assert intercept == pytest.approx(0, abs=1e-6)

        example = bdf.head()
        if example:
            blor_model.predict(example.features)
            blor_model.predictRaw(example.features)
            blor_model.predictProbability(example.features)

        if isinstance(blor_model, SparkLogisticRegressionModel):
            assert blor_model.hasSummary
            blor_model.evaluate(bdf).accuracy == blor_model.summary.accuracy
        else:
            assert not blor_model.hasSummary
            with pytest.raises(RuntimeError, match="No training summary available"):
                blor_model.summary

        output_df = blor_model.transform(bdf)
        assert isinstance(output_df.schema["features"].dataType, VectorUDT)
        output = output_df.head()
        assert output.prediction == 1.0

        assert array_equal(
            output.newProbability.toArray(),
            Vectors.dense([0.0814, 0.9186]).toArray(),
        )

        if isinstance(blor_model, SparkLogisticRegressionModel):
            assert array_equal(
                output.newRawPrediction.toArray(),
                Vectors.dense([-2.4238, 2.4238]).toArray(),
            )
        else:
            warnings.warn(
                "transform of spark rapids ml currently does not support rawPredictionCol"
            )

        blor_path = tmp_path + "/log_reg"
        blor.save(blor_path)

        blor2 = _LogisticRegression.load(blor_path)
        assert blor2.getRegParam() == 0.01

        model_path = tmp_path + "/log_reg_model"
        blor_model.save(model_path)

        model2 = _LogisticRegressionModel.load(model_path)
        assert array_equal(
            blor_model.coefficients.toArray(), model2.coefficients.toArray()
        )
        assert blor_model.intercept == model2.intercept
        assert blor_model.transform(bdf).take(1) == model2.transform(bdf).take(1)
        assert blor_model.numFeatures == 2


@pytest.mark.parametrize("feature_type", [feature_types.vector])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("n_classes", [2])
def test_lr_fit_multiple_in_single_pass(
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
    n_classes: int,
) -> None:
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    with CleanSparkSession() as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        lr = LogisticRegression()
        lr.setFeaturesCol(features_col)
        lr.setLabelCol(label_col)

        initial_lr = lr.copy()

        param_maps: List[Dict[Param, Any]] = [
            {
                lr.tol: 1,
                lr.regParam: 0,
                lr.fitIntercept: True,
                lr.maxIter: 39,
            },
            {
                lr.tol: 0.01,
                lr.regParam: 0.5,
                lr.fitIntercept: False,
                lr.maxIter: 100,
            },
            {
                lr.tol: 0.03,
                lr.regParam: 0.7,
                lr.fitIntercept: True,
                lr.maxIter: 29,
            },
            {
                lr.tol: 0.0003,
                lr.regParam: 0.9,
                lr.fitIntercept: False,
                lr.maxIter: 89,
            },
        ]
        models = lr.fit(train_df, param_maps)

        for i, param_map in enumerate(param_maps):
            rf = initial_lr.copy()
            single_model = rf.fit(train_df, param_map)

            assert array_equal(
                single_model.coefficients.toArray(), models[i].coefficients.toArray()
            )
            assert array_equal([single_model.intercept], [models[i].intercept])

            for k, v in param_map.items():
                assert models[i].getOrDefault(k.name) == v
                assert single_model.getOrDefault(k.name) == v


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("feature_type", ["array", "multi_cols", "vector"])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
@pytest.mark.parametrize("max_record_batch", [20, 10000])
@pytest.mark.parametrize("n_classes", [8])
@pytest.mark.slow
def test_multiclass(
    fit_intercept: bool,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
) -> None:
    tolerance = 0.005

    test_classifier(
        fit_intercept=fit_intercept,
        feature_type=feature_type,
        data_shape=data_shape,
        data_type=data_type,
        max_record_batch=max_record_batch,
        n_classes=n_classes,
        gpu_number=gpu_number,
        reg_param=0.1,
        tolerance=tolerance,
    )


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("reg_param", [0.0, 0.1])
@pytest.mark.parametrize("feature_type", ["vector"])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("max_record_batch", [20])
@pytest.mark.parametrize("n_classes", [2, 4])
def test_quick(
    fit_intercept: bool,
    reg_param: float,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
) -> None:
    tolerance = 0.005

    test_classifier(
        fit_intercept=fit_intercept,
        feature_type=feature_type,
        data_shape=data_shape,
        data_type=data_type,
        max_record_batch=max_record_batch,
        n_classes=n_classes,
        gpu_number=gpu_number,
        reg_param=reg_param,
        tolerance=tolerance,
    )

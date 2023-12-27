from typing import Any, Dict, List, Tuple, Type, TypeVar

import cuml
import numpy as np
import pyspark
import pytest
from _pytest.logging import LogCaptureFixture
from packaging import version

if version.parse(pyspark.__version__) < version.parse("3.4.0"):
    from pyspark.sql.utils import IllegalArgumentException  # type: ignore
else:
    from pyspark.errors import IllegalArgumentException  # type: ignore

from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import (
    LogisticRegressionModel as SparkLogisticRegressionModel,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseMatrix, DenseVector, SparseVector, Vectors, VectorUDT
from pyspark.ml.param import Param
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder
from pyspark.sql import Row
from pyspark.sql.functions import array, col

if version.parse(cuml.__version__) < version.parse("23.08.00"):
    raise ValueError(
        "Logistic Regression requires cuml 23.08.00 or above. Try upgrading cuml or ignoring this file in testing"
    )

import warnings

from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel
from spark_rapids_ml.tuning import CrossValidator

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    feature_types,
    idfn,
    make_classification_dataset,
)


def test_toy_example(gpu_number: int) -> None:
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
            probs = [row["probs"] for row in preds_df_local]
            raw_preds = [row["rawPrediction"] for row in preds_df_local]
            assert preds == [1.0, 1.0, 0.0, 0.0]
            assert len(probs) == len(preds)
            assert [p[1] > 0.5 for p in probs] == [True, True, False, False]
            assert [p[1] > 0 for p in raw_preds] == [True, True, False, False]

        assert_transform(lr_model)

        # test with regParam set to 0
        lr_regParam_zero = LogisticRegression(
            regParam=0.0,
        )

        lr_regParam_zero.setProbabilityCol(probability_col)

        assert lr_regParam_zero.getRegParam() == 0
        assert lr_regParam_zero.cuml_params["C"] == 0
        model_regParam_zero = lr_regParam_zero.fit(df)
        assert_transform(model_regParam_zero)

        lr_regParam_zero.setRegParam(0.1)
        assert lr_regParam_zero.getRegParam() == 0.1
        assert lr_regParam_zero.cuml_params["C"] == 1.0 / 0.1

        lr_regParam_zero.setRegParam(0.0)

        assert lr_regParam_zero.getRegParam() == 0.0
        assert lr_regParam_zero.cuml_params["C"] == 0.0


def test_params(tmp_path: str, caplog: LogCaptureFixture) -> None:
    # Default params: no regularization
    default_spark_params = {
        "maxIter": 100,
        "regParam": 0.0,
        "elasticNetParam": 0.0,
        "tol": 1e-06,
        "fitIntercept": True,
    }

    default_cuml_params = {
        "max_iter": 100,
        "penalty": "none",
        "C": 0.0,
        "l1_ratio": 0.0,
        "tol": 1e-6,
        "fit_intercept": True,
    }

    default_lr = LogisticRegression()

    assert_params(default_lr, default_spark_params, default_cuml_params)

    # L2 regularization
    spark_params: Dict[str, Any] = {
        "maxIter": 30,
        "regParam": 0.5,
        "elasticNetParam": 0.0,
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
            "penalty": "l2",
            "C": 2.0,  # C should be equal to 1.0 / regParam
            "l1_ratio": 0.0,
            "tol": 1e-2,
            "fit_intercept": False,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)

    # L1 regularization
    spark_params = {
        "maxIter": 30,
        "regParam": 0.5,
        "elasticNetParam": 1.0,
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
            "penalty": "l1",
            "C": 2.0,  # C should be equal to 1.0 / regParam
            "l1_ratio": 1.0,
            "tol": 1e-2,
            "fit_intercept": False,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)

    # elasticnet(L1 + L2) regularization
    spark_params = {
        "maxIter": 30,
        "regParam": 0.5,
        "elasticNetParam": 0.3,
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
            "penalty": "elasticnet",
            "C": 2.0,  # C should be equal to 1.0 / regParam
            "l1_ratio": 0.3,
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
    elasticNet_param: float = 0.0,
    tolerance: float = 0.001,
) -> LogisticRegression:
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

    penalty, C, l1_ratio = LogisticRegression._reg_params_value_mapping(
        reg_param=reg_param, elasticNet_param=elasticNet_param
    )

    cu_lr = cuLR(fit_intercept=fit_intercept, penalty=penalty, C=C, l1_ratio=l1_ratio)
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
            elasticNetParam=elasticNet_param,
            num_workers=gpu_number,
        )

        assert spark_lr._cuml_params["penalty"] == cu_lr.penalty
        assert spark_lr._cuml_params["C"] == cu_lr.C
        if cu_lr.penalty == "elasticnet":
            assert spark_lr._cuml_params["l1_ratio"] == cu_lr.l1_ratio
        else:
            assert spark_lr._cuml_params["l1_ratio"] == spark_lr.getElasticNetParam()
            assert cu_lr.l1_ratio == None

        spark_lr.setFeaturesCol(features_col)
        spark_lr.setLabelCol(label_col)
        spark_lr_model: LogisticRegressionModel = spark_lr.fit(train_df)

        # test coefficients and intercepts
        assert spark_lr_model.n_cols == cu_lr.n_cols
        assert spark_lr_model.dtype == "float32"

        assert array_equal(np.array(spark_lr_model.coef_), cu_lr.coef_, tolerance)
        assert array_equal(spark_lr_model.intercept_, cu_lr.intercept_, tolerance)  # type: ignore

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

        return spark_lr


LogisticRegressionType = TypeVar(
    "LogisticRegressionType", Type[LogisticRegression], Type[SparkLogisticRegression]
)
LogisticRegressionModelType = TypeVar(
    "LogisticRegressionModelType",
    Type[LogisticRegressionModel],
    Type[SparkLogisticRegressionModel],
)


@pytest.mark.compat
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat(
    fit_intercept: bool,
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    tmp_path: str,
) -> None:
    tolerance = 0.001

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
            blor = _LogisticRegression(
                regParam=0.1, fitIntercept=fit_intercept, standardization=False
            )
        else:
            warnings.warn("spark rapids ml does not accept standardization")
            blor = _LogisticRegression(regParam=0.1, fitIntercept=fit_intercept)

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

        assert isinstance(blor_model.coefficients, DenseVector)
        assert array_equal(
            blor_model.coefficients.toArray(), [-2.42377087, 2.42377087], tolerance
        )
        assert blor_model.intercept == pytest.approx(0, abs=1e-6)

        assert isinstance(blor_model.coefficientMatrix, DenseMatrix)
        assert array_equal(
            blor_model.coefficientMatrix.toArray(),
            np.array([[-2.42377087, 2.42377087]]),
            tolerance,
        )
        assert isinstance(blor_model.interceptVector, DenseVector)
        assert array_equal(blor_model.interceptVector.toArray(), [0.0])

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
            tolerance,
        )

        array_equal(
            output.newRawPrediction.toArray(),
            Vectors.dense([-2.4238, 2.4238]).toArray(),
            tolerance,
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


@pytest.mark.compat
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat_multinomial(
    fit_intercept: bool,
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    tmp_path: str,
) -> None:
    _LogisticRegression, _LogisticRegressionModel = lr_types
    tolerance = 0.001

    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [-1.0, -2.0],
            [-1.0, -3.0],
            [-2.0, -1.0],
            [-3.0, -1.0],
        ]
    )
    y = np.array(
        [
            1.0,
            1.0,
            0.0,
            0.0,
            3.0,
            3.0,
            2.0,
            2.0,
        ]
    )
    data_type = np.float32

    num_rows = len(X)
    weight = np.ones([num_rows])

    feature_cols = ["c0", "c1"]
    schema = ["c0 float, c1 float, weight float, label float"]

    with CleanSparkSession() as spark:
        np_array = np.concatenate(
            (X, weight.reshape(num_rows, 1), y.reshape(num_rows, 1)), axis=1
        )

        mdf = spark.createDataFrame(
            np_array.tolist(),
            ",".join(schema),
        )

        mdf = mdf.withColumn("features", array_to_vector(array(*feature_cols))).drop(
            *feature_cols
        )

        np_array = np.concatenate(
            (X, weight.reshape(num_rows, 1), y.reshape(num_rows, 1)), axis=1
        )

        mdf = spark.createDataFrame(
            np_array.tolist(),
            ",".join(schema),
        )

        mdf = mdf.withColumn("features", array_to_vector(array(*feature_cols))).drop(
            *feature_cols
        )

        assert _LogisticRegression().getRegParam() == 0.0
        if lr_types[0] is SparkLogisticRegression:
            mlor = _LogisticRegression(
                regParam=0.1,
                elasticNetParam=0.2,
                fitIntercept=fit_intercept,
                family="multinomial",
                standardization=False,
            )
        else:
            warnings.warn("spark rapids ml does not accept standardization")
            mlor = _LogisticRegression(
                regParam=0.1,
                elasticNetParam=0.2,
                fitIntercept=fit_intercept,
                family="multinomial",
            )

        assert mlor.getRegParam() == 0.1
        assert mlor.getElasticNetParam() == 0.2

        mlor.setRegParam(0.15)
        mlor.setElasticNetParam(0.25)
        assert mlor.getRegParam() == 0.15
        assert mlor.getElasticNetParam() == 0.25

        mlor.setRegParam(0.1)
        mlor.setElasticNetParam(0.2)

        mlor.setFeaturesCol("features")
        mlor.setLabelCol("label")

        if isinstance(mlor, SparkLogisticRegression):
            mlor.setWeightCol("weight")

        mlor_model = mlor.fit(mdf)

        mlor_model.setProbabilityCol("newProbability")
        assert mlor_model.getProbabilityCol() == "newProbability"

        with pytest.raises(
            Exception,
            match="Multinomial models contain a matrix of coefficients, use coefficientMatrix instead.",
        ):
            mlor_model.coefficients

        with pytest.raises(
            Exception,
            match="Multinomial models contain a vector of intercepts, use interceptVector instead.",
        ):
            mlor_model.intercept

        assert isinstance(mlor_model.coefficientMatrix, DenseMatrix)
        coef_mat = mlor_model.coefficientMatrix.toArray()

        if fit_intercept == False:
            assert isinstance(mlor_model.interceptVector, SparseVector)
        elif isinstance(mlor_model, SparkLogisticRegressionModel):
            # Note Spark may return a SparseVector of all zeroes
            assert isinstance(mlor_model.interceptVector, DenseVector) or isinstance(
                mlor_model.interceptVector, SparseVector
            )
        else:
            # Note Spark Rapids ML returns a DenseVector of tiny non-zeroes
            assert isinstance(mlor_model.interceptVector, DenseVector)

        intercept_vec = mlor_model.interceptVector.toArray()

        coef_ground = [
            [0.96766883, -0.06190176],
            [-0.06183558, 0.96774077],
            [-0.96773398, 0.06184808],
            [0.06187553, -0.96768212],
        ]

        intercept_ground = [
            1.78813821e-07,
            2.82220935e-05,
            1.44387586e-05,
            4.82081663e-09,
        ]
        assert array_equal(coef_mat, np.array(coef_ground), tolerance)
        assert array_equal(intercept_vec, intercept_ground, tolerance)

        example = mdf.head()
        if example:
            mlor_model.predict(example.features)
            mlor_model.predictRaw(example.features)
            mlor_model.predictProbability(example.features)

        if isinstance(mlor_model, SparkLogisticRegressionModel):
            assert mlor_model.hasSummary
            mlor_model.evaluate(mdf).accuracy == mlor_model.summary.accuracy
        else:
            assert not mlor_model.hasSummary
            with pytest.raises(RuntimeError, match="No training summary available"):
                mlor_model.summary
            assert mlor_model.classes_ == [0.0, 1.0, 2.0, 3.0]

        # test transform
        output_df = mlor_model.transform(mdf)
        assert isinstance(output_df.schema["features"].dataType, VectorUDT)

        # TODO: support (1) weight and rawPrediction (2) newProbability column is before prediction column
        if isinstance(mlor_model, SparkLogisticRegressionModel):
            assert output_df.schema.fieldNames() == [
                "weight",
                "label",
                "features",
                "rawPrediction",
                "newProbability",
                "prediction",
            ]
            assert (
                output_df.schema.simpleString()
                == "struct<weight:float,label:float,features:vector,rawPrediction:vector,newProbability:vector,prediction:double>"
            )
        else:
            assert output_df.schema.fieldNames() == [
                "weight",
                "label",
                "features",
                "prediction",
                "newProbability",
                "rawPrediction",
            ]
            assert (
                output_df.schema.simpleString()
                == "struct<weight:float,label:float,features:vector,prediction:double,newProbability:vector,rawPrediction:vector>"
            )

        output_res = output_df.collect()
        assert array_equal(
            [row.prediction for row in output_res],
            [1.0, 1.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0],
        )

        assert array_equal(
            output_res[0].newProbability.toArray(),
            [0.24686976, 0.69117839, 0.04564774, 0.01630411],
            tolerance,
        )
        assert array_equal(
            output_res[1].newProbability.toArray(),
            [0.11019694, 0.86380164, 0.02305966, 0.00294177],
            tolerance,
        )
        assert array_equal(
            output_res[2].newProbability.toArray(),
            [0.69117839, 0.24686976, 0.01630411, 0.04564774],
            tolerance,
        )
        assert array_equal(
            output_res[3].newProbability.toArray(),
            [0.86380164, 0.11019694, 0.00294177, 0.02305966],
            tolerance,
        )
        assert array_equal(
            output_res[4].newProbability.toArray(),
            [0.04564774, 0.01630411, 0.24686976, 0.69117839],
            tolerance,
        )
        assert array_equal(
            output_res[5].newProbability.toArray(),
            [0.02305966, 0.00294177, 0.11019694, 0.86380164],
            tolerance,
        )
        assert array_equal(
            output_res[6].newProbability.toArray(),
            [0.01630352, 0.04563958, 0.6912151, 0.24684186],
            tolerance,
        )
        assert array_equal(
            output_res[7].newProbability.toArray(),
            [0.00294145, 0.02305316, 0.86383104, 0.11017438],
            tolerance,
        )

        assert array_equal(
            output_res[0].rawPrediction.toArray(),
            [0.84395339, 1.87349042, -0.84395339, -1.87349042],
            tolerance,
        )
        assert array_equal(
            output_res[1].rawPrediction.toArray(),
            [0.78209218, 2.84116623, -0.78209218, -2.84116623],
            tolerance,
        )
        assert array_equal(
            output_res[2].rawPrediction.toArray(),
            [1.87349042, 0.84395339, -1.87349042, -0.84395339],
            tolerance,
        )
        assert array_equal(
            output_res[3].rawPrediction.toArray(),
            [2.84116623, 0.78209218, -2.84116623, -0.78209218],
            tolerance,
        )
        assert array_equal(
            output_res[4].rawPrediction.toArray(),
            [-0.84395339, -1.87349042, 0.84395339, 1.87349042],
            tolerance,
        )
        assert array_equal(
            output_res[5].rawPrediction.toArray(),
            [-0.78209218, -2.84116623, 0.78209218, 2.84116623],
            tolerance,
        )
        assert array_equal(
            output_res[6].rawPrediction.toArray(),
            [-1.87349042, -0.84395339, 1.87349042, 0.84395339],
            tolerance,
        )
        assert array_equal(
            output_res[7].rawPrediction.toArray(),
            [-2.84116623, -0.78209218, 2.84116623, 0.78209218],
            tolerance,
        )

        mlor_path = tmp_path + "/m_log_reg"
        mlor.save(mlor_path)

        mlor2 = _LogisticRegression.load(mlor_path)
        assert mlor2.getRegParam() == mlor.getRegParam()
        assert mlor2.getElasticNetParam() == mlor.getElasticNetParam()

        model_path = tmp_path + "/m_log_reg_model"
        mlor_model.save(model_path)

        model2 = _LogisticRegressionModel.load(model_path)
        assert array_equal(
            model2.coefficientMatrix.toArray(), mlor_model.coefficientMatrix.toArray()
        )
        assert model2.interceptVector == mlor_model.interceptVector
        assert model2.transform(mdf).collect() == output_res
        assert model2.numFeatures == 2


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("feature_type", ["vector"])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
@pytest.mark.parametrize("max_record_batch", [20])
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
@pytest.mark.parametrize(
    "reg_factors", [(0.0, 0.0), (0.1, 0.0), (0.1, 1.0), (0.1, 0.2)]
)
@pytest.mark.parametrize("feature_type", ["vector"])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("max_record_batch", [20])
@pytest.mark.parametrize("n_classes", [2, 4])
def test_quick(
    fit_intercept: bool,
    reg_factors: Tuple[float, float],
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
) -> None:
    tolerance = 0.005
    reg_param = reg_factors[0]
    elasticNet_param = reg_factors[1]

    lr = test_classifier(
        fit_intercept=fit_intercept,
        feature_type=feature_type,
        data_shape=data_shape,
        data_type=data_type,
        max_record_batch=max_record_batch,
        n_classes=n_classes,
        gpu_number=gpu_number,
        tolerance=tolerance,
        reg_param=reg_param,
        elasticNet_param=elasticNet_param,
    )

    assert lr.getRegParam() == reg_param
    assert lr.getElasticNetParam() == elasticNet_param

    penalty, C, l1_ratio = lr._reg_params_value_mapping(reg_param, elasticNet_param)
    assert lr._cuml_params["penalty"] == penalty
    assert lr._cuml_params["C"] == C
    assert lr._cuml_params["l1_ratio"] == l1_ratio

    from cuml import LogisticRegression as CUMLSG

    sg = CUMLSG(penalty=penalty, C=C, l1_ratio=l1_ratio)
    l1_strength, l2_strength = sg._get_qn_params()
    if reg_param == 0.0:
        assert penalty == "none"
        assert l1_strength == 0.0
        assert l2_strength == 0.0
    elif elasticNet_param == 0.0:
        assert penalty == "l2"
        assert l1_strength == 0.0
        assert l2_strength == reg_param
    elif elasticNet_param == 1.0:
        assert penalty == "l1"
        assert l1_strength == reg_param
        assert l2_strength == 0.0
    else:
        assert penalty == "elasticnet"
        assert l1_strength == reg_param * elasticNet_param
        assert l2_strength == reg_param * (1 - elasticNet_param)


@pytest.mark.parametrize("metric_name", ["accuracy", "logLoss", "areaUnderROC"])
@pytest.mark.parametrize("feature_type", [feature_types.vector])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
def test_crossvalidator_logistic_regression(
    metric_name: str,
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
) -> None:
    # Train a toy model

    n_classes = 2 if metric_name == "areaUnderROC" else 10

    X, _, y, _ = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=data_shape[1],
        n_redundant=0,
        n_repeated=0,
    )

    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )
        assert label_col is not None

        lr = LogisticRegression()
        lr.setFeaturesCol(features_col)
        lr.setLabelCol(label_col)

        evaluator = (
            BinaryClassificationEvaluator()
            if n_classes == 2
            else MulticlassClassificationEvaluator()
        )
        evaluator.setLabelCol(label_col)  # type: ignore
        evaluator.setMetricName(metric_name)  # type: ignore

        grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, [0.1, 0.2])
            .addGrid(lr.elasticNetParam, [0.2, 0.5])
            .build()
        )

        cv = CrossValidator(
            estimator=lr,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=2,
            seed=1,
        )

        # without exception
        model: CrossValidatorModel = cv.fit(df)

        spark_cv = SparkCrossValidator(
            estimator=lr,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=2,
            seed=1,
        )
        spark_cv_model = spark_cv.fit(df)

        assert array_equal(model.avgMetrics, spark_cv_model.avgMetrics, 0.0005)


def test_parameters_validation() -> None:
    data = [
        ([1.0, 2.0], 1.0),
        ([3.0, 1.0], 0.0),
    ]

    with CleanSparkSession() as spark:
        features_col = "features"
        label_col = "label"
        schema = features_col + " array<float>, " + label_col + " float"
        df = spark.createDataFrame(data, schema=schema)
        with pytest.raises(
            IllegalArgumentException, match="maxIter given invalid value -1"
        ):
            LogisticRegression(maxIter=-1).fit(df)

        # regParam is mapped to different value in LogisticRegression which should be in
        # charge of validating it.
        with pytest.raises(ValueError, match="C or regParam given invalid value -1.0"):
            LogisticRegression().setRegParam(-1.0).fit(df)


@pytest.mark.compat
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("label", [1.0, 0.0, -3.0, 4.0])
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat_one_label(
    fit_intercept: bool,
    label: float,
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    caplog: LogCaptureFixture,
) -> None:
    assert label % 1 == 0.0, "label value must be an integer"

    tolerance = 0.001
    _LogisticRegression, _LogisticRegressionModel = lr_types

    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )
    y = np.array([label] * 4)

    num_rows = len(X)

    feature_cols = ["c0", "c1"]
    schema = ["c0 float, c1 float, label float"]

    with CleanSparkSession() as spark:
        np_array = np.concatenate((X, y.reshape(num_rows, 1)), axis=1)

        bdf = spark.createDataFrame(
            np_array.tolist(),
            ",".join(schema),
        )

        bdf = bdf.withColumn("features", array_to_vector(array(*feature_cols))).drop(
            *feature_cols
        )

        blor = _LogisticRegression(
            regParam=0.1, fitIntercept=fit_intercept, standardization=False
        )

        if label < 0:
            from py4j.protocol import Py4JJavaError

            msg = f"Labels MUST be in [0, 2147483647), but got {label}"

            try:
                blor_model = blor.fit(bdf)
                assert False, "There should be a java exception"
            except Py4JJavaError as e:
                assert msg in e.java_exception.getMessage()

            return

        if label > 1:  # Spark and Cuml do not match
            if _LogisticRegression is SparkLogisticRegression:
                blor_model = blor.fit(bdf)
                assert blor_model.numClasses == label + 1
            else:
                blor_model = blor.fit(bdf)
                assert blor_model.numClasses == 1

            return

        assert label == 1.0 or label == 0.0

        blor_model = blor.fit(bdf)

        if fit_intercept is False:
            if _LogisticRegression is SparkLogisticRegression:
                # Got empty caplog.text. Spark prints warning message from jvm
                assert caplog.text == ""
            else:
                assert (
                    "All labels belong to a single class and fitIntercept=false. It's a dangerous ground, so the algorithm may not converge."
                    in caplog.text
                )

            if label == 1.0:
                assert array_equal(
                    blor_model.coefficients.toArray(),
                    [0.85431526, 0.85431526],
                    tolerance,
                )
            else:
                assert array_equal(
                    blor_model.coefficients.toArray(),
                    [-0.85431526, -0.85431526],
                    tolerance,
                )
            assert blor_model.intercept == 0.0
        else:
            if _LogisticRegression is SparkLogisticRegression:
                # Got empty caplog.text. Spark prints warning message from jvm
                assert caplog.text == ""
            else:
                assert (
                    "All labels are the same value and fitIntercept=true, so the coefficients will be zeros. Training is not needed."
                    in caplog.text
                )

            assert array_equal(blor_model.coefficients.toArray(), [0, 0], 0.0)
            assert blor_model.intercept == (
                float("inf") if label == 1.0 else float("-inf")
            )

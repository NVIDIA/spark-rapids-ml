# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import cuml
import cupyx.scipy.sparse
import numpy as np
import pandas as pd
import pyspark
import pytest
from _pytest.logging import LogCaptureFixture
from gen_data_distributed import SparseRegressionDataGen
from packaging import version
from py4j.protocol import Py4JJavaError

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
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import array, col, sum, udf
from pyspark.sql.types import FloatType, LongType

if version.parse(cuml.__version__) < version.parse("23.08.00"):
    raise ValueError(
        "Logistic Regression requires cuml 23.08.00 or above. Try upgrading cuml or ignoring this file in testing"
    )

import random

random.seed(0)

from scipy.sparse import csr_matrix

from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel
from spark_rapids_ml.core import _CumlEstimator, _use_sparse_in_cuml, alias
from spark_rapids_ml.tuning import CrossValidator

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    feature_types,
    get_default_cuml_parameters,
    idfn,
    make_classification_dataset,
)


def check_sparse_estimator_preprocess(
    lr: LogisticRegression, df: DataFrame, dimension: int
) -> None:
    (select_cols, multi_col_names, dimension, feature_type) = lr._pre_process_data(df)
    internal_df = df.select(*select_cols)
    field_names = internal_df.schema.fieldNames()
    assert field_names == [
        alias.featureVectorType,
        alias.featureVectorSize,
        alias.featureVectorIndices,
        alias.data,
        alias.label,
    ]
    assert multi_col_names is None
    assert dimension == dimension
    assert feature_type == FloatType
    assert _use_sparse_in_cuml(internal_df) is True


def check_sparse_model_preprocess(
    model: LogisticRegressionModel, df: DataFrame
) -> None:
    (internal_df, select_cols, input_is_multi_cols, tmp_cols) = model._pre_process_data(
        df
    )
    df_field_names = df.schema.fieldNames()
    internal_df_field_names = internal_df.schema.fieldNames()
    unwrapped_col_names = [
        alias.featureVectorType,
        alias.featureVectorSize,
        alias.featureVectorIndices,
        alias.data,
    ]
    assert internal_df_field_names == df_field_names + unwrapped_col_names
    assert select_cols == unwrapped_col_names
    assert tmp_cols == select_cols
    assert input_is_multi_cols is False
    assert _use_sparse_in_cuml(internal_df) is True


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

        lr_estimator = LogisticRegression(
            standardization=False, regParam=1.0, num_workers=gpu_number
        )
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
    from cuml import LogisticRegression as CumlLogisticRegression
    from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression

    # Default params: no regularization
    default_spark_params = {
        param.name: value
        for param, value in SparkLogisticRegression().extractParamMap().items()
    }

    default_cuml_params = get_default_cuml_parameters(
        cuml_classes=[CumlLogisticRegression],
        excludes=[
            "class_weight",
            "linesearch_max_iter",
            "solver",
            "handle",
            "output_type",
        ],
    )

    default_cuml_params["standardization"] = (
        False  # Standardization param exists in LogisticRegressionMG (default = False) but not in SG, and we support it. Add it in manually for this check.
    )

    # Ensure internal cuml defaults match actual cuml defaults
    assert default_cuml_params == LogisticRegression()._get_cuml_params_default()

    # Our algorithm overrides the following cuml parameters with their spark defaults:
    spark_default_overrides = {
        "tol": default_spark_params["tol"],
        "max_iter": default_spark_params["maxIter"],
        "standardization": default_spark_params["standardization"],
        "C": default_spark_params["regParam"],
        "l1_ratio": default_spark_params[
            "elasticNetParam"
        ],  # set to 0.0 when reg_param == 0.0
        "penalty": None,  # set to None when reg_param == 0.0
    }

    default_cuml_params.update(spark_default_overrides)

    default_lr = LogisticRegression()

    assert_params(default_lr, default_spark_params, default_cuml_params)
    assert default_lr.cuml_params == default_cuml_params

    # L2 regularization
    spark_params: Dict[str, Any] = {
        "maxIter": 30,
        "regParam": 0.5,
        "elasticNetParam": 0.0,
        "tol": 1e-2,
        "fitIntercept": False,
        "standardization": False,
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
            "standardization": False,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)
    assert spark_lr.cuml_params == expected_cuml_params

    # L1 regularization
    spark_params = {
        "maxIter": 30,
        "regParam": 0.5,
        "elasticNetParam": 1.0,
        "tol": 1e-2,
        "fitIntercept": False,
        "standardization": False,
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
            "standardization": False,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)
    assert spark_lr.cuml_params == expected_cuml_params

    # elasticnet(L1 + L2) regularization
    spark_params = {
        "maxIter": 30,
        "regParam": 0.5,
        "elasticNetParam": 0.3,
        "tol": 1e-2,
        "fitIntercept": False,
        "standardization": True,
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
            "standardization": True,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/logistic_regression_tests"
    estimator_path = f"{path}/logistic_regression"
    spark_lr.write().overwrite().save(estimator_path)
    loaded_lr = LogisticRegression.load(estimator_path)
    assert_params(loaded_lr, expected_spark_params, expected_cuml_params)

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(LogisticRegression)


def test_lr_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [
        (
            {"regParam": 0.1, "elasticNetParam": 0.5},
            {"penalty": "elasticnet", "C": 10.0, "l1_ratio": 0.5},
        ),
        (
            {"maxIter": 13},
            {"max_iter": 13},
        ),
        (
            {"regParam": 0.25, "elasticNetParam": 0.0},
            {"penalty": "l2", "C": 4.0, "l1_ratio": 0.0},
        ),
        (
            {"regParam": 0.2, "elasticNetParam": 1.0},
            {"penalty": "l1", "C": 5.0, "l1_ratio": 1.0},
        ),
        (
            {"tol": 1e-3},
            {"tol": 1e-3},
        ),
        (
            {"fitIntercept": False},
            {"fit_intercept": False},
        ),
        (
            {"standardization": False},
            {"standardization": False},
        ),
        (
            {"enable_sparse_data_optim": True},
            None,
        ),
        (
            {"verbose": True},
            {"verbose": True},
        ),
    ]

    for pair in param_list:
        input_spark_params = pair[0]
        cuml_params_update = pair[1]
        _test_est_copy(LogisticRegression, input_spark_params, cuml_params_update)


def test_lr_model_copy() -> None:

    from .test_common_estimator import _test_model_copy
    from .utils import get_toy_model

    model_params: List[Dict[str, Any]] = [
        {"featuresCol": "fea_dummy"},
        {"predictionCol": "fea_dummy"},
        {"probabilityCol": "fea_dummy"},
        {"rawPredictionCol": "fea_dummy"},
    ]
    with CleanSparkSession() as spark:
        gpu_model = get_toy_model(LogisticRegression, spark)
        cpu_model = get_toy_model(SparkLogisticRegression, spark)

        for p in model_params:
            _test_model_copy(gpu_model, cpu_model, p)


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
    convert_to_sparse: bool = False,
    set_float32_inputs: Optional[bool] = None,
    verbose: bool = False,
    spark_conf: Dict[str, Any] = {},
) -> LogisticRegression:
    standardization: bool = False

    float32_inputs = set_float32_inputs
    if float32_inputs is None:
        float32_inputs = random.choice([True, False])

    if convert_to_sparse is True:
        assert feature_type == "vector"

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

    spark_conf.update(
        {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    )

    with CleanSparkSession(spark_conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )
        if convert_to_sparse:
            assert type(features_col) is str

            from pyspark.sql.functions import udf

            def to_sparse_func(v: Union[SparseVector, DenseVector]) -> SparseVector:
                if isinstance(v, DenseVector):
                    return SparseVector(len(v), range(len(v)), v.toArray())
                else:
                    return v

            udf_to_sparse = udf(to_sparse_func, VectorUDT())
            train_df = train_df.withColumn(features_col, udf_to_sparse(features_col))

        assert label_col is not None
        spark_lr = LogisticRegression(
            enable_sparse_data_optim=convert_to_sparse,
            standardization=standardization,
            fitIntercept=fit_intercept,
            regParam=reg_param,
            elasticNetParam=elasticNet_param,
            num_workers=gpu_number,
            float32_inputs=float32_inputs,
            verbose=verbose,
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

        # test float32_inputs
        assert spark_lr_model._float32_inputs == float32_inputs
        if float32_inputs is True:
            assert spark_lr_model.dtype == "float32"
        elif feature_type is "vector":
            assert spark_lr_model.dtype == "float64"
        else:
            assert spark_lr_model.dtype == np.dtype(data_type)

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
        assert array_equal(cu_preds, spark_preds, total_tol=tolerance)

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
@pytest.mark.parametrize("standardization", [True, False])
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat(
    fit_intercept: bool,
    standardization: bool,
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
        blor = _LogisticRegression(
            regParam=0.1, fitIntercept=fit_intercept, standardization=standardization
        )

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

        coef_gnd = (
            [-2.48197058, 2.48197058]
            if standardization is True
            else [-2.42377087, 2.42377087]
        )
        assert array_equal(blor_model.coefficients.toArray(), coef_gnd, tolerance)
        assert blor_model.intercept == pytest.approx(0, abs=tolerance)

        assert isinstance(blor_model.coefficientMatrix, DenseMatrix)
        assert array_equal(
            blor_model.coefficientMatrix.toArray(),
            np.array([coef_gnd]),
            tolerance,
        )
        assert isinstance(blor_model.interceptVector, DenseVector)
        assert array_equal(blor_model.interceptVector.toArray(), [0.0], tolerance)

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

        prob_gnd = (
            [0.07713181, 0.92286819] if standardization is True else [0.0814, 0.9186]
        )
        assert array_equal(
            output.newProbability.toArray(),
            Vectors.dense(prob_gnd).toArray(),
            tolerance,
        )

        rawPredict_gnd = (
            [-2.48197058, 2.48197058] if standardization is True else [-2.4238, 2.4238]
        )
        assert array_equal(
            output.newRawPrediction.toArray(),
            Vectors.dense(rawPredict_gnd).toArray(),
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
    tolerance = 1e-3
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
                single_model.coefficients.toArray(),
                models[i].coefficients.toArray(),
                tolerance,
            )
            assert array_equal(
                [single_model.intercept], [models[i].intercept], tolerance
            )

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
        mlor = _LogisticRegression(
            regParam=0.1,
            elasticNetParam=0.2,
            fitIntercept=fit_intercept,
            family="multinomial",
            standardization=False,
        )

        assert mlor.getStandardization() == False
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
    float32_inputs: bool = True,
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
        set_float32_inputs=float32_inputs,
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
        assert penalty == None
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
    convert_to_sparse: bool = False,
) -> None:

    if convert_to_sparse:
        assert feature_type == feature_types.vector

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            import logging

            err_msg = "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function. "
            "The test case will be skipped. Please install pyspark>=3.4."
            logging.info(err_msg)
        return

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

        if convert_to_sparse:
            assert type(features_col) is str

            from pyspark.sql.functions import udf

            def to_sparse_func(v: Union[SparseVector, DenseVector]) -> SparseVector:
                if isinstance(v, DenseVector):
                    return SparseVector(len(v), range(len(v)), v.toArray())
                else:
                    return v

            udf_to_sparse = udf(to_sparse_func, VectorUDT())
            df = df.withColumn(features_col, udf_to_sparse(features_col))

        lr = LogisticRegression(enable_sparse_data_optim=convert_to_sparse)
        lr.setFeaturesCol(features_col)
        lr.setLabelCol(label_col)

        if convert_to_sparse is True:
            check_sparse_estimator_preprocess(lr, df, data_shape[1])

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
            spark_v34_msg = f"Labels MUST be in [0, 2147483647), but got {label}"
            spark_v33_msg = (
                f"Classification labels should be in [0 to -1]. Found 4 invalid labels."
            )

            try:
                blor_model = blor.fit(bdf)
                assert False, "There should be a java exception"
            except Py4JJavaError as e:
                java_msg = e.java_exception.getMessage()
                assert spark_v34_msg in java_msg or spark_v33_msg in java_msg

            return

        if label > 1:  # Spark and Cuml do not match
            if _LogisticRegression is SparkLogisticRegression:
                blor_model = blor.fit(bdf)
                assert blor_model.numClasses == label + 1
            else:
                msg = "class value must be either 1. or 0. when dataset has one label"
                try:
                    blor_model = blor.fit(bdf)
                except Py4JJavaError as e:
                    assert msg in e.java_exception.getMessage()

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


@pytest.mark.compat
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat_wrong_label(
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    caplog: LogCaptureFixture,
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

    num_rows = len(X)
    feature_cols = ["c0", "c1"]
    schema = ["c0 float, c1 float, label float"]

    def test_functor(
        y: np.ndarray, err_msg_spark_v34: str, err_msg_spark_v33: str
    ) -> None:
        with CleanSparkSession() as spark:
            np_array = np.concatenate((X, y.reshape(num_rows, 1)), axis=1)

            df = spark.createDataFrame(
                np_array.tolist(),
                ",".join(schema),
            )

            df = df.withColumn("features", array_to_vector(array(*feature_cols))).drop(
                *feature_cols
            )

            lr = _LogisticRegression(standardization=False)

            try:
                lr.fit(df)
                assert False, "There should be a java exception"
            except Py4JJavaError as e:
                java_msg = e.java_exception.getMessage()
                assert err_msg_spark_v34 in java_msg or err_msg_spark_v33 in java_msg

    # negative label
    wrong_label = -1.1
    y = np.array([1.0, 0.0, wrong_label, 2.0])
    spark_v34_msg = f"Labels MUST be in [0, 2147483647), but got {wrong_label}"
    spark_v33_msg = (
        f"Classification labels should be in [0 to 2]. Found 1 invalid labels."
    )
    test_functor(y, spark_v34_msg, spark_v33_msg)

    # non-integer label
    wrong_label = 0.4
    y = np.array([1.0, 0.0, wrong_label, 2.0])
    spark_v34_msg = f"Labels MUST be Integers, but got {wrong_label}"
    spark_v33_msg = (
        f"Classification labels should be in [0 to 2]. Found 1 invalid labels."
    )
    test_functor(y, spark_v34_msg, spark_v33_msg)


def compare_model(
    gpu_model: LogisticRegressionModel,
    cpu_model: SparkLogisticRegressionModel,
    df_test: DataFrame,
    unit_tol: float = 1e-4,
    total_tol: float = 0.0,
    accuracy_and_probability_only: bool = False,
) -> Tuple[LogisticRegressionModel, SparkLogisticRegressionModel]:
    gpu_res = gpu_model.transform(df_test).collect()

    cpu_res = cpu_model.transform(df_test).collect()

    # compare accuracy
    gpu_pred = [row["prediction"] for row in gpu_res]
    cpu_pred = [row["prediction"] for row in cpu_res]
    ytest_true = [row["label"] for row in df_test.select(["label"]).collect()]
    from sklearn.metrics import accuracy_score

    gpu_acc = accuracy_score(ytest_true, gpu_pred)
    cpu_acc = accuracy_score(ytest_true, cpu_pred)
    assert gpu_acc >= cpu_acc or abs(gpu_acc - cpu_acc) < 1e-3

    # compare probability column
    gpu_prob = [row["probability"].toArray().tolist() for row in gpu_res]
    cpu_prob = [row["probability"].toArray().tolist() for row in cpu_res]

    assert array_equal(gpu_prob, cpu_prob, unit_tol, total_tol)

    if accuracy_and_probability_only:
        return (gpu_model, cpu_model)

    # compare rawPrediction column
    gpu_rawpred = [row["rawPrediction"].toArray().tolist() for row in gpu_res]
    cpu_rawpred = [row["rawPrediction"].toArray().tolist() for row in cpu_res]
    assert array_equal(gpu_rawpred, cpu_rawpred, unit_tol, total_tol)

    # compare coefficients
    assert array_equal(
        gpu_model.coefficientMatrix.toArray(),
        cpu_model.coefficientMatrix.toArray(),
        unit_tol=unit_tol,
    )
    assert array_equal(
        gpu_model.interceptVector.toArray(),
        cpu_model.interceptVector.toArray(),
        unit_tol=unit_tol,
    )

    return (gpu_model, cpu_model)


@pytest.mark.compat
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_compat_sparse_binomial(
    fit_intercept: bool,
) -> None:
    tolerance = 0.001

    with CleanSparkSession() as spark:
        data = [
            Row(
                label=1.0, weight=1.0, features=Vectors.sparse(3, {2: 1.0})
            ),  # (0., 0., 1.)
            Row(
                label=1.0, weight=1.0, features=Vectors.dense([0.0, 1.0, 0.0])
            ),  # (0., 1., 0.)
            Row(
                label=0.0, weight=1.0, features=Vectors.sparse(3, {0: 1.0})
            ),  # (1., 0., 0.)
            Row(
                label=0.0, weight=1.0, features=Vectors.sparse(3, {0: 2.0, 2: -1.0})
            ),  # (2., 0., -1.)
        ]

        bdf = spark.createDataFrame(data)

        params: Dict[str, Any] = {
            "regParam": 0.1,
            "fitIntercept": fit_intercept,
            "standardization": False,
            "featuresCol": "features",
            "labelCol": "label",
        }

        gpu_lr = LogisticRegression(**params)
        assert gpu_lr.hasParam("enable_sparse_data_optim") is True
        assert gpu_lr.getOrDefault("enable_sparse_data_optim") == None

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            err_msg = "Cannot import pyspark `unwrap_udt` function. Please install pyspark>=3.4 "
            "or run on Databricks Runtime."
            with pytest.raises(RuntimeError, match=err_msg):
                gpu_lr.fit(bdf)
            return

        check_sparse_estimator_preprocess(gpu_lr, bdf, dimension=3)

        gpu_model = gpu_lr.fit(bdf)
        check_sparse_model_preprocess(gpu_model, bdf)

        cpu_lr = SparkLogisticRegression(**params)
        cpu_model = cpu_lr.fit(bdf)
        compare_model(gpu_model, cpu_model, bdf)


@pytest.mark.compat
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_compat_sparse_multinomial(
    fit_intercept: bool,
) -> None:
    with CleanSparkSession() as spark:
        data = [
            Row(
                label=1.0, weight=1.0, features=Vectors.sparse(3, {2: 1.0})
            ),  # (0., 0., 1.)
            Row(
                label=1.0, weight=1.0, features=Vectors.sparse(3, {1: 1.0})
            ),  # (0., 1., 0.)
            Row(
                label=0.0, weight=1.0, features=Vectors.sparse(3, {0: 1.0})
            ),  # (1., 0., 0.)
            Row(
                label=2.0, weight=1.0, features=Vectors.sparse(3, {0: 2.0, 2: -1.0})
            ),  # (2., 0., -1.)
        ]

        mdf = spark.createDataFrame(data)

        params: Dict[str, Any] = {
            "regParam": 0.1,
            "fitIntercept": fit_intercept,
            "standardization": False,
            "featuresCol": "features",
            "labelCol": "label",
        }

        gpu_lr = LogisticRegression(**params)
        assert gpu_lr.hasParam("enable_sparse_data_optim") is True
        assert gpu_lr.getOrDefault("enable_sparse_data_optim") == None

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            err_msg = "Cannot import pyspark `unwrap_udt` function. Please install pyspark>=3.4 "
            "or run on Databricks Runtime."
            with pytest.raises(RuntimeError, match=err_msg):
                gpu_lr.fit(mdf)
            return

        gpu_model = gpu_lr.fit(mdf)

        cpu_lr = SparkLogisticRegression(**params)
        cpu_model = cpu_lr.fit(mdf)
        compare_model(gpu_model, cpu_model, mdf)

        for value in {True, False}:
            gpu_lr = LogisticRegression(enable_sparse_data_optim=value, **params)
            assert gpu_lr.getOrDefault("enable_sparse_data_optim") == value
            gpu_model = gpu_lr.fit(mdf)
            compare_model(gpu_model, cpu_model, mdf)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("standardization", [True, False])
@pytest.mark.slow
def test_sparse_nlp20news(
    fit_intercept: bool,
    standardization: bool,
    caplog: LogCaptureFixture,
) -> None:
    if version.parse(pyspark.__version__) < version.parse("3.4.0"):
        import logging

        err_msg = (
            "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function. "
        )
        "The test case will be skipped. Please install pyspark>=3.4."
        logging.info(err_msg)
        return

    tolerance = 0.001
    reg_param = 1e-2

    from pyspark.ml.feature import CountVectorizer, RegexTokenizer
    from sklearn.datasets import fetch_20newsgroups

    try:
        twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
    except:
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    X = twenty_train.data
    y = twenty_train.target.tolist()

    conf: Dict[str, Any] = {
        "spark.rapids.ml.uvm.enabled": True
    }  # enable memory management to run the test case on GPU with small memory (e.g. 2G)
    with CleanSparkSession(conf) as spark:
        data = [
            Row(
                label=y[i],
                weight=1.0,
                text=X[i],
            )
            for i in range(len(X))
        ]
        df = spark.createDataFrame(data)
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens")
        df = tokenizer.transform(df)

        cv = CountVectorizer(inputCol="tokens", outputCol="features")
        cv_model = cv.fit(df)
        df = cv_model.transform(df)

        df_train, df_test = df.randomSplit([0.8, 0.2])

        gpu_lr = LogisticRegression(
            enable_sparse_data_optim=True,
            verbose=6,
            regParam=reg_param,
            fitIntercept=fit_intercept,
            standardization=standardization,
            featuresCol="features",
            labelCol="label",
        )

        cpu_lr = SparkLogisticRegression(
            regParam=reg_param,
            fitIntercept=fit_intercept,
            standardization=standardization,
            featuresCol="features",
            labelCol="label",
        )

        gpu_model = gpu_lr.fit(df_train)

        cpu_model = cpu_lr.fit(df_train)
        cpu_objective = cpu_model.summary.objectiveHistory[-1]

        assert (
            gpu_model.objective < cpu_objective
            or abs(gpu_model.objective - cpu_objective) < tolerance
        )

        if standardization is True:
            compare_model(
                gpu_model,
                cpu_model,
                df_train,
                unit_tol=tolerance,
                total_tol=tolerance,
                accuracy_and_probability_only=True,
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
@pytest.mark.slow
def test_quick_sparse(
    fit_intercept: bool,
    reg_factors: Tuple[float, float],
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
    float32_inputs: bool = True,
) -> None:
    if version.parse(pyspark.__version__) < version.parse("3.4.0"):
        import logging

        err_msg = (
            "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function. "
        )
        "The test case will be skipped. Please install pyspark>=3.4."
        logging.info(err_msg)
        return

    convert_to_sparse = True
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
        convert_to_sparse=convert_to_sparse,
        set_float32_inputs=float32_inputs,
    )


@pytest.mark.parametrize("metric_name", ["accuracy", "logLoss", "areaUnderROC"])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
def test_sparse_crossvalidator_logistic_regression(
    metric_name: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
) -> None:
    test_crossvalidator_logistic_regression(
        metric_name=metric_name,
        feature_type=feature_types.vector,
        data_type=data_type,
        data_shape=data_shape,
        convert_to_sparse=True,
    )


@pytest.mark.compat
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize(
    "lr_types",
    [
        (SparkLogisticRegression, SparkLogisticRegressionModel),
        (LogisticRegression, LogisticRegressionModel),
    ],
)
def test_compat_standardization(
    fit_intercept: bool,
    data_type: np.dtype,
    lr_types: Tuple[LogisticRegressionType, LogisticRegressionModelType],
    caplog: LogCaptureFixture,
) -> None:
    _LogisticRegression, _LogisticRegressionModel = lr_types
    tolerance = 1e-3

    X, _, y, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=10000,
        ncols=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
    )

    X[:, 0] *= 1000  # Scale up the first features by 1000
    X[:, 0] += 50  # Shift the first features by 50

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

        blor = _LogisticRegression(
            regParam=0.01, fitIntercept=fit_intercept, standardization=True
        )

        if isinstance(blor, SparkLogisticRegression):
            blor.setWeightCol("weight")

        blor_model = blor.fit(bdf)

        blor_model.setFeaturesCol("features")
        blor_model.setProbabilityCol("newProbability")
        blor_model.setRawPredictionCol("newRawPrediction")

        if fit_intercept is False:
            array_equal(
                blor_model.coefficients.toArray(),
                [-1.59550205e-04, 1.35555146e00],
                tolerance,
            )
            array_equal(
                blor_model.coefficientMatrix.toArray(),
                [-1.59550205e-04, 1.35555146e00],
                tolerance,
            )
            assert blor_model.intercept == 0.0
            assert blor_model.interceptVector.toArray() == [0.0]
        else:
            array_equal(
                blor_model.coefficients.toArray(),
                [-1.63432342e-04, 1.35951030e00],
                tolerance,
            )
            array_equal(
                blor_model.coefficientMatrix.toArray(),
                [-1.63432342e-04, 1.35951030e00],
                tolerance,
            )
            assert array_equal([blor_model.intercept], [-0.05060137], tolerance)
            assert array_equal(
                blor_model.interceptVector.toArray(), [-0.05060137], tolerance
            )


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_factors", [(0.0, 0.0), (0.1, 0.0), (0.1, 1.0), (0.1, 0.2)]
)
@pytest.mark.parametrize("feature_type", ["vector"])
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("max_record_batch", [20])
@pytest.mark.parametrize("ncols_nclasses", [(2, 2), (4, 3), (4, 4)])
@pytest.mark.slow
def test_standardization(
    fit_intercept: bool,
    reg_factors: Tuple[float, float],
    feature_type: str,
    data_type: np.dtype,
    max_record_batch: int,
    ncols_nclasses: Tuple[int, int],
    gpu_number: int,
    float32_inputs: bool = True,
) -> None:
    tolerance = 0.001
    reg_param = reg_factors[0]
    elasticNet_param = reg_factors[1]
    n_rows = 10000
    n_cols = ncols_nclasses[0]
    n_classes = ncols_nclasses[1]

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=n_rows,
        ncols=n_cols,
        n_classes=n_classes,
        n_informative=n_cols,
        n_redundant=0,
        n_repeated=0,
    )
    X_train[:, 0] *= 1000  # Scale up the first features by 1000
    X_train[:, 0] += 50  # Shift the first features by 50

    X_test[:, 0] *= 1000  # Scale up the first features by 1000
    X_test[:, 0] += 50  # Shift the first features by 50

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )
        test_df, _, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X_test, y_test
        )

        assert label_col is not None

        def train_model(EstimatorClass, ModelClass):  # type: ignore
            estimator = EstimatorClass(
                standardization=True,
                fitIntercept=fit_intercept,
                regParam=reg_param,
                elasticNetParam=elasticNet_param,
            )

            if isinstance(estimator, LogisticRegression):
                estimator._float32_inputs = float32_inputs

            estimator.setFeaturesCol(features_col)
            estimator.setLabelCol(label_col)
            model = estimator.fit(train_df)

            preds = model.transform(train_df).collect()
            y_preds = [row[label_col] for row in preds]
            from sklearn.metrics import accuracy_score

            train_acc = accuracy_score(y_train, y_preds)

            preds = model.transform(test_df).collect()
            y_preds = [row[label_col] for row in preds]
            test_acc = accuracy_score(y_test, y_preds)

            return (estimator, model, train_acc, test_acc)

        mg, mg_model, mg_train_acc, mg_test_acc = train_model(
            LogisticRegression, LogisticRegressionModel
        )
        mc, mc_model, mc_train_acc, mc_test_acc = train_model(
            SparkLogisticRegression, SparkLogisticRegressionModel
        )

        assert array_equal(
            mg_model.coefficientMatrix.toArray(),
            mc_model.coefficientMatrix.toArray(),
            tolerance,
        )
        assert array_equal(
            mg_model.interceptVector.toArray(),
            mc_model.interceptVector.toArray(),
            tolerance,
        )
        assert (
            mg_train_acc > mc_train_acc or abs(mg_train_acc - mc_train_acc) < tolerance
        )
        assert mg_test_acc > mc_test_acc or abs(mg_test_acc - mc_test_acc) < tolerance


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_factors",
    [(0.0, 0.0), (0.1, 0.0), (0.1, 1.0), (0.1, 0.2)],
)
def test_standardization_sparse_example(
    fit_intercept: bool,
    reg_factors: Tuple[float, float],
    float32_inputs: bool = False,
) -> None:
    _convert_index = "int32" if random.choice([True, False]) is True else "int64"

    if version.parse(pyspark.__version__) < version.parse("3.4.0"):
        import logging

        err_msg = (
            "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function. "
        )
        "The test case will be skipped. Please install pyspark>=3.4."
        logging.info(err_msg)
        return

    tolerance = 0.001
    # Compare accuracy and probability only when regularizaiton is disabled.
    # It is observed that no regularization leads to large absolute values of coefficients, and
    # therefore large difference of GPU and CPU in raw Predictions (e.g. 23.1068 v.s. 27.6741)
    # and in coefficients (e.g. -23.57752037 v.s. -28.48549335).
    accuracy_and_probability_only = True if reg_factors[0] == 0.0 else False

    datatype = np.float32

    est_params: Dict[str, Any] = {
        "standardization": True,
        "regParam": reg_factors[0],
        "elasticNetParam": reg_factors[1],
        "fitIntercept": fit_intercept,
        "featuresCol": "features",
        "labelCol": "label",
    }

    def prepare_csr_matrix_and_y() -> Tuple[csr_matrix, List[float]]:
        X_origin = np.array(
            [
                [-1.1258, 0.0000, 0.0000, -0.4339, 0.0000],
                [-1.5551, -0.3414, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.2660, 0.0000, 0.0000, 0.9463],
                [-0.8437, 0.0000, 1.2590, 0.0000, 0.0000],
            ],
            datatype,
        )

        X_origin = np.ascontiguousarray(X_origin.T)

        X = csr_matrix(X_origin)
        assert X.nnz == 8 and X.shape == (5, 4)
        y = [0.0, 1.0, 2.0, 0.0, 1.0]
        return X, y

    X, y = prepare_csr_matrix_and_y()

    conf = {
        "spark.rapids.ml.uvm.enabled": True
    }  # enable memory management to run the test case on GPU with small memory (e.g. 2G)
    with CleanSparkSession(conf) as spark:

        def sparse_to_df(X: csr_matrix, y: List[float]) -> DataFrame:
            assert X.shape[0] == len(y)
            dimension = X.shape[1]
            data = [
                Row(
                    features=SparseVector(dimension, X[i].indices, X[i].data),
                    label=y[i],
                )
                for i in range(len(y))
            ]
            df = spark.createDataFrame(data)

            return df

        df = sparse_to_df(X, y)

        gpu_lr = LogisticRegression(float32_inputs=float32_inputs, **est_params)
        cpu_lr = SparkLogisticRegression(**est_params)

        # _convert_index is used for converting input csr sparse matrix to gpu SparseCumlArray for calling cuml C++ layer. If not None, cuml converts the dtype of indices array and indptr array to the value of _convert_index (e.g. 'int64').
        gpu_lr.cuml_params["_convert_index"] = _convert_index
        gpu_model = gpu_lr.fit(df)
        assert hasattr(gpu_lr, "_index_dtype") and (
            gpu_lr._index_dtype == _convert_index
        )

        cpu_model = cpu_lr.fit(df)

        compare_model(
            gpu_model,
            cpu_model,
            df,
            tolerance,
            accuracy_and_probability_only=accuracy_and_probability_only,
        )


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_factors", [(0.0, 0.0), (0.1, 0.0), (0.1, 1.0), (0.1, 0.2)]
)
@pytest.mark.parametrize("feature_type", ["vector"])
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
@pytest.mark.parametrize("max_record_batch", [20])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.slow
def test_double_precision(
    fit_intercept: bool,
    reg_factors: Tuple[float, float],
    feature_type: str,
    data_shape: Tuple[int, int],
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
) -> None:

    random_bool = random.choice([True, False])
    data_type = np.float32 if random_bool is True else np.float64
    float32_inputs = random.choice([True, False])

    test_quick(
        fit_intercept=fit_intercept,
        reg_factors=reg_factors,
        feature_type=feature_type,
        data_shape=data_shape,
        data_type=data_type,
        max_record_batch=max_record_batch,
        n_classes=n_classes,
        gpu_number=gpu_number,
        float32_inputs=float32_inputs,
    )

    test_quick_sparse(
        fit_intercept=fit_intercept,
        reg_factors=reg_factors,
        feature_type=feature_type,
        data_shape=data_shape,
        data_type=data_type,
        max_record_batch=max_record_batch,
        n_classes=n_classes,
        gpu_number=gpu_number,
        float32_inputs=float32_inputs,
    )


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("reg_factors", [(0.0, 0.0), (0.1, 0.2)])
def test_quick_double_precision(
    fit_intercept: bool,
    reg_factors: Tuple[float, float],
    gpu_number: int,
) -> None:

    data_type = np.float64
    float32_inputs = False

    feature_type = "vector"
    max_record_batch = 20
    ncols_nclasses = (3, 4)

    test_standardization(
        fit_intercept=fit_intercept,
        reg_factors=reg_factors,
        feature_type=feature_type,
        data_type=data_type,
        max_record_batch=max_record_batch,
        ncols_nclasses=ncols_nclasses,
        gpu_number=gpu_number,
        float32_inputs=float32_inputs,
    )

    test_standardization_sparse_example(
        fit_intercept=fit_intercept,
        reg_factors=reg_factors,
        float32_inputs=float32_inputs,
    )


@pytest.mark.slow
def test_sparse_int64() -> None:
    from spark_rapids_ml.core import col_name_unique_tag

    output_data_dir = f"/tmp/spark_rapids_ml_{col_name_unique_tag}"
    gpu_number = 1
    data_shape = (int(1e5), 2200)
    fraction_sampled_for_test = (
        1.0 if data_shape[0] <= 100000 else 100000 / data_shape[0]
    )
    n_classes = 8
    tolerance = 0.001
    est_params: Dict[str, Any] = {
        "regParam": 0.02,
        "maxIter": 10,
        "standardization": False,  # reduce GPU memory since standardization copies the value array
    }
    density = 0.1

    data_gen_args = [
        "--n_informative",
        f"{math.ceil(data_shape[1] / 3)}",
        "--num_rows",
        str(data_shape[0]),
        "--num_cols",
        str(data_shape[1]),
        "--dtype",
        "float64",
        "--feature_type",
        "vector",
        "--output_dir",
        output_data_dir,
        "--n_classes",
        str(n_classes),
        "--random_state",
        "0",
        "--logistic_regression",
        "True",
        "--density",
        str(density),
        "--use_gpu",
        "True",
    ]

    from . import conftest

    data_gen = SparseRegressionDataGen(data_gen_args)
    df, _, _ = data_gen.gen_dataframe_and_meta(conftest._spark)

    # ensure same dataset for comparing CPU and GPU
    df.cache()

    # convert index dtype to int64 for testing purpose
    gpu_est = LogisticRegression(num_workers=gpu_number, verbose=True, **est_params)
    gpu_est.cuml_params["_convert_index"] = "int64"

    gpu_model = gpu_est.fit(df)
    assert hasattr(gpu_est, "_index_dtype") and (gpu_est._index_dtype == "int64")

    # compare gpu with spark cpu
    cpu_est = SparkLogisticRegression(**est_params)
    cpu_model = cpu_est.fit(df)
    cpu_objective = cpu_model.summary.objectiveHistory[-1]
    assert (
        gpu_model.objective < cpu_objective
        or abs(gpu_model.objective - cpu_objective) < tolerance
    )

    df_test = df.sample(fraction=fraction_sampled_for_test, seed=0)
    compare_model(
        gpu_model,
        cpu_model,
        df_test,
        unit_tol=tolerance,
        total_tol=tolerance,
        accuracy_and_probability_only=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("standardization", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_sparse_all_zeroes(
    standardization: bool,
    fit_intercept: bool,
) -> None:
    tolerance = 0.001

    with CleanSparkSession() as spark:
        data = [
            Row(label=1.0, features=Vectors.sparse(2, {})),
            Row(label=1.0, features=Vectors.sparse(2, {})),
            Row(label=0.0, features=Vectors.sparse(2, {})),
            Row(label=0.0, features=Vectors.sparse(2, {})),
        ]

        bdf = spark.createDataFrame(data)

        params: Dict[str, Any] = {
            "regParam": 0.1,
            "fitIntercept": fit_intercept,
            "standardization": standardization,
            "featuresCol": "features",
            "labelCol": "label",
        }

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            return

        gpu_lr = LogisticRegression(enable_sparse_data_optim=True, **params)
        gpu_model = gpu_lr.fit(bdf)
        check_sparse_model_preprocess(gpu_model, bdf)

        cpu_lr = SparkLogisticRegression(**params)
        cpu_model = cpu_lr.fit(bdf)
        compare_model(gpu_model, cpu_model, bdf)


@pytest.mark.parametrize("standardization", [True])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_sparse_one_gpu_all_zeroes(
    standardization: bool,
    fit_intercept: bool,
    gpu_number: int,
) -> None:
    tolerance = 0.001

    if gpu_number < 2:
        pytest.skip(reason="test_sparse_one_gpu_zeroes requires at least 2 GPUs")
    gpu_number = 2

    with CleanSparkSession() as spark:
        data = [
            Row(label=1.0, features=Vectors.sparse(2, {0: 10.0, 1: 20.0})),
            Row(label=1.0, features=Vectors.sparse(2, {})),
            Row(label=0.0, features=Vectors.sparse(2, {})),
            Row(label=0.0, features=Vectors.sparse(2, {})),
        ]

        bdf = spark.createDataFrame(data)

        params: Dict[str, Any] = {
            "regParam": 0.1,
            "fitIntercept": fit_intercept,
            "standardization": standardization,
            "featuresCol": "features",
            "labelCol": "label",
        }

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            return

        gpu_lr = LogisticRegression(
            enable_sparse_data_optim=True, verbose=True, **params
        )
        gpu_model = gpu_lr.fit(bdf)
        check_sparse_model_preprocess(gpu_model, bdf)

        cpu_lr = SparkLogisticRegression(**params)
        cpu_model = cpu_lr.fit(bdf)
        compare_model(gpu_model, cpu_model, bdf)

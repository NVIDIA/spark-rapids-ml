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

from typing import Tuple, Type, TypeVar, Union

import numpy as np
import pytest
from cuml import accuracy_score
from pyspark.ml.classification import (
    RandomForestClassificationModel as SparkRFClassificationModel,
)
from pyspark.ml.classification import RandomForestClassifier as SparkRFClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressionModel as SparkRFRegressionModel
from pyspark.ml.regression import RandomForestRegressor as SparkRFRegressor
from sklearn.metrics import r2_score

from spark_rapids_ml.classification import (
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from spark_rapids_ml.regression import (
    RandomForestRegressionModel,
    RandomForestRegressor,
)

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types,
    get_default_cuml_parameters,
    idfn,
    make_classification_dataset,
    make_regression_dataset,
    pyspark_supported_feature_types,
)

RandomForest = TypeVar(
    "RandomForest", Type[RandomForestClassifier], Type[RandomForestRegressor]
)
RandomForestModel = TypeVar(
    "RandomForestModel",
    Type[RandomForestClassificationModel],
    Type[RandomForestRegressionModel],
)

RandomForestType = TypeVar(
    "RandomForestType",
    Type[SparkRFClassifier],
    Type[SparkRFRegressor],
    Type[RandomForestClassifier],
    Type[RandomForestRegressor],
)
RandomForestModelType = TypeVar(
    "RandomForestModelType",
    Type[SparkRFClassificationModel],
    Type[SparkRFRegressionModel],
    Type[RandomForestClassificationModel],
    Type[RandomForestRegressionModel],
)


@pytest.mark.parametrize("Estimator", [RandomForestClassifier, RandomForestRegressor])
def test_default_cuml_params(Estimator: RandomForest) -> None:
    from cuml.ensemble.randomforest_common import BaseRandomForestModel

    cuml_params = get_default_cuml_parameters(
        [BaseRandomForestModel],
        [
            "handle",
            "output_type",
            "accuracy_metric",
            "dtype",
            "criterion",
            "min_weight_fraction_leaf",
            "max_leaf_nodes",
            "min_impurity_split",
            "oob_score",
            "n_jobs",
            "warm_start",
            "class_weight",
        ],
    )
    spark_params = Estimator()._get_cuml_params_default()
    assert cuml_params == spark_params


@pytest.mark.parametrize("RFEstimator", [RandomForestClassifier, RandomForestRegressor])
def test_random_forest_params(tmp_path: str, RFEstimator: RandomForest) -> None:
    # Default params
    default_spark_params = {
        "maxBins": 32,
        "maxDepth": 5,
        "numTrees": 20,
        "bootstrap": True,
        "featureSubsetStrategy": "auto",
    }
    default_cuml_params = {
        "n_bins": 32,
        "n_estimators": 20,
        "max_depth": 5,
        "bootstrap": True,
        "max_features": "auto",
    }
    est = RFEstimator()
    assert_params(est, default_spark_params, default_cuml_params)

    # Spark ML Params
    spark_params = {
        "maxBins": 17,
        "maxDepth": 9,
        "numTrees": 17,
        "featureSubsetStrategy": "onethird",
    }
    est = RFEstimator(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(
        {
            "n_bins": 17,
            "max_depth": 9,
            "n_estimators": 17,
            "max_features": 1 / 3.0,
        }
    )
    assert_params(est, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/random_forest_classifier_tests"
    estimator_path = f"{path}/random_forest_classifier_tests"
    est.write().overwrite().save(estimator_path)
    loaded_est = RandomForestClassifier.load(estimator_path)
    assert_params(loaded_est, expected_spark_params, expected_cuml_params)

    if RFEstimator == RandomForestRegressor:
        est = RFEstimator(impurity="variance")
        est.cuml_params["split_criterion"] == "mse"


rf_est_model_classes = [
    # (estimator, model, n_classes)
    (RandomForestClassifier, RandomForestClassificationModel, 2),
    (RandomForestClassifier, RandomForestClassificationModel, 4),
    (RandomForestRegressor, RandomForestRegressionModel, -1),
]


@pytest.mark.parametrize("est_model_classes", rf_est_model_classes, ids=idfn)
@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("data_shape", [(100, 8)], ids=idfn)
def test_random_forest_basic(
    tmp_path: str,
    est_model_classes: Tuple[RandomForest, RandomForestModel, int],
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
) -> None:
    RFEstimator, RFEstimatorModel, n_classes = est_model_classes

    # Train a toy model
    if RFEstimator == RandomForestClassifier:
        X, _, y, _ = make_classification_dataset(
            datatype=data_type,
            nrows=data_shape[0],
            ncols=data_shape[1],
            n_classes=n_classes,
            n_informative=8,
            n_redundant=0,
            n_repeated=0,
        )
    else:
        X, _, y, _ = make_regression_dataset(
            datatype=data_type,
            nrows=data_shape[0],
            ncols=data_shape[1],
        )

    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )

        est = RFEstimator()

        est.setFeaturesCol(features_col)
        assert est.getFeaturesCol() == features_col

        assert label_col is not None
        est.setLabelCol(label_col)
        assert est.getLabelCol() == label_col

        def assert_model(lhs: RandomForestModel, rhs: RandomForestModel) -> None:
            assert lhs.cuml_params == rhs.cuml_params

            # Vector type will be cast to array(double)
            if feature_type == "vector":
                assert lhs.dtype == np.dtype(np.float64).name
            else:
                assert lhs.dtype == np.dtype(data_type).name

            assert lhs.dtype == rhs.dtype
            assert lhs.n_cols == rhs.n_cols
            assert lhs.n_cols == data_shape[1]

            if isinstance(lhs, RandomForestClassificationModel):
                assert lhs.numClasses == rhs.numClasses
                assert lhs.numClasses == n_classes

        # train a model
        model = est.fit(df)

        # model persistence
        path = tmp_path + "/random_forest_tests"
        model_path = f"{path}/random_forest_tests"
        model.write().overwrite().save(model_path)

        model_loaded = RFEstimatorModel.load(model_path)
        assert_model(model, model_loaded)


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
@pytest.mark.parametrize("RFEstimator", [RandomForestClassifier, RandomForestRegressor])
def test_random_forest_numeric_type(
    gpu_number: int, RFEstimator: RandomForest, data_type: str
) -> None:
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
        lr = RFEstimator(num_workers=gpu_number)
        lr.setFeaturesCol(feature_cols)
        lr.fit(df)


from .conftest import _gpu_number

if _gpu_number > 1:
    num_workers = [1, _gpu_number]
else:
    num_workers = [1]


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("num_workers", num_workers)
@pytest.mark.slow
def test_random_forest_classifier(
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    num_workers: int,
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

    rf_params = {
        "n_estimators": 100,
        "n_bins": 128,
        "max_depth": 16,
        "bootstrap": False,
        "max_features": 1.0,
    }

    from cuml import RandomForestClassifier as cuRf

    cu_rf = cuRf(n_streams=1, **rf_params)
    cu_rf.fit(X_train, y_train)
    cu_preds = cu_rf.predict(X_test)
    cu_preds_proba = cu_rf.predict_proba(X_test)

    cu_acc = accuracy_score(y_test, cu_preds)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_rf = RandomForestClassifier(
            num_workers=num_workers,
            **rf_params,
        )
        spark_rf.setFeaturesCol(features_col)
        spark_rf.setLabelCol(label_col)
        spark_rf_model = spark_rf.fit(train_df)

        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = spark_rf_model.transform(test_df).collect()
        pred_result = [row.prediction for row in result]

        if feature_type == feature_types.vector:
            # no need to compare all feature type.
            spark_cpu_result = spark_rf_model.cpu().transform(test_df).collect()
            spark_cpu_pred_result = [row.prediction for row in spark_cpu_result]
            assert array_equal(spark_cpu_pred_result, pred_result)

        spark_acc = accuracy_score(y_test, np.array(pred_result))

        # Since vector type will force to convert to array<double>
        # which may cause precision issue for random forest.
        if num_workers == 1 and not (
            data_type == np.float32 and feature_type == feature_types.vector
        ):
            assert cu_acc == spark_acc

            pred_proba_result = [row.probability for row in result]
            np.testing.assert_allclose(pred_proba_result, cu_preds_proba, rtol=1e-3)
        else:
            assert cu_acc - spark_acc < 0.07


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize("num_workers", num_workers)
@pytest.mark.slow
def test_random_forest_regressor(
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    num_workers: int,
) -> None:
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
    )

    rf_params = {
        "n_estimators": 100,
        "n_bins": 128,
        "max_depth": 16,
        "bootstrap": False,
        "max_features": 1.0,
        "random_state": 1.0,
    }

    from cuml import RandomForestRegressor as cuRf

    cu_rf = cuRf(n_streams=1, **rf_params)
    cu_rf.fit(X_train, y_train)
    cu_preds = cu_rf.predict(X_test)

    cu_acc = r2_score(y_test, cu_preds)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_rf = RandomForestRegressor(
            num_workers=num_workers,
            **rf_params,
        )
        spark_rf.setFeaturesCol(features_col)
        spark_rf.setLabelCol(label_col)
        spark_rf_model = spark_rf.fit(train_df)

        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = spark_rf_model.transform(test_df).collect()
        pred_result = [row.prediction for row in result]

        if feature_type == feature_types.vector:
            # no need to compare all feature type.
            spark_cpu_result = spark_rf_model.cpu().transform(test_df).collect()
            spark_cpu_pred_result = [row.prediction for row in spark_cpu_result]
            assert array_equal(spark_cpu_pred_result, pred_result)

        spark_acc = r2_score(y_test, np.array(pred_result))

        # Since vector type will force to convert to array<double>
        # which may cause precision issue for random forest.
        if num_workers == 1 and not (
            data_type == np.float32 and feature_type == feature_types.vector
        ):
            assert pytest.approx(cu_acc) == spark_acc
        else:
            assert cu_acc - spark_acc < 0.09


@pytest.mark.parametrize("rf_type", [RandomForestClassifier, RandomForestRegressor])
@pytest.mark.parametrize(
    "feature_subset", ["auto", "all", "0.85", "2", "onethird", "log2", "sqrt", "foo"]
)
def test_random_forest_featuresubset(
    rf_type: RandomForestType,
    feature_subset: str,
) -> None:
    with CleanSparkSession() as spark:
        df = spark.createDataFrame(
            [
                (1.0, Vectors.dense(1.0, 0.0)),
                (1.0, Vectors.dense(0.8, 1.0)),
                (0.0, Vectors.dense(0.2, 0.8)),
                (0.0, Vectors.sparse(2, [1], [1.0])),
                (1.0, Vectors.dense(1.0, 0.0)),
                (1.0, Vectors.dense(0.8, 1.0)),
                (0.0, Vectors.dense(0.2, 0.8)),
                (0.0, Vectors.sparse(2, [1], [1.0])),
            ],
            ["label", "features"],
        )

        if feature_subset != "foo":
            rf = rf_type(
                numTrees=3,
                maxDepth=2,
                labelCol="label",
                seed=42,
                featureSubsetStrategy=feature_subset,
            )
            m = rf.fit(df)
        else:
            with pytest.raises(ValueError):
                rf = rf_type(
                    numTrees=3,
                    maxDepth=2,
                    labelCol="label",
                    seed=42,
                    featureSubsetStrategy=feature_subset,
                )


@pytest.mark.compat
@pytest.mark.parametrize(
    "rf_types",
    [
        (SparkRFClassifier, SparkRFClassificationModel),
        (RandomForestClassifier, RandomForestClassificationModel),
    ],
)
@pytest.mark.parametrize("impurity", ["gini", "entropy"])
def test_random_forest_classifier_spark_compat(
    rf_types: Tuple[RandomForestType, RandomForestModelType],
    gpu_number: int,
    tmp_path: str,
    impurity: str,
) -> None:
    # based on https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html
    # cuML does not support single feature, so using expanded dataset
    _RandomForestClassifier, _RandomForestClassificationModel = rf_types

    with CleanSparkSession() as spark:
        df = spark.createDataFrame(
            [
                (1.0, Vectors.dense(1.0, 0.0)),
                (1.0, Vectors.dense(0.8, 1.0)),
                (0.0, Vectors.dense(0.2, 0.8)),
                (0.0, Vectors.sparse(2, [1], [1.0])),
                (1.0, Vectors.dense(1.0, 0.0)),
                (1.0, Vectors.dense(0.8, 1.0)),
                (0.0, Vectors.dense(0.2, 0.8)),
                (0.0, Vectors.sparse(2, [1], [1.0])),
            ],
            ["label", "features"],
        )

        rf = _RandomForestClassifier(
            numTrees=3, maxDepth=2, labelCol="label", seed=42, impurity=impurity
        )
        rf.setLeafCol("leafId")
        assert rf.getLeafCol() == "leafId"

        if isinstance(rf, RandomForestClassifier):
            # reduce the number of GPUs for toy dataset to avoid empty partition
            gpu_number = min(gpu_number, 2)
            rf.num_workers = gpu_number
            df = df.repartition(gpu_number)

        assert rf.getMinWeightFractionPerNode() == 0.0
        assert rf.getNumTrees() == 3
        assert rf.getMaxDepth() == 2
        assert rf.getSeed() == 42
        assert rf.getFeaturesCol() == "features"
        assert rf.getLabelCol() == "label"

        model = rf.fit(df)

        assert model.getFeaturesCol() == "features"
        assert model.getLabelCol() == "label"
        assert model.getBootstrap()

        model.setRawPredictionCol("newRawPrediction")
        assert model.getRawPredictionCol() == "newRawPrediction"
        featureImportances = model.featureImportances
        assert np.allclose(model.treeWeights, [1.0, 1.0, 1.0])
        if isinstance(rf, SparkRFClassifier):
            assert featureImportances == Vectors.sparse(2, {0: 1.0})
        else:
            # TODO: investigate difference
            assert featureImportances == Vectors.sparse(2, {})

        test0 = spark.createDataFrame([(Vectors.dense(-1.0, 0.0),)], ["features"])
        example = test0.head()
        if example:
            model.predict(example.features)
            model.predictRaw(example.features)
            model.predictProbability(example.features)

        result = model.transform(test0).head()
        if result:
            if isinstance(model, SparkRFClassificationModel):
                assert result.prediction == 0.0
                assert np.argmax(result.probability) == 0
            else:
                # TODO: investigate difference
                assert result.prediction == 1.0
                assert np.argmax(result.probability) == 1

        if isinstance(model, SparkRFClassificationModel):
            assert np.argmax(result.newRawPrediction) == 0
            assert result.leafId == Vectors.dense([0.0, 0.0, 0.0])
        else:
            with pytest.raises((NotImplementedError, AttributeError)):
                assert np.argmax(result.newRawPrediction) == 0
            with pytest.raises((NotImplementedError, AttributeError)):
                assert result.leafId == Vectors.dense([0.0, 0.0, 0.0])

        test1 = spark.createDataFrame([(Vectors.sparse(2, [0], [1.0]),)], ["features"])
        example = test1.head()
        if example:
            assert model.transform(test1).head().prediction == 1.0

        trees = model.trees
        assert len(trees) == 3

        rfc_path = tmp_path + "/rfc"
        rf.save(rfc_path)
        rf2 = _RandomForestClassifier.load(rfc_path)
        assert rf2.getNumTrees() == 3

        model_path = tmp_path + "/rfc_model"
        model.save(model_path)
        model2 = _RandomForestClassificationModel.load(model_path)
        assert model.transform(test0).take(1) == model2.transform(test0).take(1)
        assert model.featureImportances == model2.featureImportances


@pytest.mark.compat
@pytest.mark.parametrize(
    "rf_types",
    [
        (SparkRFRegressor, SparkRFRegressionModel),
        (RandomForestRegressor, RandomForestRegressionModel),
    ],
)
def test_random_forest_regressor_spark_compat(
    rf_types: Tuple[RandomForestType, RandomForestModelType],
    gpu_number: int,
    tmp_path: str,
) -> None:
    # based on https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html
    # cuML does not support single feature, so using expanded dataset
    _RandomForestRegressor, _RandomForestRegressionModel = rf_types

    with CleanSparkSession() as spark:
        df = spark.createDataFrame(
            [(1.0, Vectors.dense(1.0, 1.0)), (0.0, Vectors.sparse(2, [], []))],
            ["label", "features"],
        )
        rf = _RandomForestRegressor(numTrees=2, maxDepth=2)
        rf.setSeed(42)
        assert rf.getMaxDepth() == 2
        assert rf.getMinWeightFractionPerNode() == 0.0
        assert rf.getNumTrees() == 2
        assert rf.getSeed() == 42

        if isinstance(rf, RandomForestRegressor):
            # force single GPU worker while testing compat
            rf.num_workers = 1

        model = rf.fit(df)
        model.setLeafCol("leafId")

        assert np.allclose(model.treeWeights, [1.0, 1.0])
        if isinstance(model, SparkRFRegressionModel):
            assert model.featureImportances == Vectors.sparse(2, {1: 1.0})
        else:
            # need to investigate
            assert model.featureImportances == Vectors.sparse(2, {})

        assert model.getBootstrap()
        assert model.getSeed() == 42
        assert model.getLeafCol() == "leafId"

        test0 = spark.createDataFrame([(Vectors.dense(-1.0, -1.0),)], ["features"])
        example = test0.head()
        if example:
            assert model.predict(example.features) == 0.0
            assert model.predictLeaf(example.features) == Vectors.dense([0.0, 0.0])

        result = model.transform(test0).head()

        assert result.prediction == 0.0

        assert len(model.trees) == 2

        if isinstance(model, SparkRFRegressionModel):
            assert result.leafId == Vectors.dense([0.0, 0.0])
        else:
            with pytest.raises((NotImplementedError, AttributeError)):
                result.leafId

        assert model.numFeatures == 2
        assert model.getNumTrees == 2  # implemented as a property

        test1 = spark.createDataFrame([(Vectors.sparse(2, [0], [1.0]),)], ["features"])
        result = model.transform(test1).head()
        if result:
            assert result.prediction == 0.0

        rfr_path = tmp_path + "/rfr"
        rf.save(rfr_path)
        rf2 = _RandomForestRegressor.load(rfr_path)
        assert rf2.getNumTrees() == 2  # implemented as a method

        model_path = tmp_path + "/rfr_model"
        model.save(model_path)
        model2 = _RandomForestRegressionModel.load(model_path)

        assert model.featureImportances == model2.featureImportances
        assert model.transform(test0).take(1) == model2.transform(test0).take(1)

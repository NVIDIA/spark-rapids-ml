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

from typing import Tuple

import numpy as np
import pytest
from cuml import accuracy_score

from spark_rapids_ml.classification import (
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from tests.sparksession import CleanSparkSession
from tests.utils import (
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types,
    idfn,
    make_classification_dataset,
    pyspark_supported_feature_types,
)


def test_random_forest_classifier_params(tmp_path: str) -> None:
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
    est = RandomForestClassifier()
    assert_params(est, default_spark_params, default_cuml_params)

    # Spark ML Params
    spark_params = {
        "maxBins": 17,
        "maxDepth": 9,
        "numTrees": 17,
        "featureSubsetStrategy": "onethird",
    }
    est = RandomForestClassifier(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(
        {
            "n_bins": 17,
            "max_depth": 9,
            "n_estimators": 17,
            "max_features": "0.3333",
        }
    )
    assert_params(est, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/random_forest_classifier_tests"
    estimator_path = f"{path}/random_forest_classifier_tests"
    est.write().overwrite().save(estimator_path)
    loaded_est = RandomForestClassifier.load(estimator_path)
    assert_params(loaded_est, expected_spark_params, expected_cuml_params)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("data_shape", [(10, 8)], ids=idfn)
@pytest.mark.parametrize("n_classes", [2, 4])
def test_random_forest_classifier_basic(
    tmp_path: str,
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
    n_classes: int,
) -> None:
    # Train a toy model
    X, _, y, _ = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )
    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )

        est = RandomForestClassifier()

        est.setFeaturesCol(features_col)
        assert est.getFeaturesCol() == features_col

        assert label_col is not None
        est.setLabelCol(label_col)
        assert est.getLabelCol() == label_col

        def assert_model(
            lhs: RandomForestClassificationModel, rhs: RandomForestClassificationModel
        ) -> None:
            assert lhs.cuml_params == rhs.cuml_params

            # Vector type will be cast to array(double)
            if feature_type == "vector":
                assert lhs.dtype == np.dtype(np.float64).name
            else:
                assert lhs.dtype == np.dtype(data_type).name

            assert lhs.dtype == rhs.dtype
            assert lhs.n_cols == rhs.n_cols
            assert lhs.n_cols == data_shape[1]

        # train a model
        model = est.fit(df)

        # model persistence
        path = tmp_path + "/random_forest_classifier_tests"
        model_path = f"{path}/random_forest_classifier_tests"
        model.write().overwrite().save(model_path)

        model_loaded = RandomForestClassificationModel.load(model_path)
        assert_model(model, model_loaded)


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_random_forest_classifier_numeric_type(gpu_number: int, data_type: str) -> None:
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
        lr = RandomForestClassifier(num_workers=gpu_number)
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
        spark_acc = accuracy_score(y_test, np.array(pred_result))

        # Since vector type will force to convert to array<double>
        # which may cause precision issue for random forest.
        if num_workers == 1 and not (
            data_type == np.float32 and feature_type == feature_types.vector
        ):
            assert cu_acc == spark_acc
        else:
            assert cu_acc - spark_acc < 0.07

#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from typing import Any, Dict, Tuple

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spark_rapids_ml.feature import PCA, PCAModel

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types,
    idfn,
    pyspark_supported_feature_types,
)


def test_fit(gpu_number: int) -> None:
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    topk = 1

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["features"])
        )
        gpu_pca = (
            PCA(num_workers=gpu_number, verbose=6).setInputCol("features").setK(topk)
        )
        gpu_model = gpu_pca.fit(df)

        assert gpu_model.getInputCol() == "features"

        assert len(gpu_model.mean_) == 2
        assert gpu_model.mean_[0] == pytest.approx(2.0, 0.001)
        assert gpu_model.mean_[1] == pytest.approx(2.0, 0.001)

        assert len(gpu_model.components_) == 1
        assert len(gpu_model.components_[0]) == 2
        assert gpu_model.components_[0][0] == pytest.approx(0.707, 0.001)
        assert gpu_model.components_[0][1] == pytest.approx(0.707, 0.001)

        assert len(gpu_model.explained_variance_ratio_) == 1
        assert gpu_model.explained_variance_ratio_[0] == pytest.approx(1.0, 0.001)


def test_fit_rectangle(gpu_number: int) -> None:
    data = [[1.0, 1.0], [1.0, 3.0], [5.0, 1.0], [5.0, 3.0]]

    topk = 2

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["coordinates"])
        )

        gpu_pca = PCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)
        gpu_model = gpu_pca.fit(df)

        assert gpu_model.getInputCol() == "coordinates"

        assert len(gpu_model.mean_) == 2
        assert gpu_model.mean_[0] == pytest.approx(3.0, 0.001)
        assert gpu_model.mean_[1] == pytest.approx(2.0, 0.001)

        assert len(gpu_model.components_) == 2

        first_pc = gpu_model.components_[0]
        assert len(first_pc) == 2
        assert first_pc[0] == pytest.approx(1.0, 0.001)
        assert first_pc[1] == pytest.approx(0.0, 0.001)

        second_pc = gpu_model.components_[1]
        assert len(second_pc) == 2
        assert second_pc[0] == pytest.approx(0.0, 0.001)
        assert second_pc[1] == pytest.approx(1.0, 0.001)

        assert len(gpu_model.explained_variance_ratio_) == 2
        assert gpu_model.explained_variance_ratio_[0] == pytest.approx(0.8, 0.001)
        assert gpu_model.explained_variance_ratio_[1] == pytest.approx(0.2, 0.001)


def test_pca_params(gpu_number: int, tmp_path: str) -> None:
    # Default constructor
    default_spark_params = {
        "k": None,
    }
    default_cuml_params = {
        "num_workers": 1,
        "n_components": None,
        "svd_solver": "auto",
        "verbose": False,
        "whiten": False,
    }
    default_pca = PCA()
    assert_params(default_pca, default_spark_params, default_cuml_params)

    # Spark Params constructor
    spark_params = {"k": 2}
    spark_pca = PCA(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)  # type: ignore
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update({"n_components": 2})
    assert_params(spark_pca, expected_spark_params, expected_cuml_params)

    # cuml_params constructor
    cuml_params = {
        "n_components": 5,
        "num_workers": 5,
        "svd_solver": "jacobi",
        "verbose": True,
        "whiten": True,
    }
    cuml_pca = PCA(**cuml_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update({"k": 5})  # type: ignore
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(cuml_params)  # type: ignore
    assert_params(cuml_pca, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/pca_tests"
    estimator_path = f"{path}/pca"
    cuml_pca.write().overwrite().save(estimator_path)
    custom_pca_loaded = PCA.load(estimator_path)
    assert_params(custom_pca_loaded, expected_spark_params, expected_cuml_params)

    # Conflicting params
    conflicting_params = {
        "k": 1,
        "n_components": 2,
    }
    with pytest.raises(ValueError, match="set one or the other"):
        conflicting_pca = PCA(**conflicting_params)


def test_pca_basic(gpu_number: int, tmp_path: str) -> None:
    # Train a PCA model
    data = [[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [5.0, 1.0, 3.9], [5.0, 3.0, 2.9]]
    topk = 2
    path = tmp_path + "/pca_tests"

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["coordinates"])
        )

        gpu_pca = PCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)
        pca_model: PCAModel = gpu_pca.fit(df)

        model_path = f"{path}/pca_model"
        pca_model.write().overwrite().save(model_path)
        pca_model_loaded = PCAModel.load(model_path)

        def assert_pca_model(model: PCAModel, loaded_model: PCAModel) -> None:
            """
            Expect the model attributes are same
            """
            assert model.mean_ == loaded_model.mean_
            assert model.singular_values_ == loaded_model.singular_values_
            assert (
                model.explained_variance_ratio_
                == loaded_model.explained_variance_ratio_
            )
            assert model.components_ == loaded_model.components_
            assert (
                model.cuml_params["n_components"]
                == loaded_model.cuml_params["n_components"]
            )
            assert model.dtype == loaded_model.dtype
            assert model.n_cols == model.n_cols
            assert model.n_cols == 3
            assert model.dtype == "float64"

        assert_pca_model(pca_model, pca_model_loaded)


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_pca_numeric_type(gpu_number: int, data_type: str) -> None:
    data = [
        [1, 4, 4, 4, 0],
        [2, 2, 2, 2, 1],
        [3, 3, 3, 2, 2],
        [3, 3, 3, 2, 3],
        [5, 2, 1, 3, 4],
    ]

    with CleanSparkSession() as spark:
        feature_cols = ["c1", "c2", "c3", "c4", "c5"]
        schema = ", ".join([f"{c} {data_type}" for c in feature_cols])
        df = spark.createDataFrame(data, schema=schema)
        pca = PCA(num_workers=gpu_number, inputCols=feature_cols)
        pca.fit(df)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
def test_pca(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
) -> None:

    X, _ = make_blobs(n_samples=data_shape[0], n_features=data_shape[1], random_state=0)

    from cuml import PCA as cuPCA

    n_components = 3

    cu_pca = cuPCA(n_components=n_components, output_type="numpy", verbose=7)
    cu_model = cu_pca.fit(X)
    # TODO: adding mean to match Spark transform
    # cu_result = cu_model.transform(X + np.array(cu_model.mean_, data_type))
    cu_result = cu_model.transform(X)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )

        output_col = (
            "pca_features"
            if isinstance(features_col, str)
            else ["pca_feature_" + str(i) for i in range(n_components)]
        )

        spark_pca = (
            PCA(n_components=3).setInputCol(features_col).setOutputCol(output_col)
        )

        model = spark_pca.fit(train_df)
        assert array_equal(cu_pca.components_, model.components_, 1e-3, with_sign=False)
        assert array_equal(
            cu_pca.explained_variance_ratio_, model.explained_variance_ratio_, 1e-3
        )
        assert array_equal(cu_pca.mean_, model.mean_, 1e-3)
        assert array_equal(cu_pca.singular_values_, model.singular_values_, 1e-3)
        transform_df = model.transform(train_df)

        if isinstance(output_col, str):
            spark_result = transform_df.collect()
            spark_result = [v[0] for v in spark_result]
        else:
            spark_result = transform_df.toPandas().to_numpy()

        assert array_equal(cu_result, spark_result, 1e-2, with_sign=False)

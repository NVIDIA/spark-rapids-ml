#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import math
from typing import List, Tuple, Union

import cupy as cp
import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from cuml.metrics import trustworthiness
from pyspark.sql.functions import array
from sklearn.datasets import load_digits, load_iris

from spark_rapids_ml.umap import UMAP, UMAPModel

from .sparksession import CleanSparkSession
from .utils import (
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    get_default_cuml_parameters,
    pyspark_supported_feature_types,
)


def _load_dataset(dataset: str, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    if dataset == "digits":
        local_X, local_y = load_digits(return_X_y=True)

    else:  # dataset == "iris"
        local_X, local_y = load_iris(return_X_y=True)

    local_X = cp.asarray(local_X)
    local_y = cp.asarray(local_y)

    local_X = local_X.repeat(math.ceil(n_rows / len(local_X)), axis=0)
    local_y = local_y.repeat(math.ceil(n_rows / len(local_y)), axis=0)

    # Add some gaussian noise
    local_X += cp.random.standard_normal(local_X.shape, dtype=cp.float32)

    return local_X, local_y


def _local_umap_trustworthiness(
    local_X: np.ndarray,
    local_y: np.ndarray,
    n_neighbors: int,
    supervised: bool,
) -> float:
    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors, random_state=42, init="random")
    y_train = local_y if supervised else None
    local_model.fit(local_X, y=y_train)
    embedding = local_model.transform(local_X)

    return trustworthiness(local_X, embedding, n_neighbors=n_neighbors, batch_size=5000)


def _spark_umap_trustworthiness(
    local_X: np.ndarray,
    local_y: np.ndarray,
    n_neighbors: int,
    supervised: bool,
    n_parts: int,
    gpu_number: int,
    dtype: np.dtype,
    feature_type: str,
) -> float:
    umap_estimator = UMAP(
        n_neighbors=n_neighbors,
        random_state=42,
        init="random",
        num_workers=gpu_number,
    )

    with CleanSparkSession() as spark:
        if supervised:
            data_df, features_col, label_col = create_pyspark_dataframe(
                spark, feature_type, dtype, local_X, local_y
            )
            assert label_col is not None
            umap_estimator.setLabelCol(label_col)
        else:
            data_df, features_col, _ = create_pyspark_dataframe(
                spark, feature_type, dtype, local_X, None
            )

        data_df = data_df.repartition(n_parts)
        umap_estimator.setFeaturesCol(features_col)
        umap_model = umap_estimator.fit(data_df)
        pdf = umap_model.transform(data_df).toPandas()
        embedding = cp.asarray(pdf["embedding"].to_list()).astype(cp.float32)
        input = cp.asarray(pdf["features"].to_list()).astype(cp.float32)

    return trustworthiness(input, embedding, n_neighbors=n_neighbors, batch_size=5000)


def _run_spark_test(
    n_parts: int,
    gpu_number: int,
    n_rows: int,
    supervised: bool,
    dataset: str,
    n_neighbors: int,
    dtype: np.dtype,
    feature_type: str,
) -> bool:
    local_X, local_y = _load_dataset(dataset, n_rows)

    dist_umap = _spark_umap_trustworthiness(
        local_X,
        local_y,
        n_neighbors,
        supervised,
        n_parts,
        gpu_number,
        dtype,
        feature_type,
    )

    loc_umap = _local_umap_trustworthiness(local_X, local_y, n_neighbors, supervised)

    print("Local UMAP trustworthiness score : {:.4f}".format(loc_umap))
    print("Spark UMAP trustworthiness score : {:.4f}".format(dist_umap))

    trust_diff = loc_umap - dist_umap

    return trust_diff <= 0.15


@pytest.mark.parametrize("n_parts", [2, 9])
@pytest.mark.parametrize("n_rows", [100, 500])
@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("dataset", ["digits", "iris"])
@pytest.mark.parametrize("n_neighbors", [10])
@pytest.mark.parametrize("dtype", cuml_supported_data_types)
@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.slow
def test_spark_umap(
    n_parts: int,
    gpu_number: int,
    n_rows: int,
    supervised: bool,
    dataset: str,
    n_neighbors: int,
    dtype: np.dtype,
    feature_type: str,
) -> None:
    result = _run_spark_test(
        n_parts,
        gpu_number,
        n_rows,
        supervised,
        dataset,
        n_neighbors,
        dtype,
        feature_type,
    )

    if not result:
        result = _run_spark_test(
            n_parts,
            gpu_number,
            n_rows,
            supervised,
            dataset,
            n_neighbors,
            dtype,
            feature_type,
        )

    assert result


@pytest.mark.parametrize("n_parts", [5])
@pytest.mark.parametrize("n_rows", [500])
@pytest.mark.parametrize("supervised", [True])
@pytest.mark.parametrize("dataset", ["digits"])
@pytest.mark.parametrize("n_neighbors", [10])
@pytest.mark.parametrize("dtype", [cuml_supported_data_types[0]])
@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
def test_spark_umap_fast(
    n_parts: int,
    gpu_number: int,
    n_rows: int,
    supervised: bool,
    dataset: str,
    n_neighbors: int,
    dtype: np.dtype,
    feature_type: str,
    caplog: LogCaptureFixture,
) -> None:
    result = _run_spark_test(
        n_parts,
        gpu_number,
        n_rows,
        supervised,
        dataset,
        n_neighbors,
        dtype,
        feature_type,
    )

    if not result:
        result = _run_spark_test(
            n_parts,
            gpu_number,
            n_rows,
            supervised,
            dataset,
            n_neighbors,
            dtype,
            feature_type,
        )

    assert result

    assert UMAP()._float32_inputs

    # float32_inputs warn, umap only accepts float32
    umap_float32 = UMAP(float32_inputs=False)
    # float32_inputs warn, umap only accepts float32
    assert "float32_inputs to False" in caplog.text
    assert umap_float32._float32_inputs


@pytest.mark.parametrize("default_params", [True, False])
def test_params(tmp_path: str, default_params: bool) -> None:
    from cuml import UMAP as cumlUMAP

    spark_params = {
        "n_neighbors": 15,
        "n_components": 2,
        "metric": "euclidean",
        "n_epochs": None,
        "learning_rate": 1.0,
        "init": "spectral",
        "min_dist": 0.1,
        "spread": 1.0,
        "set_op_mix_ratio": 1.0,
        "local_connectivity": 1.0,
        "repulsion_strength": 1.0,
        "negative_sample_rate": 5,
        "transform_queue_size": 4.0,
        "a": None,
        "b": None,
        "precomputed_knn": None,
        "random_state": None,
        "verbose": False,
    }

    cuml_params = get_default_cuml_parameters(
        cuml_classes=[cumlUMAP],
        excludes=[
            "build_algo",
            "build_kwds",
            "callback",
            "handle",
            "hash_input",
            "metric_kwds",
            "output_type",
            "target_metric",
            "target_n_neighbors",
            "target_weight",
        ],
    )

    # Ensure internal cuml defaults match actual cuml defaults
    assert UMAP()._get_cuml_params_default() == cuml_params

    if default_params:
        umap = UMAP()
    else:
        nondefault_params = {
            "n_neighbors": 12,
            "learning_rate": 0.9,
            "random_state": 42,
        }
        umap = UMAP(**nondefault_params)
        cuml_params.update(nondefault_params)
        spark_params.update(nondefault_params)

    # Ensure both Spark API params and internal cuml_params are set correctly
    assert_params(umap, spark_params, cuml_params)
    assert umap.cuml_params == cuml_params

    # Estimator persistence
    path = tmp_path + "/umap_tests"
    estimator_path = f"{path}/umap"
    umap.write().overwrite().save(estimator_path)
    loaded_umap = UMAP.load(estimator_path)
    assert_params(loaded_umap, spark_params, cuml_params)
    assert umap.cuml_params == cuml_params
    assert loaded_umap._float32_inputs

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(UMAP)


def test_umap_model_persistence(gpu_number: int, tmp_path: str) -> None:
    from cuml.datasets import make_blobs

    X, _ = make_blobs(
        100,
        20,
        centers=42,
        cluster_std=0.1,
        dtype=np.float32,
        random_state=10,
    )

    with CleanSparkSession() as spark:
        pyspark_type = "float"
        feature_cols = [f"c{i}" for i in range(X.shape[1])]
        schema = [f"{c} {pyspark_type}" for c in feature_cols]
        df = spark.createDataFrame(X.tolist(), ",".join(schema))
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)

        umap = UMAP(num_workers=gpu_number).setFeaturesCol("features")

        def assert_umap_model(model: UMAPModel) -> None:
            embedding = np.array(model.embedding)
            raw_data = np.array(model.raw_data)
            assert embedding.shape == (100, 2)
            assert raw_data.shape == (100, 20)
            assert np.array_equal(raw_data, X.get())
            assert model.dtype == "float32"
            assert model.n_cols == X.shape[1]

        umap_model = umap.fit(df)
        assert_umap_model(model=umap_model)

        # Model persistence
        path = tmp_path + "/umap_tests"
        model_path = f"{path}/umap_model"
        umap_model.write().overwrite().save(model_path)
        umap_model_loaded = UMAPModel.load(model_path)
        assert_umap_model(model=umap_model_loaded)


@pytest.mark.parametrize("BROADCAST_LIMIT", [8 << 20, 8 << 18])
def test_umap_broadcast_chunks(gpu_number: int, BROADCAST_LIMIT: int) -> None:
    from cuml.datasets import make_blobs

    X, _ = make_blobs(
        5000,
        3000,
        centers=42,
        cluster_std=0.1,
        dtype=np.float32,
        random_state=10,
    )

    with CleanSparkSession() as spark:
        pyspark_type = "float"
        feature_cols = [f"c{i}" for i in range(X.shape[1])]
        schema = [f"{c} {pyspark_type}" for c in feature_cols]
        df = spark.createDataFrame(X.tolist(), ",".join(schema))
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)

        umap = UMAP(num_workers=gpu_number).setFeaturesCol("features")
        umap.BROADCAST_LIMIT = BROADCAST_LIMIT

        umap_model = umap.fit(df)

        def assert_umap_model(model: UMAPModel) -> None:
            embedding = np.array(model.embedding)
            raw_data = np.array(model.raw_data)
            assert embedding.shape == (5000, 2)
            assert raw_data.shape == (5000, 3000)
            assert np.array_equal(raw_data, X.get())
            assert model.dtype == "float32"
            assert model.n_cols == X.shape[1]

        assert_umap_model(model=umap_model)

        pdf = umap_model.transform(df).toPandas()
        embedding = cp.asarray(pdf["embedding"].to_list()).astype(cp.float32)
        input = cp.asarray(pdf["features"].to_list()).astype(cp.float32)

        dist_umap = trustworthiness(input, embedding, n_neighbors=15, batch_size=5000)
        loc_umap = _local_umap_trustworthiness(X, np.zeros(0), 15, False)
        trust_diff = loc_umap - dist_umap

        assert trust_diff <= 0.15


def test_umap_sample_fraction(gpu_number: int) -> None:
    from cuml.datasets import make_blobs

    n_rows = 5000
    sample_fraction = 0.5
    random_state = 42

    X, _ = make_blobs(
        n_rows,
        10,
        centers=42,
        cluster_std=0.1,
        dtype=np.float32,
        random_state=10,
    )

    with CleanSparkSession() as spark:
        pyspark_type = "float"
        feature_cols = [f"c{i}" for i in range(X.shape[1])]
        schema = [f"{c} {pyspark_type}" for c in feature_cols]
        df = spark.createDataFrame(X.tolist(), ",".join(schema)).coalesce(1)
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)

        umap = (
            UMAP(num_workers=gpu_number, random_state=random_state)
            .setFeaturesCol("features")
            .setSampleFraction(sample_fraction)
        )
        assert umap.getSampleFraction() == sample_fraction
        assert umap.getRandomState() == random_state

        umap_model = umap.fit(df)

        def assert_umap_model(model: UMAPModel) -> None:
            embedding = np.array(model.embedding)
            raw_data = np.array(model.raw_data)

            threshold = 2 * np.sqrt(
                n_rows * sample_fraction * (1 - sample_fraction)
            )  # 2 std devs

            assert np.abs(n_rows * sample_fraction - embedding.shape[0]) <= threshold
            assert np.abs(n_rows * sample_fraction - raw_data.shape[0]) <= threshold
            assert model.dtype == "float32"
            assert model.n_cols == X.shape[1]

        assert_umap_model(model=umap_model)

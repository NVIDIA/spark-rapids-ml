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

import os
import sys

file_path = os.path.abspath(__file__)
file_dir_path = os.path.dirname(file_path)
extra_python_path = file_dir_path + "/../benchmark"
sys.path.append(extra_python_path)

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame
from sklearn.datasets import make_blobs

from benchmark.bench_nearest_neighbors import CPUNearestNeighborsModel
from spark_rapids_ml.core import alias

from .sparksession import CleanSparkSession
from .utils import array_equal


def get_sgnn_res(
    X_item: np.ndarray, X_query: np.ndarray, n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors as SGNN

    sg_nn = SGNN(n_neighbors=n_neighbors)
    sg_nn.fit(X_item)
    sg_distances, sg_indices = sg_nn.kneighbors(X_query)
    return (sg_distances, sg_indices)


def assert_knn_equal(
    knn_df: DataFrame,
    id_col_name: str,
    distances: np.ndarray,
    indices: np.ndarray,
    total_tol: float = 1e-3,
) -> None:
    res_pd: pd.DataFrame = knn_df.sort(f"query_{id_col_name}").toPandas()
    mg_indices: np.ndarray = np.array(res_pd["indices"].to_list())
    mg_distances: np.ndarray = np.array(res_pd["distances"].to_list())

    assert array_equal(mg_distances, distances)

    # set total_tol because two nearest neighbors may have the same distance to the query
    assert array_equal(mg_indices, indices, total_tol=total_tol)

    for i in range(len(mg_indices)):
        for j in range(len(mg_indices[i])):
            if mg_indices[i][j] != indices[i][j]:
                assert mg_distances[i][j] == distances[i][j]


def test_cpunn_withid() -> None:

    n_samples = 1000
    n_features = 50
    n_clusters = 10
    n_neighbors = 30

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=0,
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    sg_distances, sg_indices = get_sgnn_res(X, X, n_neighbors)

    with CleanSparkSession({}) as spark:

        def py_func(id: int) -> List[int]:
            return X[id].tolist()

        from pyspark.sql.functions import udf

        spark_func = udf(py_func, "array<float>")
        df = spark.range(len(X)).select("id", spark_func("id").alias("features"))

        mg_model = (
            CPUNearestNeighborsModel(df)
            .setInputCol("features")
            .setIdCol("id")
            .setK(n_neighbors)
        )

        _, _, knn_df = mg_model.kneighbors(df)
        assert_knn_equal(knn_df, "id", sg_distances, sg_indices)


def test_cpunn_noid() -> None:

    n_samples = 1000
    n_features = 50
    n_clusters = 10
    n_neighbors = 30

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=0,
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    with CleanSparkSession({}) as spark:

        X_pylist = X.tolist()
        df = spark.createDataFrame(X_pylist)

        from pyspark.sql.functions import array

        df = df.select(array(df.columns).alias("features"))

        mg_model = (
            CPUNearestNeighborsModel(df).setInputCol("features").setK(n_neighbors)
        )

        df_withid, _, knn_df = mg_model.kneighbors(df)

        pdf: pd.DataFrame = df_withid.sort(alias.row_number).toPandas()
        X_vec = np.array(pdf["features"].to_list())
        distances, indices = get_sgnn_res(X_vec, X_vec, n_neighbors)

        X_sparkid = np.array(pdf[alias.row_number].to_list())
        indices_mapped_to_sparkid = X_sparkid[
            indices
        ]  # note spark created ids are non-continuous

        assert_knn_equal(knn_df, alias.row_number, distances, indices_mapped_to_sparkid)

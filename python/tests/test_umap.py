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

import math
from typing import List, Tuple, Type, TypeVar

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cuml.datasets import make_blobs
from cuml.internals import logger
from cuml.metrics import trustworthiness
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from sklearn.datasets import load_digits, load_iris

from .sparksession import CleanSparkSession
from .utils import assert_params, create_pyspark_dataframe, cuml_supported_data_types


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
    dtype: np.dtype,
) -> float:
    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors, random_state=42, init="random")
    y_train = local_y if supervised else None
    local_model.fit(local_X, y=y_train)

    embedding = local_model.embedding_
    return trustworthiness(local_X, embedding, n_neighbors=n_neighbors, batch_size=5000)


def _spark_umap_trustworthiness(
    local_X: np.ndarray,
    local_y: np.ndarray,
    n_neighbors: int,
    supervised: bool,
    n_parts: int,
    sampling_ratio: float,
    dtype: np.dtype,
) -> float:
    from spark_rapids_ml.umap import UMAP

    local_model = UMAP(
        n_neighbors=n_neighbors,
        sample_fraction=sampling_ratio,
        random_state=42,
        init="random",
        num_workers=n_parts,
    )

    with CleanSparkSession() as spark:
        if supervised:
            data_df, features_col, label_col = create_pyspark_dataframe(
                spark, "array", dtype, local_X, local_y
            )
            assert label_col is not None
            local_model.setLabelCol(label_col)
        else:
            data_df, features_col, _ = create_pyspark_dataframe(
                spark, "array", dtype, local_X, None
            )

        local_model.setFeaturesCol(features_col)
        distributed_model = local_model.fit(data_df)
        # embedding = distributed_model.transform(data_df)
        embedding = cp.array(distributed_model.embedding_)

    return trustworthiness(local_X, embedding, n_neighbors=n_neighbors, batch_size=5000)


def _run_spark_test(
    n_parts: int,
    n_rows: int,
    sampling_ratio: float,
    supervised: bool,
    dataset: str,
    n_neighbors: int,
    dtype: np.dtype,
) -> bool:
    local_X, local_y = _load_dataset(dataset, n_rows)

    dist_umap = _spark_umap_trustworthiness(
        local_X, local_y, n_neighbors, supervised, n_parts, sampling_ratio, dtype
    )

    loc_umap = _local_umap_trustworthiness(
        local_X, local_y, n_neighbors, supervised, dtype
    )

    print("Local UMAP trustworthiness score : {:.2f}".format(loc_umap))
    print("Spark UMAP trustworthiness score : {:.2f}".format(dist_umap))

    trust_diff = loc_umap - dist_umap

    return trust_diff <= 0.15


#@pytest.mark.parametrize("n_parts", [2, 9])
@pytest.mark.parametrize("n_parts", [2])
#@pytest.mark.parametrize("n_rows", [100, 500])
@pytest.mark.parametrize("n_rows", [500])
@pytest.mark.parametrize("sampling_ratio", [1.0])
@pytest.mark.parametrize("supervised", [True, False])  # TODO: add supervised UMAP support
#@pytest.mark.parametrize("dataset", ["digits", "iris"])
@pytest.mark.parametrize("dataset", ["digits"])
@pytest.mark.parametrize("n_neighbors", [10])
@pytest.mark.parametrize("dtype", cuml_supported_data_types)
def test_spark_umap(
    n_parts: int,
    n_rows: int,
    sampling_ratio: float,
    supervised: bool,
    dataset: str,
    n_neighbors: int,
    dtype: np.dtype,
) -> None:
    result = _run_spark_test(
        n_parts,
        n_rows,
        sampling_ratio,
        supervised,
        dataset,
        n_neighbors,
        dtype,
    )

    if not result:
        result = _run_spark_test(
            n_parts,
            n_rows,
            sampling_ratio,
            supervised,
            dataset,
            n_neighbors,
            dtype,
        )

    assert result


def _test() -> None:
    from spark_rapids_ml.umap import UMAP

    X, _ = make_blobs(
        1000, 10, centers=42, cluster_std=0.1, dtype=np.float32, random_state=10
    )

    with CleanSparkSession() as spark:
        print("input shape:", X.shape)
        data_df, features_col, _ = create_pyspark_dataframe(
            spark, "array", cuml_supported_data_types[0], X, None
        )
        local_model = UMAP(sample_fraction=0.25).setFeaturesCol(features_col)
        print("model params:", local_model.cuml_params)
        gpu_model = local_model.fit(data_df)
        print(
            "embedding shape:",
            len(gpu_model.embedding_),
            ",",
            len(gpu_model.embedding_[0]),
        )
        gpu_model.transform(data_df).show()

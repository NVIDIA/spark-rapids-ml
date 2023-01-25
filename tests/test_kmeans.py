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

from typing import List, Tuple

import numpy as np
import pytest

from sparkcuml.cluster import SparkCumlKMeans, SparkCumlKMeansModel

from .sparksession import CleanSparkSession
from .utils import (
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types_alias,
    idfn,
)


def test_kmeans_parameters(gpu_number: int, tmp_path: str) -> None:
    """
    Sparkcuml keeps the algorithmic parameters and their default values
    exactly the same as cuml multi-node multi-GPU KMeans,
    which follows scikit-learn convention.
    Please refer to https://docs.rapids.ai/api/cuml/stable/api.html#cuml.dask.cluster.KMeans
    """

    default_kmeans = SparkCumlKMeans()
    assert default_kmeans.getOrDefault("n_clusters") == 8
    assert default_kmeans.getOrDefault("max_iter") == 300
    assert default_kmeans.getOrDefault("tol") == 1e-4
    assert not default_kmeans.getOrDefault("verbose")
    assert default_kmeans.getOrDefault("random_state") == 1
    assert default_kmeans.getOrDefault("init") == "scalable-k-means++"
    assert default_kmeans.getOrDefault("oversampling_factor") == 2
    assert default_kmeans.getOrDefault("max_samples_per_batch") == 1 << 15
    assert default_kmeans.getOrDefault("num_workers") == 1
    assert default_kmeans.get_num_workers() == 1

    custom_params = {
        "n_clusters": 10,
        "max_iter": 100,
        "tol": 1e-1,
        "verbose": True,
        "random_state": 5,
        "init": "k-means||",
        "oversampling_factor": 3,
        "max_samples_per_batch": 45678,
    }

    def assertKmeansParameters(kmeans: SparkCumlKMeans) -> None:
        for key in custom_params:
            assert kmeans.getOrDefault(key) == custom_params[key]

    custom_kmeans = SparkCumlKMeans(**custom_params)
    assertKmeansParameters(kmeans=custom_kmeans)

    # Estimator persistence
    path = tmp_path + "/kmeans_tests"
    estimator_path = f"{path}/kmeans"
    custom_kmeans.write().overwrite().save(estimator_path)
    custom_kmeans_loaded = SparkCumlKMeans.load(estimator_path)

    assertKmeansParameters(kmeans=custom_kmeans_loaded)


def assert_centers_equal(
    a_clusters: List[List[float]], b_clusters: List[List[float]], tolerance: float
) -> None:
    assert len(a_clusters) == len(b_clusters)
    a_clusters = sorted(a_clusters, key=lambda l: l)
    b_clusters = sorted(b_clusters, key=lambda l: l)
    for i in range(len(a_clusters)):
        a_center = a_clusters[i]
        b_center = b_clusters[i]
        assert len(a_center) == len(b_center)
        assert a_center == pytest.approx(b_center, tolerance)


def test_toy_example(gpu_number: int, tmp_path: str) -> None:
    data = [[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]]

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data, gpu_number)
            .map(lambda row: (row,))
            .toDF(["features"])
        )
        sparkcuml_kmeans = SparkCumlKMeans(
            num_workers=gpu_number, n_clusters=2
        ).setFeaturesCol("features")

        def assert_kmeans_model(model: SparkCumlKMeansModel) -> None:
            assert len(model.cluster_centers_) == 2
            sorted_centers = sorted(model.cluster_centers_, key=lambda p: p)
            assert sorted_centers[0] == pytest.approx([1.0, 1.5], 0.001)
            assert sorted_centers[1] == pytest.approx([3.5, 2.5], 0.001)
            assert model.dtype == "float64"
            assert model.n_cols == 2

        kmeans_model = sparkcuml_kmeans.fit(df)
        assert_kmeans_model(model=kmeans_model)

        # Model persistence
        path = tmp_path + "/kmeans_tests"
        model_path = f"{path}/kmeans_model"
        kmeans_model.write().overwrite().save(model_path)
        kmeans_model_loaded = SparkCumlKMeansModel.load(model_path)

        assert_kmeans_model(model=kmeans_model_loaded)

        # test transform function
        label_df = kmeans_model.transform(df)
        o_col = kmeans_model.getOutputCol()
        labels = [row[o_col] for row in label_df.collect()]

        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[1] != labels[2]
        assert labels[2] == labels[3]


@pytest.mark.parametrize(
    "feature_type", [feature_types_alias.array, feature_types_alias.multi_cols]
)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
def test_compare_cuml(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
) -> None:
    """
    The dataset of this test case comes from cuml:
    https://github.com/rapidsai/cuml/blob/496f1f155676fb4b7d99aeb117cbb456ce628a4b/python/cuml/tests/test_kmeans.py#L39
    """
    from cuml.datasets import make_blobs

    n_rows = data_shape[0]
    n_cols = data_shape[1]
    n_clusters = 8
    cluster_std = 1.0
    tolerance = 0.001

    X, _ = make_blobs(
        n_rows, n_cols, n_clusters, cluster_std=cluster_std, random_state=0
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    from cuml import KMeans

    cuml_kmeans = KMeans(n_clusters=n_clusters, output_type="numpy", tol=0.0, verbose=7)

    import cudf

    gdf = cudf.DataFrame(X)
    cuml_kmeans.fit(gdf)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )

        sparkcuml_kmeans = SparkCumlKMeans(
            num_workers=gpu_number, n_clusters=n_clusters, verbose=7
        ).setFeaturesCol(features_col)
        sparkcuml_model = sparkcuml_kmeans.fit(df)

        cuml_cluster_centers = cuml_kmeans.cluster_centers_.tolist()
        assert_centers_equal(
            sparkcuml_model.cluster_centers_,
            cuml_cluster_centers,
            tolerance,
        )

        # test transform function

        sid_ordered = sorted(
            range(n_clusters), key=lambda idx: sparkcuml_model.cluster_centers_[idx]
        )
        cid_ordered = sorted(
            range(n_clusters), key=lambda idx: cuml_cluster_centers[idx]
        )
        s2c = dict(
            zip(sid_ordered, cid_ordered)
        )  # map sparkcuml center id to cuml center id

        labelDf = sparkcuml_model.transform(df)
        o_col = sparkcuml_model.getOutputCol()
        slabels = [row[o_col] for row in labelDf.collect()]

        clabels = cuml_kmeans.predict(gdf).tolist()

        assert len(slabels) == len(clabels)
        to_clabels = [s2c[v] for v in slabels]
        assert to_clabels == clabels


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_kmeans_numeric_type(gpu_number: int, data_type: str) -> None:
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
        kmeans = SparkCumlKMeans(
            num_workers=gpu_number, inputCols=feature_cols, n_clusters=2
        )
        kmeans.fit(df)

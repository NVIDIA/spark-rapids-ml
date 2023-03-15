from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spark_rapids_ml.knn import NearestNeighbors

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    create_pyspark_dataframe,
    idfn,
    pyspark_supported_feature_types,
)


def test_example(gpu_number: int) -> None:
    data = [
        ([1.0, 1.0], "a"),
        ([2.0, 2.0], "b"),
        ([3.0, 3.0], "c"),
        ([4.0, 4.0], "d"),
        ([5.0, 5.0], "e"),
        ([6.0, 6.0], "f"),
        ([7.0, 7.0], "g"),
        ([8.0, 8.0], "h"),
    ]

    query = [
        ([0.0, 0.0], "aa"),
        ([1.0, 1.0], "bb"),
        ([4.1, 4.1], "cc"),
        ([8.0, 8.0], "dd"),
        ([9.0, 9.0], "ee"),
    ]

    topk = 2

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(2)}
    with CleanSparkSession(conf) as spark:
        schema = f"features array<float>, metadata string"
        data_df = spark.createDataFrame(data, schema)
        query_df = spark.createDataFrame(query, schema)

        gpu_knn = NearestNeighbors(num_workers=gpu_number)
        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setK(topk)

        gpu_model = gpu_knn.fit(data_df)
        query_df_withid, item_df_withid, knn_df = gpu_model.kneighbors(query_df)
        query_df_withid.show()
        item_df_withid.show()
        knn_df.show()

        # check knn results
        import math

        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")

        distance_rows = distances_df.collect()
        distances = [r.distances for r in distance_rows]
        index_rows = indices_df.collect()
        indices = [r.indices for r in index_rows]

        assert array_equal(distances[0], [math.sqrt(2.0), math.sqrt(8.0)])
        assert indices[0] == [0, 1]

        assert array_equal(distances[1], [0.0, math.sqrt(2.0)])
        assert indices[1] == [0, 1]

        assert array_equal(
            distances[2], [math.sqrt(0.01 + 0.01), math.sqrt(0.81 + 0.81)]
        )
        assert indices[2] == [3, 4]

        assert array_equal(distances[3], [0.0, math.sqrt(2.0)])
        assert indices[3] == [7, 6]

        assert array_equal(distances[4], [math.sqrt(2.0), math.sqrt(8.0)])
        assert indices[4] == [7, 6]

        # throw an error if transform function is called
        with pytest.raises(NotImplementedError):
            gpu_model.transform(query_df)


def test_example_with_id(gpu_number: int) -> None:
    data = [
        (101, [1.0, 1.0], "a"),
        (102, [2.0, 2.0], "b"),
        (103, [3.0, 3.0], "c"),
        (104, [4.0, 4.0], "d"),
        (105, [5.0, 5.0], "e"),
        (106, [6.0, 6.0], "f"),
        (107, [7.0, 7.0], "g"),
        (108, [8.0, 8.0], "h"),
    ]

    query = [
        (201, [0.0, 0.0], "aa"),
        (202, [1.0, 1.0], "bb"),
        (203, [4.1, 4.1], "cc"),
        (204, [8.0, 8.0], "dd"),
        (205, [9.0, 9.0], "ee"),
    ]

    topk = 2

    with CleanSparkSession() as spark:
        schema = f"id int, features array<float>, metadata string"
        data_df = spark.createDataFrame(data, schema)
        query_df = spark.createDataFrame(query, schema)

        gpu_knn = NearestNeighbors(num_workers=gpu_number)
        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setIdCol("id")
        gpu_knn = gpu_knn.setK(topk)

        gpu_model = gpu_knn.fit(data_df)
        query_df, item_df, knn_df = gpu_model.kneighbors(query_df)
        query_df.show()
        item_df.show()
        knn_df.show()

        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")

        indices = indices_df.collect()
        indices = [r[0] for r in indices]

        assert indices[0] == [101, 102]
        assert indices[1] == [101, 102]
        assert indices[2] == [104, 105]
        assert indices[3] == [108, 107]
        assert indices[4] == [108, 107]


@pytest.mark.parametrize(
    "feature_type", pyspark_supported_feature_types
)  # vector feature type will be converted to float32 to be compatible with cuml multi-gpu NearestNeighbors Class
@pytest.mark.parametrize("data_shape", [(1000, 50)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize(
    "batch_size", [100, 10000]
)  # larger batch_size higher query throughput, yet more memory
@pytest.mark.slow
def test_nearest_neighbors(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    batch_size: int,
) -> None:
    n_neighbors = 5
    n_clusters = 10
    batch_size = batch_size

    X, _ = make_blobs(
        n_samples=data_shape[0],
        n_features=data_shape[1],
        centers=n_clusters,
        random_state=0,
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    # obtain cuml results
    from cuml import NearestNeighbors as cuNN

    cuml_knn = cuNN(n_neighbors=n_neighbors, output_type="numpy")

    cuml_knn.fit(X)
    cuml_distances, cuml_indices = cuml_knn.kneighbors(X)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        data_df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )

        knn_est = NearestNeighbors(
            n_neighbors=n_neighbors, batch_size=batch_size
        ).setInputCol(features_col)

        # obtain spark results
        knn_model = knn_est.fit(data_df)
        query_df = data_df
        (query_df_withid, item_df_withid, knn_df) = knn_model.kneighbors(query_df)

        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")
        # compare spark results with cuml results
        distances = distances_df.collect()
        distances = [r[0] for r in distances]
        indices = indices_df.collect()
        indices = [r[0] for r in indices]

        # compare top-1 nn indices(self) and distances(self)
        self_index = [knn[0] for knn in indices]
        assert self_index == list(range(data_shape[0]))
        cuml_self_index = [knn[0] for knn in cuml_indices]
        assert self_index == cuml_self_index

        self_distance = [kdist[0] for kdist in distances]
        assert array_equal(self_distance, [0.0 for i in range(data_shape[0])])
        cuml_self_distance = [kdist[0] for kdist in cuml_distances]
        assert array_equal(cuml_self_distance, [0.0 for i in range(data_shape[0])], 0.1)

        # compare non-self distances
        assert len(distances) == len(cuml_distances)
        nonself_distances = [knn[1:] for knn in distances]
        cuml_nonself_distances = [knn[1:] for knn in cuml_distances]
        for i in range(len(distances)):
            assert array_equal(nonself_distances[i], cuml_nonself_distances[i])

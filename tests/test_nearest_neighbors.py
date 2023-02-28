from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spark_rapids_ml.knn import NearestNeighbors

from .sparksession import CleanSparkSession
from .utils import array_equal, create_pyspark_dataframe, idfn


def test_example(gpu_number: int) -> None:
    data = [
        ([1.0, 1.0],),
        ([2.0, 2.0],),
        ([3.0, 3.0],),
        ([4.0, 4.0],),
        ([5.0, 5.0],),
        ([6.0, 6.0],),
        ([7.0, 7.0],),
        ([8.0, 8.0],),
    ]

    query = [
        ([0.0, 0.0],),
        ([1.0, 1.0],),
        ([4.1, 4.1],),
        ([8.0, 8.0],),
        ([9.0, 9.0],),
    ]

    topk = 2

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(2)}
    with CleanSparkSession(conf) as spark:
        schema = f"features array<float>"
        data_df = spark.createDataFrame(data, schema)
        query_df = spark.createDataFrame(query, schema)

        gpu_knn = NearestNeighbors(num_workers=gpu_number)
        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setK(topk)

        gpu_knn = gpu_knn.fit(data_df)
        distances_df, indices_df = gpu_knn.kneighbors(query_df)

        import math

        distances = distances_df.collect()
        distances = [r[0] for r in distances]
        indices = indices_df.collect()
        indices = [r[0] for r in indices]

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


@pytest.mark.parametrize("feature_type", ["array"])
@pytest.mark.parametrize("data_shape", [(1000, 50)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize("batch_size", [100, 10000]) # larger batch_size higher query throughput, yet more memory
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

        sparkcuml_knn = NearestNeighbors(n_neighbors=n_neighbors, inputCol=features_col, batch_size=batch_size)

        # obtain spark results
        sparkcuml_knn = sparkcuml_knn.fit(data_df)
        query_df = data_df
        distances_df, indices_df = sparkcuml_knn.kneighbors(query_df)

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

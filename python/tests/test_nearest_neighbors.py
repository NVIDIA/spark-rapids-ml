from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame
from sklearn.datasets import make_blobs

from spark_rapids_ml.core import alias
from spark_rapids_ml.knn import NearestNeighbors

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    create_pyspark_dataframe,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)


def test_default_cuml_params() -> None:
    from cuml import NearestNeighbors as CumlNearestNeighbors
    from cuml.neighbors.nearest_neighbors_mg import (
        NearestNeighborsMG,  # to include the batch_size parameter that exists in the MG class
    )

    cuml_params = get_default_cuml_parameters(
        [CumlNearestNeighbors, NearestNeighborsMG],
        [
            "handle",
            "algorithm",
            "metric",
            "p",
            "algo_params",
            "metric_expanded",
            "metric_params",
            "output_type",
        ],
    )
    spark_params = NearestNeighbors()._get_cuml_params_default()
    assert cuml_params == spark_params


def test_example(gpu_number: int, tmp_path: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

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
        ([0.0, 0.0], "qa"),
        ([1.0, 1.0], "qb"),
        ([4.1, 4.1], "qc"),
        ([8.0, 8.0], "qd"),
        ([9.0, 9.0], "qe"),
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

        with pytest.raises(NotImplementedError):
            gpu_knn.save(tmp_path + "/knn_esimator")

        gpu_model = gpu_knn.fit(data_df)

        with pytest.raises(NotImplementedError):
            gpu_model.save(tmp_path + "/knn_model")

        (item_df_withid, query_df_withid, knn_df) = gpu_model.kneighbors(query_df)
        item_df_withid.show()
        query_df_withid.show()
        knn_df.show()

        # check knn results
        import math

        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")

        distance_rows = distances_df.collect()
        distances = [r.distances for r in distance_rows]
        index_rows = indices_df.collect()
        indices = [r.indices for r in index_rows]

        def assert_distances_equal(distances: List[List[float]]) -> None:
            assert len(distances) == len(query)
            assert array_equal(distances[0], [math.sqrt(2.0), math.sqrt(8.0)])
            assert array_equal(distances[1], [0.0, math.sqrt(2.0)])
            assert array_equal(
                distances[2], [math.sqrt(0.01 + 0.01), math.sqrt(0.81 + 0.81)]
            )
            assert array_equal(distances[3], [0.0, math.sqrt(2.0)])
            assert array_equal(distances[4], [math.sqrt(2.0), math.sqrt(8.0)])

        item_ids = list(
            item_df_withid.select(alias.row_number).toPandas()[alias.row_number]
        )

        def assert_indices_equal(indices: List[List[int]]) -> None:
            assert len(indices) == len(query)
            assert indices[0] == [item_ids[0], item_ids[1]]
            assert indices[1] == [item_ids[0], item_ids[1]]
            assert indices[2] == [item_ids[3], item_ids[4]]
            assert indices[3] == [item_ids[7], item_ids[6]]
            assert indices[4] == [item_ids[7], item_ids[6]]

        assert_distances_equal(distances=distances)
        assert_indices_equal(indices=indices)

        # test transform: throw an error if transform function is called
        with pytest.raises(NotImplementedError):
            gpu_model.transform(query_df)

        # test exactNearestNeighborsJoin
        knnjoin_df = gpu_model.exactNearestNeighborsJoin(query_df, distCol="distCol")
        knnjoin_df.show()

        assert len(knnjoin_df.dtypes) == 3
        assert knnjoin_df.dtypes[0] == (
            "item_df",
            "struct<features:array<float>,metadata:string>",
        )
        assert knnjoin_df.dtypes[1] == (
            "query_df",
            "struct<features:array<float>,metadata:string>",
        )
        assert knnjoin_df.dtypes[2] == ("distCol", "float")

        def assert_knn_metadata_equal(knn_metadata: List[List[str]]) -> None:
            """
            This is equivalent to assert_indices_equal but replaces indices with item_metadata.
            """
            assert len(knn_metadata) == len(query)
            assert knn_metadata[0] == ["a", "b"]
            assert knn_metadata[1] == ["a", "b"]
            assert knn_metadata[2] == ["d", "e"]
            assert knn_metadata[3] == ["h", "g"]
            assert knn_metadata[4] == ["h", "g"]

        reconstructed_knn_df = reconstruct_knn_df(
            knnjoin_df=knnjoin_df, row_identifier_col="metadata", distCol="distCol"
        )

        reconstructed_rows = reconstructed_knn_df.collect()
        reconstructed_knn_metadata = [r.indices for r in reconstructed_rows]
        assert_knn_metadata_equal(reconstructed_knn_metadata)
        reconstructed_distances = [r.distances for r in reconstructed_rows]
        assert_distances_equal(reconstructed_distances)
        reconstructed_query_ids = [r.query_id for r in reconstructed_rows]
        assert reconstructed_query_ids == ["qa", "qb", "qc", "qd", "qe"]

        knnjoin_items = (
            knnjoin_df.select(
                knnjoin_df["item_df.features"].alias("features"),
                knnjoin_df["item_df.metadata"].alias("metadata"),
            )
            .distinct()
            .sort("metadata")
            .collect()
        )

        assert len(knnjoin_items) == 6
        assert knnjoin_items[0]["features"] == data[0][0]
        assert knnjoin_items[0]["metadata"] == data[0][1]
        assert knnjoin_items[1]["features"] == data[1][0]
        assert knnjoin_items[1]["metadata"] == data[1][1]
        assert knnjoin_items[2]["features"] == data[3][0]
        assert knnjoin_items[2]["metadata"] == data[3][1]
        assert knnjoin_items[3]["features"] == data[4][0]
        assert knnjoin_items[3]["metadata"] == data[4][1]
        assert knnjoin_items[4]["features"] == data[6][0]
        assert knnjoin_items[4]["metadata"] == data[6][1]
        assert knnjoin_items[5]["features"] == data[7][0]
        assert knnjoin_items[5]["metadata"] == data[7][1]

        knnjoin_queries = (
            knnjoin_df.select(
                knnjoin_df["query_df.features"].alias("features"),
                knnjoin_df["query_df.metadata"].alias("metadata"),
            )
            .distinct()
            .sort("metadata")
            .collect()
        )

        assert len(knnjoin_queries) == len(query)
        for i in range(len(knnjoin_queries)):
            if i is 2:
                assert array_equal(knnjoin_queries[i]["features"], query[i][0])
            else:
                assert knnjoin_queries[i]["features"] == query[i][0]
            assert knnjoin_queries[i]["metadata"] == query[i][1]


def test_example_with_id(gpu_number: int) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

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
        (201, [0.0, 0.0], "qa"),
        (202, [1.0, 1.0], "qb"),
        (203, [4.1, 4.1], "qc"),
        (204, [8.0, 8.0], "qd"),
        (205, [9.0, 9.0], "qe"),
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
        item_df_withid, query_df_withid, knn_df = gpu_model.kneighbors(query_df)
        item_df_withid.show()
        query_df_withid.show()
        knn_df.show()

        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")

        indices = indices_df.collect()
        indices = [r[0] for r in indices]

        def assert_indices_equal(indices: List[List[int]]) -> None:
            assert indices[0] == [101, 102]
            assert indices[1] == [101, 102]
            assert indices[2] == [104, 105]
            assert indices[3] == [108, 107]
            assert indices[4] == [108, 107]

        # test exactNearestNeighborsJoin
        knnjoin_df = gpu_model.exactNearestNeighborsJoin(query_df, distCol="distCol")
        knnjoin_df.show()

        assert len(knnjoin_df.dtypes) == 3
        assert knnjoin_df.dtypes[0] == (
            "item_df",
            f"struct<id:int,features:array<float>,metadata:string>",
        )
        assert knnjoin_df.dtypes[1] == (
            "query_df",
            "struct<id:int,features:array<float>,metadata:string>",
        )
        assert knnjoin_df.dtypes[2] == ("distCol", "float")

        reconstructed_knn_df = reconstruct_knn_df(
            knnjoin_df=knnjoin_df, row_identifier_col="id", distCol="distCol"
        )

        reconstructed_rows = reconstructed_knn_df.collect()
        reconstructed_knn_indices = [r.indices for r in reconstructed_rows]
        assert_indices_equal(reconstructed_knn_indices)
        reconstructed_query_ids = [r.query_id for r in reconstructed_rows]
        assert reconstructed_query_ids == [201, 202, 203, 204, 205]


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

        # test kneighbors: obtain spark results
        knn_model = knn_est.fit(data_df)
        query_df = data_df
        (item_df_withid, query_df_withid, knn_df) = knn_model.kneighbors(query_df)

        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")
        # test kneighbors: compare spark results with cuml results
        distances = distances_df.collect()
        distances = [r[0] for r in distances]
        indices = indices_df.collect()
        indices = [r[0] for r in indices]

        # test kneighbors: compare top-1 nn indices(self) and distances(self)
        self_index = [knn[0] for knn in indices]

        assert self_index == list(
            item_df_withid.select(alias.row_number).toPandas()[alias.row_number]
        )

        self_distance = [kdist[0] for kdist in distances]
        assert array_equal(self_distance, [0.0 for i in range(data_shape[0])])
        cuml_self_distance = [kdist[0] for kdist in cuml_distances]
        assert array_equal(cuml_self_distance, [0.0 for i in range(data_shape[0])], 0.1)

        # test kneighbors: compare non-self distances
        assert len(distances) == len(cuml_distances)
        nonself_distances = [knn[1:] for knn in distances]
        cuml_nonself_distances = [knn[1:] for knn in cuml_distances]
        for i in range(len(distances)):
            assert array_equal(nonself_distances[i], cuml_nonself_distances[i])

        # test exactNearestNeighborsJoin
        with pytest.raises(ValueError):
            knn_model.exactNearestNeighborsJoin(query_df_withid)

        knn_model.setIdCol(item_df_withid.dtypes[0][0])
        knnjoin_df = knn_model.exactNearestNeighborsJoin(query_df_withid)
        reconstructed_knn_df = reconstruct_knn_df(
            knnjoin_df, row_identifier_col=knn_model.getIdCol()
        )
        assert reconstructed_knn_df.collect() == knn_df.collect()


def test_lsh_spark_compat(gpu_number: int) -> None:
    from pyspark.ml.feature import BucketedRandomProjectionLSH
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.functions import col

    # reduce the number of GPUs for toy dataset to avoid empty partition.
    # cuml backend requires k <= the number of rows in the smallest index partition.
    gpu_number = min(gpu_number, 2)
    topk = 2

    with CleanSparkSession() as spark:
        dataA = [
            (0, Vectors.dense([1.0, 1.0])),
            (1, Vectors.dense([1.0, -1.0])),
            (2, Vectors.dense([-1.0, -1.0])),
            (3, Vectors.dense([-1.0, 1.0])),
            (4, Vectors.dense([100.0, 100.0])),
            (5, Vectors.dense([100.0, -100.0])),
            (6, Vectors.dense([-100.0, -100.0])),
            (7, Vectors.dense([-100.0, 100.0])),
        ]
        dfA = spark.createDataFrame(dataA, ["id", "features"])

        dataB = [
            (4, Vectors.dense([1.0, 0.0])),
            (5, Vectors.dense([-1.0, 0.0])),
            (6, Vectors.dense([0.0, 1.0])),
            (7, Vectors.dense([0.0, -1.0])),
        ]
        dfB = spark.createDataFrame(dataB, ["id", "features"])
        dfA.show()
        dfB.show()

        # get CPU results
        brp = BucketedRandomProjectionLSH(
            inputCol="features", outputCol="hashes", bucketLength=5.0, numHashTables=3
        )
        model = brp.fit(dfA)
        spark_res = model.approxSimilarityJoin(
            dfA, dfB, 1.5, distCol="EuclideanDistance"
        )
        spark_res.show(truncate=False)

        # get GPU results with exactNearestNeighborsJoin(dfA, dfB, k, distCol="EuclideanDistance")
        gpu_knn = NearestNeighbors(num_workers=gpu_number, inputCol="features").setK(
            topk
        )
        gpu_model = gpu_knn.fit(dfA)
        gpu_res = gpu_model.exactNearestNeighborsJoin(
            query_df=dfB, distCol="EuclideanDistance"
        )
        gpu_res.show(truncate=False)

        # check results
        def check_dtypes(res_df: DataFrame, from_spark: bool) -> None:
            assert len(res_df.dtypes) == 3
            assert res_df.dtypes[0][0] == ("datasetA" if from_spark else "item_df")
            assert res_df.dtypes[1][0] == ("datasetB" if from_spark else "query_df")
            assert res_df.dtypes[2][0] == ("EuclideanDistance")

            if from_spark:
                assert res_df.dtypes[0][1].startswith(
                    "struct<id:bigint,features:vector"
                )
                assert res_df.dtypes[1][1].startswith(
                    "struct<id:bigint,features:vector"
                )
                assert res_df.dtypes[2][1] == "double"
            else:
                assert res_df.dtypes[0][1] == "struct<id:bigint,features:vector>"
                assert res_df.dtypes[1][1] == "struct<id:bigint,features:vector>"
                assert res_df.dtypes[2][1] == "float"

        check_dtypes(res_df=spark_res, from_spark=True)
        check_dtypes(res_df=gpu_res, from_spark=False)

        items = gpu_res.select(
            gpu_res["item_df.id"], gpu_res["item_df.features"]
        ).collect()
        assert len(items) == topk * len(dataB)
        for item in items:
            id = item.id
            features = item.features
            assert features == dataA[id][1]

        queries = gpu_res.select(
            gpu_res["query_df.id"], gpu_res["query_df.features"]
        ).collect()
        for query in queries:
            id = query.id
            features = query.features
            assert features == dataB[id - 4][1]

        knn_results = reconstruct_knn_df(
            gpu_res, row_identifier_col="id", distCol="EuclideanDistance"
        ).collect()
        assert knn_results[0]["query_id"] == 4
        assert knn_results[0]["indices"] == [1, 0] or knn_results[0]["indices"] == [
            0,
            1,
        ]
        assert knn_results[0]["distances"] == [1.0, 1.0]

        assert knn_results[1]["query_id"] == 5
        assert knn_results[1]["indices"] == [2, 3] or knn_results[1]["indices"] == [
            3,
            2,
        ]
        assert knn_results[1]["distances"] == [1.0, 1.0]

        assert knn_results[2]["query_id"] == 6
        assert knn_results[2]["indices"] == [3, 0] or knn_results[1]["indices"] == [
            0,
            3,
        ]
        assert knn_results[2]["distances"] == [1.0, 1.0]

        assert knn_results[3]["query_id"] == 7
        assert knn_results[3]["indices"] == [2, 1] or knn_results[1]["indices"] == [
            1,
            2,
        ]
        assert knn_results[3]["distances"] == [1.0, 1.0]


def reconstruct_knn_df(
    knnjoin_df: DataFrame, row_identifier_col: str, distCol: str = "distCol"
) -> DataFrame:
    """
    This function accepts the returned dataframe (denoted as knnjoin_df) of exactNearestNeighborsjoin,
    then reconstructs the returned dataframe (i.e. knn_df) of kneighbors.
    """
    knn_df: DataFrame = knnjoin_df.select(
        knnjoin_df[f"query_df.{row_identifier_col}"].alias(f"query_id"),
        knnjoin_df[f"item_df.{row_identifier_col}"].alias("index"),
        knnjoin_df[distCol].alias("distance"),
    )

    def functor(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values(by=["distance"])
        indices = pdf["index"].tolist()
        distances = pdf["distance"].tolist()
        query_id = pdf[f"query_id"].tolist()[0]

        return pd.DataFrame(
            {"query_id": [query_id], "indices": [indices], "distances": [distances]}
        )

    knn_df = knn_df.groupBy("query_id").applyInPandas(
        functor,
        schema=f"query_id {knn_df.dtypes[0][1]}, "
        + f"indices array<{knn_df.dtypes[1][1]}>, "
        + f"distances array<{knn_df.dtypes[2][1]}>",
    )

    knn_df = knn_df.sort("query_id")
    return knn_df

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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, StructField, StructType
from sklearn.datasets import make_blobs

from spark_rapids_ml.core import alias
from spark_rapids_ml.knn import (
    ApproximateNearestNeighbors,
    ApproximateNearestNeighborsModel,
    NearestNeighbors,
    NearestNeighborsModel,
)

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)

NNEstimator = Union[NearestNeighbors, ApproximateNearestNeighbors]
NNModel = Union[NearestNeighborsModel, ApproximateNearestNeighborsModel]


@pytest.mark.parametrize("default_params", [True, False])
def test_params(default_params: bool, caplog: LogCaptureFixture) -> None:
    from cuml import NearestNeighbors as CumlNearestNeighbors
    from cuml.neighbors.nearest_neighbors_mg import (
        NearestNeighborsMG,  # to include the batch_size parameter that exists in the MG class
    )

    spark_params = {
        param.name: value
        for param, value in NearestNeighbors().extractParamMap().items()
    }

    cuml_params = get_default_cuml_parameters(
        cuml_classes=[CumlNearestNeighbors, NearestNeighborsMG],
        excludes=[
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
    assert cuml_params == NearestNeighbors()._get_cuml_params_default()

    if default_params:
        knn = NearestNeighbors()
    else:
        knn = NearestNeighbors(k=7)
        cuml_params["n_neighbors"] = 7
        spark_params["k"] = 7

    # Ensure both Spark API params and internal cuml_params are set correctly
    assert_params(knn, spark_params, cuml_params)
    assert knn.cuml_params == cuml_params

    # float32_inputs warn, NearestNeighbors only accepts float32
    nn_float32 = NearestNeighbors(float32_inputs=False)
    assert "float32_inputs to False" in caplog.text
    assert nn_float32._float32_inputs

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(NearestNeighbors)


def test_knn_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [
        ({"k": 37}, {"n_neighbors": 37}),
        ({"verbose": True}, {"verbose": True}),
    ]

    for pair in param_list:
        spark_param = pair[0]
        cuml_param = spark_param if len(pair) == 1 else pair[1]
        _test_est_copy(NearestNeighbors, spark_param, cuml_param)


def func_test_example_no_id(
    tmp_path: str, gpu_knn: NNEstimator
) -> Tuple[NNEstimator, NNModel]:

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

        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setK(topk)

        assert topk == gpu_knn.getK()

        with pytest.raises(NotImplementedError):
            gpu_knn.save(tmp_path + "/knn_esimator")

        gpu_model = gpu_knn.fit(data_df)

        assert topk == gpu_knn.getK()

        with pytest.raises(NotImplementedError):
            gpu_model.save(tmp_path + "/knn_model")

        # test kneighbors on empty query dataframe
        df_empty = spark.createDataFrame([], schema="features array<float>")
        (_, _, knn_df_empty) = gpu_model.kneighbors(df_empty)
        knn_df_empty.show()

        # test kneighbors on normal query dataframe
        (item_df_withid, query_df_withid, knn_df) = gpu_model.kneighbors(query_df)
        item_df_withid.show()
        query_df_withid.show()
        knn_df = knn_df.cache()
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

        if isinstance(gpu_knn, NearestNeighbors):
            knnjoin_df = gpu_model.exactNearestNeighborsJoin(
                query_df, distCol="distCol"
            )
        else:
            knnjoin_df = gpu_model.approxSimilarityJoin(query_df, distCol="distCol")

        knnjoin_df = knnjoin_df.cache()
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
        reconstructed_query_ids = [r.query_metadata for r in reconstructed_rows]
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
            if i == 2:
                assert array_equal(knnjoin_queries[i]["features"], query[i][0])
            else:
                assert knnjoin_queries[i]["features"] == query[i][0]
            assert knnjoin_queries[i]["metadata"] == query[i][1]

        # Test fit(dataset, ParamMap) that copies existing estimator
        # After copy, self.isSet("idCol") becomes true. But the added id column does not exist in the dataframe
        paramMap = gpu_knn.extractParamMap()
        gpu_model_v2 = gpu_knn.fit(data_df, paramMap)

        assert gpu_knn.isSet("idCol") is False
        assert gpu_model_v2.isSet("idCol") is True

        (_, _, knn_df_v2) = gpu_model_v2.kneighbors(query_df)
        assert knn_df_v2.collect() == knn_df.collect()

        knn_df.unpersist()
        knnjoin_df.unpersist()

        return gpu_knn, gpu_model


def test_example(gpu_number: int, tmp_path: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    gpu_knn = NearestNeighbors(num_workers=gpu_number)
    func_test_example_no_id(tmp_path, gpu_knn)


def func_test_example_with_id(
    tmp_path: str, gpu_knn: NNEstimator
) -> Tuple[NNEstimator, NNModel]:
    # reduce the number of GPUs for toy dataset to avoid empty partition

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

        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setIdCol("id")
        gpu_knn = gpu_knn.setK(topk)

        gpu_model = gpu_knn.fit(data_df)

        # test kneighbors on empty query dataframe with id column
        df_empty = spark.createDataFrame([], schema="id long, features array<float>")
        (_, _, knn_df_empty) = gpu_model.kneighbors(df_empty)
        knn_df_empty.show()

        # test kneighbors on normal query dataframe
        item_df_withid, query_df_withid, knn_df = gpu_model.kneighbors(query_df)
        item_df_withid.show()
        query_df_withid.show()

        knn_df = knn_df.cache()
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
        if isinstance(gpu_model, NearestNeighborsModel):
            knnjoin_df = gpu_model.exactNearestNeighborsJoin(
                query_df, distCol="distCol"
            )
        else:
            knnjoin_df = gpu_model.approxSimilarityJoin(query_df, distCol="distCol")

        knnjoin_df = knnjoin_df.cache()
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

        knn_df.unpersist()
        knnjoin_df.unpersist()
        return (gpu_knn, gpu_model)


def test_example_with_id(gpu_number: int, tmp_path: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    gpu_knn = NearestNeighbors(num_workers=gpu_number)
    func_test_example_no_id(tmp_path, gpu_knn)


@pytest.mark.parametrize(
    "feature_type", pyspark_supported_feature_types
)  # vector feature type will be converted to float32 to be compatible with cuml multi-gpu NearestNeighbors Class
@pytest.mark.parametrize("data_shape", [(1000, 50)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize(
    "max_record_batch", [pytest.param(100, marks=pytest.mark.slow), 10000]
)
@pytest.mark.parametrize(
    "batch_size", [pytest.param(100, marks=pytest.mark.slow), 10000]
)  # larger batch_size higher query throughput, yet more memory
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

    # set average norm sq to be 1 to allow comparisons with default error thresholds
    # below
    root_ave_norm_sq = np.sqrt(np.average(np.linalg.norm(X, ord=2, axis=1) ** 2))
    X = X / root_ave_norm_sq

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

        # test kneighbors: compare squared distances
        # note that single node and multi node may run slightly different kernels resulting
        # in different distances.  This is especially an issue for self distances which don't come out
        # to be 0 necessarily due to expanded form of calculation (|x-y|^2 = |x|^2 + |y|^2 - 2 <x,y>).
        # sqrt amplifies this error so we compare Euclidean distance squared and expect error to be below
        # default threshold in array_equal
        assert len(distances) == len(cuml_distances)
        np_distances = np.array(distances)
        np_distances *= np_distances
        cuml_distances *= cuml_distances
        for i in range(len(distances)):
            assert array_equal(np_distances[i], cuml_distances[i])

        # test exactNearestNeighborsJoin
        with pytest.raises(ValueError):
            knn_model.exactNearestNeighborsJoin(query_df_withid)

        knn_model.setIdCol(item_df_withid.dtypes[0][0])
        knnjoin_df = knn_model.exactNearestNeighborsJoin(query_df_withid)
        reconstructed_knn_df = reconstruct_knn_df(
            knnjoin_df, row_identifier_col=knn_model._getIdColOrDefault()
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
    knnjoin_df: DataFrame,
    row_identifier_col: str,
    distCol: str = "distCol",
    ascending: bool = True,
) -> DataFrame:
    """
    This function accepts the returned dataframe (denoted as knnjoin_df) of exactNearestNeighborsjoin,
    then reconstructs the returned dataframe (i.e. knn_df) of kneighbors.

    Note the reconstructed knn_df does not guarantee the same indices as the original knn_df, because the distances to two neighbors can be the same.
    """
    knn_df: DataFrame = knnjoin_df.select(
        knnjoin_df[f"query_df.{row_identifier_col}"].alias(f"query_id"),
        knnjoin_df[f"item_df.{row_identifier_col}"].alias("index"),
        knnjoin_df[distCol].alias("distance"),
    )

    def functor(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values(by=["distance"], ascending=ascending)
        indices = pdf["index"].tolist()
        distances = pdf["distance"].tolist()
        query_id = pdf[f"query_id"].tolist()[0]

        return pd.DataFrame(
            {
                f"query_{row_identifier_col}": [query_id],
                "indices": [indices],
                "distances": [distances],
            }
        )

    knn_df = knn_df.groupBy("query_id").applyInPandas(
        functor,
        schema=f"query_{row_identifier_col} {knn_df.dtypes[0][1]}, "
        + f"indices array<{knn_df.dtypes[1][1]}>, "
        + f"distances array<{knn_df.dtypes[2][1]}>",
    )

    knn_df = knn_df.sort(f"query_{row_identifier_col}")
    return knn_df

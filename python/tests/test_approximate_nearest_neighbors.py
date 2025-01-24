# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import Row
from sklearn.datasets import make_blobs

from spark_rapids_ml.core import alias
from spark_rapids_ml.knn import (
    ApproximateNearestNeighbors,
    ApproximateNearestNeighborsModel,
)

from .sparksession import CleanSparkSession
from .test_nearest_neighbors import (
    NNEstimator,
    NNModel,
    func_test_example_no_id,
    func_test_example_with_id,
    reconstruct_knn_df,
)
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)


def cal_dist(v1: np.ndarray, v2: np.ndarray, metric: str) -> float:
    if metric == "inner_product":
        return np.dot(v1, v2)
    elif metric in {"euclidean", "l2", "sqeuclidean"}:
        dist = float(np.linalg.norm(v1 - v2))
        if metric == "sqeuclidean":
            return dist * dist
        else:
            return dist
    elif metric == "cosine":
        v1_l2norm = np.linalg.norm(v1)
        v2_l2norm = np.linalg.norm(v2)
        if v1_l2norm == 0 or v2_l2norm == 0:
            return 0.0
        return 1 - np.dot(v1, v2) / (v1_l2norm * v2_l2norm)
    else:
        assert False, f"Does not recognize metric '{metric}'"


@pytest.mark.parametrize("default_params", [True, False])
def test_params(default_params: bool) -> None:
    from cuml import NearestNeighbors as CumlNearestNeighbors

    spark_params = {
        param.name: value
        for param, value in ApproximateNearestNeighbors().extractParamMap().items()
    }

    # obtain n_neighbors, verbose, algorithm, algo_params, metric
    cuml_params = get_default_cuml_parameters(
        cuml_classes=[CumlNearestNeighbors],
        excludes=[
            "handle",
            "p",
            "metric_expanded",
            "metric_params",
            "output_type",
        ],
    )

    cuml_params["algorithm"] = "ivfflat"  # change cuml default 'auto' to 'ivfflat'

    # Ensure internal cuml defaults match actual cuml defaults
    assert ApproximateNearestNeighbors()._get_cuml_params_default() == cuml_params

    if default_params:
        knn = ApproximateNearestNeighbors()
    else:
        knn = ApproximateNearestNeighbors(k=7)
        cuml_params["n_neighbors"] = 7
        spark_params["k"] = 7

    # Ensure both Spark API params and internal cuml_params are set correctly
    assert_params(knn, spark_params, cuml_params)
    assert knn.cuml_params == cuml_params

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(ApproximateNearestNeighbors)


def test_ann_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [
        ({"k": 38}, {"n_neighbors": 38}),
        ({"algorithm": "cagra"}, {"algorithm": "cagra"}),
        ({"metric": "cosine"}, {"metric": "cosine"}),
        (
            {"algoParams": {"nlist": 999, "nprobe": 11}},
            {"algo_params": {"nlist": 999, "nprobe": 11}},
        ),
        ({"verbose": True}, {"verbose": True}),
    ]

    for pair in param_list:
        _test_est_copy(ApproximateNearestNeighbors, pair[0], pair[1])


def test_search_index_params() -> None:
    # test cagra index params and search params
    cagra_index_param: Dict[str, Any] = {
        "intermediate_graph_degree": 80,
        "graph_degree": 60,
        "build_algo": "nn_descent",
        "compression": None,
    }

    cagra_search_param: Dict[str, Any] = {
        "max_queries": 10,
        "itopk_size": 10,
        "max_iterations": 20,
        "min_iterations": 10,
        "search_width": 2,
        "num_random_samplings": 5,
    }

    algoParams = {**cagra_index_param, **cagra_search_param}
    index_param, search_param = (
        ApproximateNearestNeighborsModel._cal_cagra_params_and_check(
            algoParams=algoParams,
            metric="sqeuclidean",
            topk=cagra_search_param["itopk_size"],
        )
    )
    assert index_param == {"metric": "sqeuclidean", **cagra_index_param}
    assert search_param == cagra_search_param


@pytest.mark.parametrize(
    "algo_and_params",
    [("ivfflat", {"nlist": 1, "nprobe": 2})],
)
@pytest.mark.parametrize(
    "func_test",
    [func_test_example_no_id, func_test_example_with_id],
)
def test_example(
    algo_and_params: Tuple[str, Optional[dict[str, Any]]],
    func_test: Callable[[NNEstimator, str], Tuple[NNEstimator, NNModel]],
    gpu_number: int,
    tmp_path: str,
) -> None:
    algorithm = algo_and_params[0]
    algoParams = algo_and_params[1]

    gpu_knn = ApproximateNearestNeighbors(algorithm=algorithm, algoParams=algoParams)
    gpu_knn, gpu_model = func_test(tmp_path, gpu_knn)  # type: ignore

    for obj in [gpu_knn, gpu_model]:
        assert obj._cuml_params["algorithm"] == algorithm
        assert obj._cuml_params["algo_params"] == algoParams


@pytest.mark.slow
def test_empty_dataframe() -> None:
    gpu_knn = ApproximateNearestNeighbors()
    gpu_knn = gpu_knn.setInputCol("features").setK(1)
    with CleanSparkSession() as spark:
        schema = f"features array<float>, metadata string"
        item_df = spark.createDataFrame([], schema)
        gpu_model = gpu_knn.fit(item_df)

        query_df = spark.createDataFrame([], schema="features array<float>")
        (_, _, knn_df_empty) = gpu_model.kneighbors(query_df)
        knn_df_empty.show()


def test_example_cosine() -> None:
    gpu_number = 1
    X = [
        (0, (1.0, 0.0)),
        (1, (1.0, 1.0)),
        (2, (-1.0, 1.0)),
    ]

    topk = 2
    metric = "cosine"
    algoParams = {"nlist": 1, "nprobe": 1}

    with CleanSparkSession() as spark:
        schema = f"id int, features array<float>"
        df = spark.createDataFrame(X, schema)
        gpu_knn = ApproximateNearestNeighbors(
            algorithm="ivfflat",
            algoParams=algoParams,
            k=topk,
            metric=metric,
            idCol="id",
            inputCol="features",
            num_workers=gpu_number,
        )
        gpu_model = gpu_knn.fit(df)
        _, _, knn_df = gpu_model.kneighbors(df)
        knn_collect = knn_df.collect()

        from sklearn.neighbors import NearestNeighbors

        X_features = np.array([row[1] for row in X])
        exact_nn = NearestNeighbors(
            algorithm="brute", metric="cosine", n_neighbors=topk
        )
        exact_nn.fit(X_features)
        distances, indices = exact_nn.kneighbors(X_features)

        assert array_equal([row["distances"] for row in knn_collect], distances)
        assert array_equal([row["indices"] for row in knn_collect], indices)


class ANNEvaluator:
    """
    obtain exact knn distances and indices
    """

    def __init__(self, X: np.ndarray, n_neighbors: int, metric: str) -> None:
        self.X = X
        self.n_neighbors = n_neighbors
        self.metric = metric
        if metric == "inner_product":
            from cuml import NearestNeighbors as cuNN

            cuml_knn = cuNN(
                algorithm="brute",
                n_neighbors=n_neighbors,
                output_type="numpy",
                metric=metric,
            )
            cuml_knn.fit(X)
            self.distances_exact, self.indices_exact = cuml_knn.kneighbors(X)
        else:
            from sklearn.neighbors import NearestNeighbors as skNN

            sk_knn = skNN(algorithm="brute", n_neighbors=n_neighbors, metric=metric)
            sk_knn.fit(X)
            self.distances_exact, self.indices_exact = sk_knn.kneighbors(X)

        assert self.distances_exact.shape == (len(self.X), self.n_neighbors)
        assert self.indices_exact.shape == (len(self.X), self.n_neighbors)

    def get_distances_exact(self) -> np.ndarray:
        return self.distances_exact

    def get_indices_exact(self) -> np.ndarray:
        return self.indices_exact

    def cal_avg_recall(self, indices_ann: np.ndarray) -> float:
        assert indices_ann.shape == self.indices_exact.shape
        assert indices_ann.shape == (len(self.X), self.n_neighbors)
        retrievals = [
            np.intersect1d(a, b) for a, b in zip(indices_ann, self.indices_exact)
        ]
        recalls = np.array([len(nns) / self.n_neighbors for nns in retrievals])
        return recalls.mean()

    def cal_avg_dist_gap(self, distances_ann: np.ndarray) -> float:
        assert distances_ann.shape == self.distances_exact.shape
        assert distances_ann.shape == (len(self.X), self.n_neighbors)
        gaps = np.abs(distances_ann - self.distances_exact)
        return gaps.mean()

    def compare_with_cuml_or_cuvs_sg(
        self,
        algorithm: str,
        algoParams: Optional[Dict[str, Any]],
        given_indices: np.ndarray,
        given_distances: np.ndarray,
        tolerance: float,
    ) -> None:
        # compare with cuml sg ANN on avg_recall and avg_dist_gap
        cuvssg_distances, cuvssg_indices = self.get_cuvs_sg_results(
            algorithm=algorithm, algoParams=algoParams
        )

        # compare cuml sg with given results
        avg_recall_cumlann = self.cal_avg_recall(cuvssg_indices)
        avg_recall = self.cal_avg_recall(given_indices)
        assert (avg_recall > avg_recall_cumlann) or abs(
            avg_recall - avg_recall_cumlann
        ) <= tolerance

        avg_dist_gap_cumlann = self.cal_avg_dist_gap(cuvssg_distances)
        avg_dist_gap = self.cal_avg_dist_gap(given_distances)
        assert (avg_dist_gap <= avg_dist_gap_cumlann) or abs(
            avg_dist_gap - avg_dist_gap_cumlann
        ) <= tolerance

    def get_cuml_sg_results(
        self,
        algorithm: str,
        algoParams: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        from cuml import NearestNeighbors as cuNN

        if algorithm == "ivfpq" and algoParams:
            if "usePrecomputedTables" not in algoParams:
                # the parameter is required in cython though ignored in C++.
                algoParams["usePrecomputedTables"] = False

        cuml_ivfflat = cuNN(
            algorithm=algorithm,
            algo_params=algoParams,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )
        cuml_ivfflat.fit(self.X)
        cumlsg_distances, cumlsg_indices = cuml_ivfflat.kneighbors(self.X)

        if self.metric == "euclidean" or self.metric == "l2":
            cumlsg_distances **= 2  # square up cuml distances to get l2 distances
        return (cumlsg_distances, cumlsg_indices)

    def get_cuvs_sg_results(
        self,
        algorithm: str = "cagra",
        algoParams: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if algorithm == "cagra":
            assert self.metric == "sqeuclidean"
            index_params, search_params = (
                ApproximateNearestNeighborsModel._cal_cagra_params_and_check(
                    algoParams=algoParams, metric=self.metric, topk=self.n_neighbors
                )
            )

            from cuvs.neighbors import cagra as cuvs_algo
        elif algorithm == "ivf_flat" or algorithm == "ivfflat":

            index_params, search_params = (
                ApproximateNearestNeighborsModel._cal_cuvs_ivf_flat_params_and_check(
                    algoParams=algoParams, metric=self.metric, topk=self.n_neighbors
                )
            )
            from cuvs.neighbors import ivf_flat as cuvs_algo
        elif algorithm in {"ivf_pq", "ivfpq"}:
            index_params, search_params = (
                ApproximateNearestNeighborsModel._cal_cuvs_ivf_pq_params_and_check(
                    algoParams=algoParams, metric=self.metric, topk=self.n_neighbors
                )
            )
            from cuvs.neighbors import ivf_pq as cuvs_algo
        else:
            assert False, f"unrecognized algorithm {algorithm}"

        import cupy as cp

        gpu_X = cp.array(self.X, dtype="float32")

        index = cuvs_algo.build(cuvs_algo.IndexParams(**index_params), gpu_X)
        sg_distances, sg_indices = cuvs_algo.search(
            cuvs_algo.SearchParams(**search_params), index, gpu_X, self.n_neighbors
        )

        if algorithm in {"ivf_pq", "ivfpq"}:
            from cuvs.neighbors import refine

            sg_distances, sg_indices = refine(
                gpu_X, gpu_X, sg_indices, self.n_neighbors, metric=self.metric
            )

        # convert results to cp array then to np array
        sg_distances = cp.array(sg_distances).get()
        sg_indices = cp.array(sg_indices).get()

        return (sg_distances, sg_indices)


def ann_algorithm_test_func(
    combo: Tuple[str, str, int, Optional[Dict[str, Any]], str],
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    expected_avg_recall: float = 0.95,
    expected_avg_dist_gap: float = 1e-4,
    distances_are_exact: bool = True,
    tolerance: float = 1e-4,
    n_neighbors: int = 50,
    cluster_std: float = 1.0,
) -> None:

    assert data_type in {
        np.float32,
        np.float64,
    }, "the test function applies to float dataset dtype only, as it scales the dataset by the average norm of rows"

    algorithm = combo[0]
    assert algorithm in {"ivfflat", "ivfpq", "cagra"}

    feature_type = combo[1]
    max_record_batch = combo[2]
    algoParams = combo[3]
    metric = combo[4]

    n_clusters = 10

    from cuml.neighbors import VALID_METRICS

    if algorithm in {"ivfflat", "ivfpq"}:
        assert VALID_METRICS[algorithm] == {
            "euclidean",
            "sqeuclidean",
            "cosine",
            "inner_product",
            "l2",
            "correlation",
        }
    else:
        assert algorithm == "cagra", f"unknown algorithm: {algorithm}"

    X, _ = make_blobs(
        n_samples=data_shape[0],
        n_features=data_shape[1],
        centers=n_clusters,
        random_state=0,
        cluster_std=cluster_std,
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    # set average norm sq to be 1 to allow comparisons with default error thresholds
    # below
    root_ave_norm_sq = np.sqrt(np.average(np.linalg.norm(X, ord=2, axis=1) ** 2))
    X = X / root_ave_norm_sq

    ann_evaluator = ANNEvaluator(X, n_neighbors, metric)

    y = np.arange(len(X))  # use label column as id column

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        data_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )
        assert label_col is not None
        data_df = data_df.withColumn(label_col, col(label_col).cast("long"))
        id_col = label_col

        knn_est = (
            ApproximateNearestNeighbors(
                algorithm=algorithm, algoParams=algoParams, k=n_neighbors, metric=metric
            )
            .setInputCol(features_col)
            .setIdCol(id_col)
        )

        # test kneighbors: obtain spark results
        knn_model = knn_est.fit(data_df)

        for obj in [knn_est, knn_model]:
            assert obj.getK() == n_neighbors
            assert obj.getAlgorithm() == algorithm
            assert obj.getAlgoParams() == algoParams
            if feature_type == "multi_cols":
                assert obj.getInputCols() == features_col
            else:
                assert obj.getInputCol() == features_col
            assert obj.getIdCol() == id_col

        query_df = data_df
        (item_df_withid, query_df_withid, knn_df) = knn_model.kneighbors(query_df)

        knn_df = knn_df.sort(f"query_{id_col}")
        knn_df_collect = knn_df.collect()

        # test kneighbors: collect spark results for comparison with cuml results
        distances = np.array([r["distances"] for r in knn_df_collect])
        indices = np.array([r["indices"] for r in knn_df_collect])

        # test kneighbors: compare top-1 nn indices(self) and distances(self)

        if metric != "inner_product":
            self_index = [knn[0] for knn in indices]
            assert np.all(self_index == y)

            self_distance = [dist[0] for dist in distances]
            assert array_equal(self_distance, [0.0] * len(X))

        # test kneighbors: compare with single-GPU cuml
        ann_evaluator.compare_with_cuml_or_cuvs_sg(
            algorithm, algoParams, indices, distances, tolerance
        )
        avg_recall = ann_evaluator.cal_avg_recall(indices)
        avg_dist_gap = ann_evaluator.cal_avg_dist_gap(distances)

        # test kneighbors: compare with sklearn brute NN on avg_recall and avg_dist_gap
        assert avg_recall >= expected_avg_recall
        if distances_are_exact:
            assert np.all(np.abs(avg_dist_gap) < expected_avg_dist_gap)

        # test exactNearestNeighborsJoin
        knnjoin_df = knn_model.approxSimilarityJoin(query_df_withid)

        ascending = False if metric == "inner_product" else True
        reconstructed_knn_df = reconstruct_knn_df(
            knnjoin_df,
            row_identifier_col=knn_model._getIdColOrDefault(),
            ascending=ascending,
        )
        reconstructed_collect = reconstructed_knn_df.collect()

        def assert_row_equal(r1: Row, r2: Row) -> None:
            assert r1[f"query_{id_col}"] == r2[f"query_{id_col}"]
            r1_distances = r1["distances"]
            r2_distances = r2["distances"]
            if algorithm == "ivfpq":
                # returned distances can be slightly different when running ivfpq multiple times due to quantization and randomness
                assert array_equal(
                    r1_distances, r2_distances, unit_tol=tolerance, total_tol=1e-3
                )
            else:
                assert array_equal(r1_distances, r2_distances, tolerance)

            assert len(r1["indices"]) == len(r2["indices"])
            assert len(r1["indices"]) == n_neighbors

            r1_i2d = dict(zip(r1["indices"], r1["distances"]))
            r2_i2d = dict(zip(r2["indices"], r2["distances"]))
            for i1, i2 in zip(r1["indices"], r2["indices"]):
                assert i1 == i2 or r1_i2d[i1] == pytest.approx(
                    r2_i2d[i2], abs=tolerance
                )

            if distances_are_exact:
                for i1, i2 in zip(r1["indices"], r2["indices"]):
                    if i1 != i2:
                        query_vec = X[r1[f"query_{id_col}"]]
                        assert cal_dist(query_vec, X[i1], metric) == pytest.approx(
                            cal_dist(query_vec, X[i2], metric), abs=tolerance
                        )

        assert len(reconstructed_collect) == len(knn_df_collect)
        if algorithm != "ivfpq" and not (algorithm == "ivfflat" and algoParams == None):
            # it is fine to skip ivfpq as long as other algorithms assert the same results of approxSimilarityJoin and kneighbors.
            # Also skip ivfflat when algoParams == None. Ivfflat probes only 1/50 of the clusters, leading to unstable results.
            # ivfpq shows non-deterministic distances due to kmeans initialization uses GPU memory runtime values.
            for i in range(len(reconstructed_collect)):
                r1 = reconstructed_collect[i]
                r2 = knn_df_collect[i]
                assert_row_equal(r1, r2)

        assert knn_est._cuml_params["metric"] == metric
        assert knn_model._cuml_params["metric"] == metric


@pytest.mark.parametrize(
    "combo",
    [
        (
            "ivfflat",
            "array",
            10000,
            None,
            "euclidean",
        ),
        (
            "ivfflat",
            "vector",
            2000,
            {"nlist": 10, "nprobe": 2},
            "euclidean",
        ),
        (
            "ivfflat",
            "multi_cols",
            5000,
            {"nlist": 20, "nprobe": 4},
            "euclidean",
        ),
        (
            "ivfflat",
            "array",
            2000,
            {"nlist": 10, "nprobe": 2},
            "sqeuclidean",
        ),
        ("ivfflat", "vector", 5000, {"nlist": 20, "nprobe": 4}, "l2"),
        (
            "ivfflat",
            "multi_cols",
            2000,
            {"nlist": 10, "nprobe": 2},
            "inner_product",
        ),
        (
            "ivfflat",
            "array",
            2000,
            {"nlist": 10, "nprobe": 2},
            "cosine",
        ),
    ],
)  # vector feature type will be converted to float32 to be compatible with cuml single-GPU NearestNeighbors Class
@pytest.mark.parametrize("data_type", [np.float32])
def test_ivfflat(
    combo: Tuple[str, str, int, Optional[Dict[str, Any]], str],
    data_type: np.dtype,
) -> None:
    algoParams = combo[3]

    # cuvs ivf_flat None sets nlist to 1000 and nprobe to 20, leading to unstable results when run multiple times
    expected_avg_recall: float = 0.95 if algoParams != None else 0.5
    expected_avg_dist_gap: float = 1e-4 if algoParams != None else 1e-2
    tolerance: float = 1e-4 if algoParams != None else 0.05
    data_shape: Tuple[int, int] = (10000, 50)
    ann_algorithm_test_func(
        combo=combo,
        data_shape=data_shape,
        data_type=data_type,
        expected_avg_recall=expected_avg_recall,
        expected_avg_dist_gap=expected_avg_dist_gap,
        tolerance=tolerance,
    )


@pytest.mark.parametrize(
    "algorithm,feature_type,max_records_per_batch,algo_params,metric",
    [
        (
            "ivfpq",
            "array",
            10000,
            {
                "nlist": 100,
                "nprobe": 20,
                "M": 20,
                "n_bits": 4,
            },
            "euclidean",
        ),
        (
            "ivfpq",
            "vector",
            200,
            {
                "nlist": 100,
                "nprobe": 20,
                "M": 40,
                "n_bits": 4,
            },
            "sqeuclidean",
        ),
        (
            "ivfpq",
            "multi_cols",
            5000,
            {
                "nlist": 100,
                "nprobe": 20,
                "M": 10,
                "n_bits": 8,
            },
            "l2",
        ),
        (
            "ivfpq",
            "array",
            2000,
            {
                "nlist": 100,
                "nprobe": 20,
                "M": 20,
                "n_bits": 4,
            },
            "inner_product",
        ),
        (
            "ivfpq",
            "array",
            3000,
            {
                "nlist": 100,
                "nprobe": 20,
                "M": 20,
                "n_bits": 4,
            },
            "cosine",
        ),
    ],
)
@pytest.mark.parametrize("data_shape", [(10000, 50)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
def test_ivfpq(
    algorithm: str,
    feature_type: str,
    max_records_per_batch: int,
    algo_params: Dict[str, Any],
    metric: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
) -> None:
    """
    (1) Currently the usePrecomputedTables is not used in cuml C++.

    (2) ivfpq has become unstable in 24.10. It does not get passed with algoParam {"nlist" : 10, "nprobe" : 2, "M": 2, "n_bits": 4} in ci where test_ivfflat is run beforehand. avg_recall shows large variance, depending on the quantization accuracy. This can be fixed by increasing nlist, nprobe, M, and n_bits. Note ivf_pq is non-deterministic, and it seems due to kmeans initialization leveraging runtime values of GPU memory.

    (3) In ivfpq, when the dataset itself is used as queries, it is observed sometimes that the top-1 indice may not be self, and top-1 distance may not be zero.
    This is because ivfpq internally uses approximated distance, i.e. the distance of the query vector to the center of quantized item.
    """
    combo = (algorithm, feature_type, max_records_per_batch, algo_params, metric)
    expected_avg_recall = 0.4
    distances_are_exact = True
    expected_avg_dist_gap = 0.05
    tolerance = 0.05  # tolerance increased to be more stable due to quantization and randomness in ivfpq, especially when expected_recall is low.

    cluster_std = (
        1.0 if metric != "cosine" else 10.0
    )  # Increasing cluster_std for cosine to make dataset more randomized and separable.

    ann_algorithm_test_func(
        combo=combo,
        data_shape=data_shape,
        data_type=data_type,
        expected_avg_recall=expected_avg_recall,
        expected_avg_dist_gap=expected_avg_dist_gap,
        distances_are_exact=distances_are_exact,
        tolerance=tolerance,
        cluster_std=cluster_std,
    )


@pytest.mark.parametrize(
    "algorithm,feature_type,max_records_per_batch,algo_params,metric",
    [
        (
            "cagra",
            "array",
            10000,
            {
                "intermediate_graph_degree": 128,
                "graph_degree": 64,
                "build_algo": "ivf_pq",
            },
            "sqeuclidean",
        ),
        (
            "cagra",
            "multi_cols",
            3000,
            {
                "intermediate_graph_degree": 256,
                "graph_degree": 128,
                "build_algo": "nn_descent",
            },
            "sqeuclidean",
        ),
        (
            "cagra",
            "vector",
            5000,
            {
                "build_algo": "ivf_pq",
                "itopk_size": 96,  # cuVS increases this to multiple of 32 and requires it to be larger than or equal to k.
                "search_width": 2,
                "num_random_samplings": 2,
            },
            "sqeuclidean",
        ),
    ],
)
@pytest.mark.parametrize("data_shape", [(10000, 50)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
def test_cagra(
    algorithm: str,
    feature_type: str,
    max_records_per_batch: int,
    algo_params: Dict[str, Any],
    metric: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    n_neighbors: int = 50,
) -> None:
    """
    TODO: support compression index param
    """

    VALID_BUILD_ALGO = {"ivf_pq", "nn_descent"}
    assert algo_params["build_algo"] in {
        "ivf_pq",
        "nn_descent",
    }, f"cagra currently supports build_algo only in {VALID_BUILD_ALGO}"

    combo = (algorithm, feature_type, max_records_per_batch, algo_params, metric)
    expected_avg_recall = 0.99
    distances_are_exact = True
    tolerance = 2e-3

    ann_algorithm_test_func(
        combo=combo,
        data_shape=data_shape,
        data_type=data_type,
        expected_avg_recall=expected_avg_recall,
        distances_are_exact=distances_are_exact,
        tolerance=tolerance,
        n_neighbors=n_neighbors,
    )


@pytest.mark.parametrize(
    "feature_type,data_type",
    [
        ("vector", np.float64),
        ("multi_cols", np.float64),
        ("multi_cols", np.int16),
        ("array", np.int64),
    ],
)
@pytest.mark.slow
def test_cagra_dtype(
    feature_type: str,
    data_type: np.dtype,
) -> None:

    algorithm = "cagra"
    algo_params = {
        "intermediate_graph_degree": 128,
        "graph_degree": 64,
        "build_algo": "ivf_pq",
    }

    gpu_number = 1
    n_neighbors = 2
    metric = "sqeuclidean"
    X = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [40.0, 40.0],
            [50.0, 50.0],
        ],
        dtype="int32",
    )
    X = X.astype(data_type)
    y = np.array(range(len(X)))
    with CleanSparkSession() as spark:
        data_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )

        gpu_knn = ApproximateNearestNeighbors(
            num_workers=gpu_number,
            inputCol=features_col,
            idCol=label_col,
            k=n_neighbors,
            metric=metric,
            algorithm=algorithm,
            algoParams=algo_params,
        )

        gpu_model = gpu_knn.fit(data_df)
        (_, _, knn_df) = gpu_model.kneighbors(data_df)
        knn_df.show()


@pytest.mark.parametrize(
    "algorithm,feature_type,max_records_per_batch,algo_params",
    [
        (
            "cagra",
            "vector",
            5000,
            {
                "build_algo": "ivf_pq",
                "itopk_size": 32,
            },
        ),
    ],
)
@pytest.mark.parametrize("data_type", [np.float32])
def test_cagra_params(
    algorithm: str,
    feature_type: str,
    max_records_per_batch: int,
    algo_params: Dict[str, Any],
    data_type: np.dtype,
    caplog: LogCaptureFixture,
) -> None:

    data_shape = (1000, 20)
    metric = "sqeuclidean"
    itopk_size = 64 if "itopk_size" not in algo_params else algo_params["itopk_size"]

    internal_topk_size = math.ceil(itopk_size / 32) * 32
    n_neighbors = 50
    error_msg = ""
    if internal_topk_size < n_neighbors:
        error_msg = f"cagra increases itopk_size to be closest multiple of 32 and expects the value, i.e. {internal_topk_size}, to be larger than or equal to k, i.e. {n_neighbors}."

    with pytest.raises(ValueError, match=error_msg):
        test_cagra(
            algorithm,
            feature_type,
            max_records_per_batch,
            algo_params,
            metric,
            data_shape,
            data_type,
            n_neighbors=n_neighbors,
        )

    # test intermediate_graph_degree restriction on ivf_pq
    algo_params["itopk_size"] = 64
    algo_params["intermediate_graph_degree"] = 257
    error_msg = f"cagra with ivf_pq build_algo expects intermediate_graph_degree (257) to be smaller than 256."
    with pytest.raises(Exception):
        test_cagra(
            algorithm,
            feature_type,
            max_records_per_batch,
            algo_params,
            metric,
            data_shape,
            data_type,
            n_neighbors=n_neighbors,
        )
        assert error_msg in caplog.text

    # test metric restriction
    algo_params["intermediate_graph_degree"] = 255
    metric = "euclidean"
    error_msg = f"when using 'cagra' algorithm, the metric must be explicitly set to 'sqeuclidean'."
    with pytest.raises(AssertionError, match=error_msg):
        test_cagra(
            algorithm,
            feature_type,
            max_records_per_batch,
            algo_params,
            metric,
            data_shape,
            data_type,
            n_neighbors=n_neighbors,
        )


@pytest.mark.parametrize(
    "combo",
    [
        ("ivfflat", "array", 2000, {"nlist": 10, "nprobe": 2}, "sqeuclidean"),
        ("ivfflat", "vector", 5000, {"nlist": 20, "nprobe": 4}, "l2"),
        ("ivfflat", "multi_cols", 2000, {"nlist": 10, "nprobe": 2}, "inner_product"),
    ],
)
@pytest.mark.parametrize("data_shape", [(4000, 3000)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.slow
def test_ivfflat_wide_matrix(
    combo: Tuple[str, str, int, Optional[Dict[str, Any]], str],
    data_shape: Tuple[int, int],
    data_type: np.dtype,
) -> None:
    """
    It seems adding a column with df.withColumn can be very slow, if df already has many columns (e.g. 3000).
    One strategy is to avoid df.withColumn on wide df and use df.select instead.
    """
    import time

    start = time.time()
    ann_algorithm_test_func(combo=combo, data_shape=data_shape, data_type=data_type)
    duration_sec = time.time() - start
    assert duration_sec < 3 * 60


@pytest.mark.parametrize(
    "algorithm,feature_type",
    [
        (
            "ivfpq",
            "array",
        ),
        (
            "ivfflat",
            "vector",
        ),
    ],
)
@pytest.mark.parametrize("data_type", [np.float32])
def test_return_fewer_k(
    algorithm: str,
    feature_type: str,
    data_type: np.dtype,
) -> None:
    """
    This tests the corner case where there are less than k neighbors found due to nprobe too small.
    More details can be found at the docstring of class ApproximateNearestNeighbors.
    """
    assert algorithm in {"ivfpq", "ivfflat"}
    metric = "euclidean"
    gpu_number = 1
    k = 4
    algo_params = {
        "nlist": k,
        "nprobe": 1,
    }

    if algorithm == "ivfpq":
        algo_params.update({"M": 2, "n_bits": 4})

    X = np.array(
        [
            (
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
            ),
            (
                2.0,
                2.0,
            ),
            (
                2.0,
                2.0,
            ),
        ]
    )
    y = np.arange(len(X))  # use label column as id column

    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y, label_dtype=np.dtype(np.int64)
        )

        est = ApproximateNearestNeighbors(
            num_workers=gpu_number,
            algorithm=algorithm,
            algoParams=algo_params,
            metric=metric,
            k=k,
            inputCol="features",
            idCol=label_col,
        )
        model = est.fit(df)
        _, _, knn_df = model.kneighbors(df)
        knn_df_collect = knn_df.collect()

        int64_max = np.iinfo("int64").max
        float_inf = float("inf")

        # check result details
        # note that cuVS produces non-deterministic results during the indexing process
        indices_none_probed = [int64_max, int64_max, int64_max, int64_max]
        distances_none_probed = [float_inf, float_inf, float_inf, float_inf]

        def check_row_results(
            i: int,
            indices_if_probed: List[int],
            distances_if_probed: List[float],
            res_indices: List[List[int]],
            res_distances: List[List[float]],
        ) -> None:
            assert i == 0 or i == 2
            j = i + 1
            assert res_indices[i] == res_indices[j]
            assert res_distances[i] == res_distances[j]
            if res_indices[i] == indices_none_probed:
                assert res_distances[i] == distances_none_probed
            else:
                assert res_indices[i] == indices_if_probed
                assert res_distances[i] == distances_if_probed

        spark_indices = [row["indices"] for row in knn_df_collect]
        spark_distances = [row["distances"] for row in knn_df_collect]
        check_row_results(
            0,
            [0, 1, 0, 0],
            [0.0, 0.0, float_inf, float_inf],
            spark_indices,
            spark_distances,
        )
        check_row_results(
            2,
            [2, 3, 2, 2],
            [0.0, 0.0, float_inf, float_inf],
            spark_indices,
            spark_distances,
        )

        # ensure consistency of cuvs (non-deterministic indexing of ivfflat/ivfpq + inconsistent outputs of refine)
        ann_evaluator = ANNEvaluator(X, k, metric)
        sg_distances, sg_indices = ann_evaluator.get_cuvs_sg_results(
            algorithm, algo_params
        )
        if algorithm == "ivfpq":

            def align_after_ivfpq_refine(
                in_distances: np.ndarray, in_indices: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
                """
                # Refine API behavior in version 24.12:
                # (1) converts all LONG_MAX values in the indices array to -1
                # (2) sets the corresponding distances for -1 indices to infinity (inf).
                # (3) updates the distances array for top-1 indices to reflect the top-1 distance value
                #     Example: for top-4 indices [2, 3, 2, 2], the distances array would be updated to [0., 0., 0., 0.].
                # Spark Rapids ML assumes the future Refine API will align with ivfflat and ivfpq in handing the
                # fewer_than_k_items issue, and currently implements this alignment in the current release.
                """
                import cupy as cp

                out_distances = in_distances.copy()
                out_indices = in_indices.copy()

                out_indices[out_indices == -1] = np.iinfo("int64").max
                out_distances[out_indices == np.iinfo("int64").max] = float("inf")

                # for the case top-1 nn got filled into indices
                top1_ind = out_indices[:, 0]
                rest_indices = out_indices[:, 1:]
                rest_distances = out_distances[:, 1:]
                rest_distances[rest_indices == top1_ind[:, cp.newaxis]] = float("inf")
                return (out_distances, out_indices)

            sg_distances, sg_indices = align_after_ivfpq_refine(
                sg_distances, sg_indices
            )

        check_row_results(
            0,
            [0, 1, 0, 0],
            [0.0, 0.0, float_inf, float_inf],
            sg_indices.tolist(),
            sg_distances.tolist(),
        )
        check_row_results(
            2,
            [2, 3, 2, 2],
            [0.0, 0.0, float_inf, float_inf],
            sg_indices.tolist(),
            sg_distances.tolist(),
        )

#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import cupy as cp
import numpy as np
import pytest
import scipy
from _pytest.logging import LogCaptureFixture
from cuml.datasets import make_blobs
from cuml.metrics import trustworthiness
from pyspark.ml.linalg import SparseVector
from pyspark.sql.functions import array
from scipy.sparse import csr_matrix
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


def _load_sparse_data(
    n_rows: int, n_cols: int, nnz: int, binary: bool = False, normalize: bool = False
) -> Tuple[List[Tuple[SparseVector]], csr_matrix]:
    """
    Generate random binary or real-valued sparse data with approximately nnz non-zeros per row.
    If normalize is True, the data is normalized to have unit row sum (e.g., for hellinger distance).
    """
    density = nnz / n_cols

    if binary:
        data_rvs = lambda n: np.ones(n)
    else:
        data_rvs = lambda n: np.random.uniform(0.1, 1.0, n)

    csr_mat = scipy.sparse.random(
        n_rows,
        n_cols,
        density=density,
        format="csr",
        dtype=np.float32,
        data_rvs=data_rvs,
    )

    if normalize:
        row_sums = np.array(csr_mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        row_sum_diag = scipy.sparse.diags(1.0 / row_sums)
        csr_mat = row_sum_diag @ csr_mat
        assert np.allclose(np.array(csr_mat.sum(axis=1)).flatten(), 1.0)

    # Convert CSR matrix to SparseVectors
    data = []
    for i in range(n_rows):
        indices = csr_mat.indices[csr_mat.indptr[i] : csr_mat.indptr[i + 1]]
        values = csr_mat.data[csr_mat.indptr[i] : csr_mat.indptr[i + 1]]
        sparse_vector = SparseVector(n_cols, dict(zip(indices, values)))
        data.append((sparse_vector,))

    return data, csr_mat


def _load_dataset(dataset: str, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load either the digits: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
    or the iris dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
    and repeat the data to have n_rows rows, and add some gaussian noise.
    """
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


def _assert_umap_model(
    model: UMAPModel,
    input_raw_data: Union[np.ndarray, csr_matrix],
) -> None:
    embedding = model.embedding
    raw_data = model.rawData
    assert embedding.shape == (
        input_raw_data.shape[0],
        model.cuml_params["n_components"],
    )
    assert raw_data.shape == input_raw_data.shape
    if isinstance(input_raw_data, csr_matrix):
        assert isinstance(raw_data, csr_matrix)
        assert model._sparse_fit
        assert (raw_data != input_raw_data).nnz == 0
        try:
            assert (
                np.all(raw_data.indices == input_raw_data.indices)
                and np.all(raw_data.indptr == input_raw_data.indptr)
                and np.allclose(raw_data.data, input_raw_data.data)
            )
        except AssertionError:
            # If exact match fails, compare the dense versions, since indices can get reordered if we normalize
            assert np.array_equal(raw_data.toarray(), input_raw_data.toarray())
    else:
        assert not model._sparse_fit
        assert np.array_equal(raw_data, input_raw_data)
    assert model.dtype == "float32"
    assert model.n_cols == input_raw_data.shape[1]


def _local_umap_trustworthiness(
    local_X: Union[np.ndarray, csr_matrix],
    local_y: np.ndarray,
    n_neighbors: int,
    supervised: bool,
    sparse: bool = False,
) -> float:
    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors, random_state=42, init="random")
    y_train = local_y if supervised else None
    local_model.fit(local_X, y=y_train)
    embedding = local_model.transform(local_X)

    if sparse:
        assert isinstance(local_X, csr_matrix)
        local_X = local_X.toarray()

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
            data_df, feature_cols, label_col = create_pyspark_dataframe(
                spark, feature_type, dtype, local_X, local_y
            )
            assert label_col is not None
            umap_estimator.setLabelCol(label_col)
        else:
            data_df, feature_cols, _ = create_pyspark_dataframe(
                spark, feature_type, dtype, local_X, None
            )

        data_df = data_df.repartition(n_parts)
        if isinstance(feature_cols, list):
            umap_estimator.setFeaturesCols(feature_cols)
        else:
            umap_estimator.setFeaturesCol(feature_cols)

        umap_model = umap_estimator.fit(data_df)
        pdf = umap_model.transform(data_df).toPandas()

        embedding = cp.asarray(pdf["embedding"].to_list()).astype(cp.float32)
        if isinstance(feature_cols, list):
            input = pdf[feature_cols].to_numpy()
        else:
            input = pdf[feature_cols].to_list()

        input = cp.asarray(input).astype(cp.float32)

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

    return trust_diff <= 0.1


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
        param.name: value for param, value in UMAP().extractParamMap().items()
    }

    cuml_params = get_default_cuml_parameters(
        cuml_classes=[cumlUMAP],
        excludes=[
            "callback",
            "handle",
            "hash_input",
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
        umap = UMAP(**nondefault_params)  # type: ignore
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


def test_umap_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [
        ({"n_neighbors": 21}, {"n_neighbors": 21}),
        ({"n_components": 23}, {"n_components": 23}),
        ({"metric": "cosine"}, {"metric": "cosine"}),
        ({"metric_kwds": {"p": 5}}, {"metric_kwds": {"p": 5}}),
        ({"n_epochs": 132}, {"n_epochs": 132}),
        ({"learning_rate": 0.19}, {"learning_rate": 0.19}),
        ({"init": "random"}, {"init": "random"}),
        ({"min_dist": 0.24}, {"min_dist": 0.24}),
        ({"spread": 0.24}, {"spread": 0.24}),
        ({"set_op_mix_ratio": 0.94}, {"set_op_mix_ratio": 0.94}),
        ({"local_connectivity": 0.98}, {"local_connectivity": 0.98}),
        ({"repulsion_strength": 0.99}, {"repulsion_strength": 0.99}),
        ({"negative_sample_rate": 7}, {"negative_sample_rate": 7}),
        ({"transform_queue_size": 0.77}, {"transform_queue_size": 0.77}),
        ({"a": 1.77}, {"a": 1.77}),
        ({"b": 2.77}, {"b": 2.77}),
        (
            {"random_state": 81},
            {"random_state": 81},
        ),
        ({"build_algo": "nn_descent"}, {"build_algo": "nn_descent"}),
        (
            {"build_kwds": {"nnd_graph_degree": 117}},
            {"build_kwds": {"nnd_graph_degree": 117}},
        ),
        ({"sample_fraction": 0.74}, None),
        ({"enable_sparse_data_optim": True}, None),
        ({"verbose": True}, {"verbose": True}),
    ]

    for params in param_list:
        _test_est_copy(UMAP, params[0], params[1])


@pytest.mark.parametrize("sparse_fit", [True, False])
def test_umap_model_persistence(
    sparse_fit: bool, gpu_number: int, tmp_path: str
) -> None:
    import os
    import re

    import pyspark
    from packaging import version

    with CleanSparkSession() as spark:

        n_rows = 5000
        n_cols = 200

        if sparse_fit:
            if version.parse(pyspark.__version__) < version.parse("3.4.0"):
                import logging

                err_msg = "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function for SparseVector. "
                "The test case will be skipped. Please install pyspark>=3.4."
                logging.info(err_msg)
                return

            sparse_vec_data, input_raw_data = _load_sparse_data(n_rows, n_cols, 30)
            df = spark.createDataFrame(sparse_vec_data, ["features"])
        else:
            X, _ = make_blobs(
                n_rows,
                n_cols,
                centers=5,
                cluster_std=0.1,
                dtype=cp.float32,
                random_state=10,
            )
            pyspark_type = "float"
            feature_cols = [f"c{i}" for i in range(X.shape[1])]
            schema = [f"{c} {pyspark_type}" for c in feature_cols]
            df = spark.createDataFrame(X.tolist(), ",".join(schema))
            df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)
            input_raw_data = X.get()

        umap = UMAP(num_workers=gpu_number).setFeaturesCol("features")

        umap_model = umap.fit(df)
        _assert_umap_model(umap_model, input_raw_data)

        # Model persistence
        path = tmp_path + "/umap_tests"
        model_path = f"{path}/umap_model"
        umap_model.write().overwrite().save(model_path)

        try:
            umap_model.write().save(model_path)
            assert False, "Overwriting should not be permitted"
        except Exception as e:
            assert re.search(r"Output directory .* already exists", str(e))

        try:
            umap_model.write().overwrite().save(model_path)
        except:
            assert False, "Overwriting should be permitted"

        # double check expected files/directories
        model_dir_contents = os.listdir(model_path)
        data_dir_contents = os.listdir(f"{model_path}/data")
        assert set(model_dir_contents) == {"data", "metadata"}
        if sparse_fit:
            assert set(data_dir_contents) == {
                "metadata.json",
                "embedding_.parquet",
                "raw_data_csr",
            }
            assert set(os.listdir(f"{model_path}/data/raw_data_csr")) == {
                "indptr.parquet",
                "indices_data.parquet",
            }
        else:
            assert set(data_dir_contents) == {
                "metadata.json",
                "embedding_.parquet",
                "raw_data_.parquet",
            }

        # make sure we can overwrite
        umap_model._cuml_params["n_neighbors"] = 10
        umap_model._cuml_params["set_op_mix_ratio"] = 0.4
        umap_model.write().overwrite().save(model_path)

        umap_model_loaded = UMAPModel.load(model_path)
        assert umap_model_loaded._cuml_params["n_neighbors"] == 10
        assert umap_model_loaded._cuml_params["set_op_mix_ratio"] == 0.4
        _assert_umap_model(umap_model_loaded, input_raw_data)


@pytest.mark.parametrize("maxRecordsPerBatch", ["2000"])
@pytest.mark.parametrize("BROADCAST_LIMIT", [8 << 15])
@pytest.mark.parametrize("sparse_fit", [True, False])
def test_umap_chunking(
    gpu_number: int, maxRecordsPerBatch: str, BROADCAST_LIMIT: int, sparse_fit: bool
) -> None:

    n_rows = int(int(maxRecordsPerBatch) * 2.5)
    n_cols = 3000
    random_state = 42

    with CleanSparkSession() as spark:
        spark.conf.set(
            "spark.sql.execution.arrow.maxRecordsPerBatch", maxRecordsPerBatch
        )

        if sparse_fit:
            import pyspark
            from packaging import version

            if version.parse(pyspark.__version__) < version.parse("3.4.0"):
                import logging

                err_msg = "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function for SparseVector. "
                "The test case will be skipped. Please install pyspark>=3.4."
                logging.info(err_msg)
                return

            sparse_vec_data, input_raw_data = _load_sparse_data(n_rows, n_cols, 30)
            df = spark.createDataFrame(sparse_vec_data, ["features"])
            nbytes = input_raw_data.data.nbytes
        else:
            X, _ = make_blobs(
                n_rows,
                n_cols,
                centers=5,
                cluster_std=0.1,
                dtype=cp.float32,
                random_state=random_state,
            )
            pyspark_type = "float"
            feature_cols = [f"c{i}" for i in range(X.shape[1])]
            schema = [f"{c} {pyspark_type}" for c in feature_cols]
            df = spark.createDataFrame(X.tolist(), ",".join(schema))
            df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)
            input_raw_data = X.get()
            nbytes = input_raw_data.nbytes

        umap = UMAP(num_workers=gpu_number, random_state=random_state).setFeaturesCol(
            "features"
        )

        assert umap.max_records_per_batch == int(maxRecordsPerBatch)
        assert nbytes > BROADCAST_LIMIT

        umap_model = umap.fit(df)
        umap_model.BROADCAST_LIMIT = BROADCAST_LIMIT

        _assert_umap_model(umap_model, input_raw_data)

        pdf = umap_model.transform(df).toPandas()
        embedding = np.vstack(pdf["embedding"]).astype(np.float32)
        input = np.vstack(pdf["features"]).astype(np.float32)

        dist_umap = trustworthiness(input, embedding, n_neighbors=15, batch_size=5000)
        loc_umap = _local_umap_trustworthiness(
            local_X=input_raw_data,
            local_y=np.zeros(0),
            n_neighbors=15,
            supervised=False,
            sparse=sparse_fit,
        )
        trust_diff = loc_umap - dist_umap

        assert trust_diff <= 0.07


def test_umap_sample_fraction(gpu_number: int) -> None:

    n_rows = 5000
    n_cols = 10
    random_state = 42
    sample_fraction = 0.5

    X, _ = make_blobs(
        n_rows,
        n_cols,
        centers=5,
        cluster_std=0.1,
        dtype=cp.float32,
        random_state=random_state,
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

        threshold = 2 * np.sqrt(
            n_rows * sample_fraction * (1 - sample_fraction)
        )  # 2 std devs

        embedding = umap_model.embedding
        raw_data = umap_model.rawData
        assert np.abs(n_rows * sample_fraction - embedding.shape[0]) <= threshold
        assert np.abs(n_rows * sample_fraction - raw_data.shape[0]) <= threshold


@pytest.mark.parametrize("metric", ["l2", "euclidean", "cosine", "l1"])
def test_umap_build_algo(gpu_number: int, metric: str) -> None:

    n_rows = 10000
    # cuml 25.06 UMAP is unstable for low dimensions
    n_cols = 100
    random_state = 42

    X, _ = make_blobs(
        n_rows,
        n_cols,
        centers=5,
        cluster_std=0.1,
        dtype=cp.float32,
        random_state=random_state,
    )

    with CleanSparkSession() as spark:
        pyspark_type = "float"
        feature_cols = [f"c{i}" for i in range(X.shape[1])]
        schema = [f"{c} {pyspark_type}" for c in feature_cols]
        df = spark.createDataFrame(X.tolist(), ",".join(schema)).coalesce(1)
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)

        build_algo = "nn_descent"
        build_kwds = {
            "nnd_graph_degree": 64,
            "nnd_intermediate_graph_degree": 128,
            "nnd_max_iterations": 40,
            "nnd_termination_threshold": 0.0001,
            "nnd_n_clusters": 5,
            "nnd_overlap_factor": 2,
        }

        umap = UMAP(
            num_workers=gpu_number,
            random_state=random_state,
            build_algo=build_algo,
            build_kwds=build_kwds,
            metric=metric,
        ).setFeaturesCol("features")

        # TODO: cuml nn_descent currently relies on the RAFT implementation (only supports L2/euclidean);
        # once they move to cuvs, it should also support cosine
        if metric not in ["l2", "euclidean", "cosine"]:
            try:
                umap.fit(df)
                assert False, f"Metric '{metric}' should throw an error with nn_descent"
            except Exception:
                assert f"NotImplementedError: Metric '{metric}' not supported" in str(
                    traceback.format_exc()
                )
        else:
            umap_model = umap.fit(df)

            _assert_umap_model(umap_model, X.get())

            pdf = umap_model.transform(df).toPandas()
            embedding = cp.asarray(pdf["embedding"].to_list()).astype(cp.float32)
            input = cp.asarray(pdf["features"].to_list()).astype(cp.float32)

            dist_umap = trustworthiness(
                input, embedding, n_neighbors=15, batch_size=10000
            )
            loc_umap = _local_umap_trustworthiness(
                local_X=X, local_y=np.zeros(0), n_neighbors=15, supervised=False
            )
            trust_diff = loc_umap - dist_umap

            assert trust_diff <= 0.07


@pytest.mark.parametrize("n_rows", [3000])
@pytest.mark.parametrize("n_cols", [64])
@pytest.mark.parametrize("nnz", [12])
@pytest.mark.parametrize(
    "metric",
    [
        pytest.param("jaccard", id="jaccard"),
        pytest.param("euclidean", id="euclidean"),
        pytest.param("hellinger", id="hellinger"),
        pytest.param("correlation", marks=pytest.mark.slow, id="correlation"),
        pytest.param("cosine", marks=pytest.mark.slow, id="cosine"),
        pytest.param("chebyshev", marks=pytest.mark.slow, id="chebyshev"),
        pytest.param("manhattan", marks=pytest.mark.slow, id="manhattan"),
        pytest.param("canberra", marks=pytest.mark.slow, id="canberra"),
        pytest.param("sqeuclidean", marks=pytest.mark.slow, id="sqeuclidean"),
        pytest.param("minkowski", marks=pytest.mark.slow, id="minkowski"),
        pytest.param("hamming", marks=pytest.mark.slow, id="hamming"),
    ],
)  # Test all metrics if runslow is enabled, otherwise just do a few
def test_umap_sparse_vector(
    n_rows: int, n_cols: int, nnz: int, metric: str, gpu_number: int, tmp_path: str
) -> None:
    import pyspark
    from cuml.manifold import UMAP as cumlUMAP
    from packaging import version

    if version.parse(pyspark.__version__) < version.parse("3.4.0"):
        import logging

        err_msg = "pyspark < 3.4 is detected. Cannot import pyspark `unwrap_udt` function for SparseVector. "
        "The test case will be skipped. Please install pyspark>=3.4."
        logging.info(err_msg)
        return

    # Hellinger measures similarity between probability distributions; normalize to prevent distances from collapsing to zero
    normalize = metric == "hellinger"
    use_binary = metric in ["jaccard", "hamming"]
    sparse_vec_data, input_raw_data = _load_sparse_data(
        n_rows, n_cols, nnz, use_binary, normalize
    )

    with CleanSparkSession() as spark:
        df = spark.createDataFrame(sparse_vec_data, ["features"])

        umap_estimator = UMAP(
            metric=metric, num_workers=gpu_number, random_state=42
        ).setFeaturesCol("features")
        umap_model = umap_estimator.fit(df)
        embedding = umap_model.embedding

        # Ensure internal and input CSR data match
        _assert_umap_model(umap_model, input_raw_data)

        # Local vs dist trustworthiness check
        output = umap_model.transform(df).toPandas()
        embedding = cp.asarray(output["embedding"].to_list())
        dist_umap = trustworthiness(input_raw_data.toarray(), embedding, n_neighbors=15)

        local_model = cumlUMAP(n_neighbors=15, random_state=42, metric=metric)
        local_model.fit(input_raw_data)
        embedding = local_model.transform(input_raw_data)
        loc_umap = trustworthiness(input_raw_data.toarray(), embedding, n_neighbors=15)

        trust_diff = loc_umap - dist_umap

        assert trust_diff <= 0.07


@pytest.mark.parametrize("knn_graph_format", ["sparse", "dense", "tuple"])
def test_umap_precomputed_knn(
    knn_graph_format: str, tmp_path: str, gpu_number: int
) -> None:

    n_rows = 10000
    n_cols = 50
    random_state = 42

    X, _ = make_blobs(
        n_rows,
        n_cols,
        centers=5,
        cluster_std=0.1,
        dtype=cp.float32,
        random_state=random_state,
    )

    k = 15
    knn_metric = "sqeuclidean"

    # Test a few different KNN implementations
    if knn_graph_format == "tuple":
        from cuvs.neighbors import cagra

        X_row_major = cp.ascontiguousarray(X)
        build_params = cagra.IndexParams(metric=knn_metric)
        index = cagra.build(build_params, X_row_major)
        distances, neighbors = cagra.search(cagra.SearchParams(), index, X_row_major, k)
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)
        precomputed_knn = (neighbors.get(), distances.get())
        assert isinstance(precomputed_knn[0], np.ndarray) and isinstance(
            precomputed_knn[1], np.ndarray
        )
    elif knn_graph_format == "sparse":
        from cuml.neighbors import NearestNeighbors

        knn_model = NearestNeighbors(n_neighbors=k, metric=knn_metric)
        knn_model.fit(X)
        precomputed_knn = knn_model.kneighbors_graph(X).get()
        assert isinstance(precomputed_knn, scipy.sparse.spmatrix)
    else:  # knn_graph_format == "dense"
        from cuml.metrics import pairwise_distances

        precomputed_knn = pairwise_distances(X, metric=knn_metric).get()
        assert isinstance(precomputed_knn, np.ndarray)

    with CleanSparkSession() as spark:
        pyspark_type = "float"
        feature_cols = [f"c{i}" for i in range(X.shape[1])]
        schema = [f"{c} {pyspark_type}" for c in feature_cols]
        df = spark.createDataFrame(X.tolist(), ",".join(schema))
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)

        try:
            umap = UMAP(
                num_workers=gpu_number,
                metric="sqeuclidean",
                sample_fraction=0.5,
                precomputed_knn=precomputed_knn,
            )
            umap.fit(df)
            assert False, "We should have raised an error"
        except ValueError as e:
            assert (
                "precomputed_knn and sample_fraction < 1.0 cannot be used simultaneously"
                in str(e)
            )

        try:
            umap = UMAP(
                num_workers=gpu_number,
                metric="sqeuclidean",
                precomputed_knn=precomputed_knn,
            )
            umap.setSampleFraction(0.5).fit(df)
            assert False, "We should have raised an error"
        except ValueError as e:
            assert (
                "precomputed_knn and sample_fraction < 1.0 cannot be used simultaneously"
                in str(e)
            )

        umap = UMAP(
            num_workers=gpu_number,
            metric="sqeuclidean",
            random_state=random_state,
            precomputed_knn=precomputed_knn,
        ).setFeaturesCol("features")

        # Double check that the precomputed_knn attribute is set correctly
        model_precomputed_knn = umap.cuml_params.get("precomputed_knn")
        assert model_precomputed_knn is not None
        if isinstance(precomputed_knn, tuple):
            assert np.array_equal(precomputed_knn[0], model_precomputed_knn[0])
            assert np.array_equal(precomputed_knn[1], model_precomputed_knn[1])
        elif isinstance(precomputed_knn, scipy.sparse.spmatrix):
            assert (precomputed_knn != model_precomputed_knn).nnz == 0
            assert (
                np.all(precomputed_knn.indices == model_precomputed_knn.indices)
                and np.all(precomputed_knn.indptr == model_precomputed_knn.indptr)
                and np.allclose(precomputed_knn.data, model_precomputed_knn.data)
            )
        else:
            assert np.array_equal(model_precomputed_knn, precomputed_knn)

        # Call fit, which will delete the precomputed_knn attribute
        umap_model = umap.fit(df)

        assert umap_model.cuml_params.get("precomputed_knn") is None
        _assert_umap_model(umap_model, X.get())

        pdf = umap_model.transform(df).toPandas()
        embedding = cp.asarray(pdf["embedding"].to_list()).astype(cp.float32)
        input = cp.asarray(pdf["features"].to_list()).astype(cp.float32)

        dist_umap = trustworthiness(input, embedding, n_neighbors=k, batch_size=10000)
        loc_umap = _local_umap_trustworthiness(
            local_X=X, local_y=np.zeros(0), n_neighbors=k, supervised=False
        )
        trust_diff = loc_umap - dist_umap

        assert trust_diff <= 0.07


def test_handle_param_spark_confs() -> None:
    """
    Test _handle_param_spark_confs method that reads Spark configuration values
    for parameters when they are not set in the constructor.
    """
    # Parameters are NOT set in constructor (should be picked up from Spark confs)
    with CleanSparkSession(
        {
            "spark.rapids.ml.verbose": "5",
            "spark.rapids.ml.float32_inputs": "false",
            "spark.rapids.ml.num_workers": "3",
        }
    ) as spark:
        # Create estimator without setting these parameters
        est = UMAP()

        # Parameters should be picked up from Spark confs, except for float32_inputs which is not supported
        assert est._input_kwargs["verbose"] == 5
        assert "float32_inputs" not in est._input_kwargs
        assert est._input_kwargs["num_workers"] == 3

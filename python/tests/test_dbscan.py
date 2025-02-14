#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from typing import Any, Dict, List, Tuple, Type, TypeVar

import numpy as np
import pyspark
import pytest
from _pytest.logging import LogCaptureFixture
from packaging import version

if version.parse(pyspark.__version__) < version.parse("3.4.0"):
    from pyspark.sql.utils import IllegalArgumentException  # type: ignore
else:
    from pyspark.errors import IllegalArgumentException  # type: ignore

from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseVector, Vectors
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

from spark_rapids_ml.clustering import DBSCAN, DBSCANModel

from .sparksession import CleanSparkSession
from .utils import (
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)


@pytest.mark.parametrize("default_params", [True, False])
def test_params(
    default_params: bool,
    tmp_path: str,
) -> None:
    from cuml import DBSCAN as cumlDBSCAN

    spark_params = {
        param.name: value for param, value in DBSCAN().extractParamMap().items()
    }

    cuml_params = get_default_cuml_parameters(
        cuml_classes=[cumlDBSCAN],
        excludes=[
            "handle",
            "output_type",
            "calc_core_sample_indices",
        ],
    )

    # Ensure internal cuml defaults match actual cuml defaults
    assert DBSCAN()._get_cuml_params_default() == cuml_params

    with pytest.raises(
        ValueError, match="Unsupported param 'calc_core_sample_indices'"
    ):
        dbscan_dummy = DBSCAN(calc_core_sample_indices=True)

    if default_params:
        dbscan = DBSCAN()
    else:
        nondefault_params = {
            "eps": 0.4,
            "metric": "cosine",
            "min_samples": 4,
        }
        dbscan = DBSCAN(**nondefault_params)  # type: ignore
        cuml_params.update(nondefault_params)
        spark_params.update(nondefault_params)

    cuml_params["calc_core_sample_indices"] = (
        False  # we override this param to False internally
    )

    # Ensure both Spark API params and internal cuml_params are set correctly
    assert_params(dbscan, spark_params, cuml_params)
    assert dbscan.cuml_params == cuml_params

    # Estimator persistence
    path = tmp_path + "/dbscan_tests"
    estimator_path = f"{path}/dbscan"
    dbscan.write().overwrite().save(estimator_path)
    loaded_dbscan = DBSCAN.load(estimator_path)
    assert_params(loaded_dbscan, spark_params, cuml_params)
    assert loaded_dbscan.cuml_params == cuml_params

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(DBSCAN)


def test_dbscan_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list: List[Dict[str, Any]] = [
        {"eps": 0.7},
        {"min_samples": 10},
        {"metric": "cosine"},
        {"algorithm": "rbc"},
        {"max_mbytes_per_batch": 1000},
        {"verbose": True},
    ]
    for param in param_list:
        _test_est_copy(DBSCAN, param, param)


def test_dbscan_basic(
    gpu_number: int, tmp_path: str, caplog: LogCaptureFixture
) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [
        ([0.0, 0.0]),
        ([1.0, 1.0]),
        ([9.0, 8.0]),
        ([8.0, 9.0]),
    ]

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data, gpu_number)
            .map(lambda row: (row,))
            .toDF(["features"])
        )
        dbscan = DBSCAN(num_workers=gpu_number, min_samples=2, eps=2).setFeaturesCol(
            "features"
        )

        dbscan_model = dbscan.fit(df)

        # Model persistence
        path = tmp_path + "/dbscan_tests"
        model_path = f"{path}/dbscan_model"
        dbscan_model.write().overwrite().save(model_path)
        dbscan_model_loaded = DBSCANModel.load(model_path)

        # test transform function
        dbscan_model.setPredictionCol("prediction")
        label_df = dbscan_model.transform(df)
        assert ["features", "prediction"] == sorted(label_df.columns)

        o_col = dbscan_model.getPredictionCol()
        labels = [row[o_col] for row in label_df.collect()]

        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[1] != labels[2]
        assert labels[2] == labels[3]

        # Test the loaded model
        dbscan_model_loaded.setPredictionCol("prediction")
        label_df = dbscan_model_loaded.transform(df)
        assert ["features", "prediction"] == sorted(label_df.columns)

        o_col = dbscan_model_loaded.getPredictionCol()
        labels = [row[o_col] for row in label_df.collect()]

        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[1] != labels[2]
        assert labels[2] == labels[3]


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_dbscan_numeric_type(gpu_number: int, data_type: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
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
        dbscan = DBSCAN(num_workers=gpu_number, featuresCols=feature_cols)
        dbscan_model = dbscan.fit(df)
        label_df = dbscan_model.transform(df)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize(
    "data_shape",
    [(1000, 20), pytest.param((10000, 200), marks=pytest.mark.slow)],
    ids=idfn,
)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize(
    "max_record_batch", [pytest.param(100, marks=pytest.mark.slow), 10000]
)
@pytest.mark.parametrize("algorithm", ["brute", "rbc"])
def test_dbscan(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    algorithm: str,
) -> None:
    from cuml.datasets import make_blobs

    n_rows = data_shape[0]
    n_cols = data_shape[1]
    n_clusters = 8
    cluster_std = 1.0

    eps = 5
    min_samples = 5
    metric = "euclidean"

    X, _ = make_blobs(
        n_rows, n_cols, n_clusters, cluster_std=cluster_std, random_state=0
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    from cuml import DBSCAN as cuDBSCAN

    cuml_dbscan = cuDBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        output_type="numpy",
        verbose=6,
    )

    import cudf

    gdf = cudf.DataFrame(X)
    cuml_transformed = cuml_dbscan.fit_predict(gdf)

    sample_to_cluster = dict()
    cluster_dict: Dict[int, int] = dict()

    np_df = X.get()
    for rid, row in enumerate(np_df):
        label = cuml_transformed[rid]

        sample_to_cluster[tuple(row)] = label

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )

        dbscan = DBSCAN(
            num_workers=gpu_number,
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            verbose=6,
        ).setFeaturesCol(features_col)

        dbscan_model = dbscan.fit(df)
        dbscan_model.setPredictionCol("prediction")
        transformed = dbscan_model.transform(df)

        # Check cluster match
        label_df = transformed.select("prediction")
        feature_df = transformed.drop("prediction")

        label_pdf = label_df.toPandas()
        feature_pdf = feature_df.toPandas()

        label_arr = label_pdf.to_numpy().squeeze()
        feature_matrix = feature_pdf.to_numpy()

        for rid, row in enumerate(feature_matrix):
            if isinstance(row[0], DenseVector):
                data = tuple(row[0].toArray())
            elif isinstance(row[0], np.float32) or isinstance(row[0], np.float64):
                data = tuple(row)
            else:
                data = tuple(row[0])

            # Get the label computed by rapids and cuml
            label_rapids = label_arr[rid]
            label_cuml = sample_to_cluster[data]

            # Check if the mapping from rapids cluster to cuml cluster holds
            if label_rapids in cluster_dict:
                assert cluster_dict[label_rapids] == label_cuml
            else:
                cluster_dict[label_rapids] = label_cuml

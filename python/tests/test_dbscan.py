#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

# from pyspark.ml.clustering import KMeans as SparkKMeans
# from pyspark.ml.clustering import KMeansModel as SparkKMeansModel
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseVector, Vectors
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

from spark_rapids_ml.dbscan import DBSCAN, DBSCANModel

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


def test_default_cuml_params() -> None:
    from cuml import DBSCAN as CumlDBSCAN

    cuml_params = get_default_cuml_parameters([CumlDBSCAN], ["handle", "output_type"])
    spark_params = DBSCAN()._get_cuml_params_default()
    assert cuml_params == spark_params


def test_dbscan_params(
    gpu_number: int, tmp_path: str, caplog: LogCaptureFixture
) -> None:
    # Default constructor
    default_spark_params: Dict[str, Any] = {}
    default_cuml_params = {
        "eps": 0.5,
        "min_samples": 5,
        "metric": "euclidean",
        "verbose": False,
        "max_mbytes_per_batch": None,
        "calc_core_sample_indices": True,
    }
    default_dbscan = DBSCAN()
    assert_params(default_dbscan, default_spark_params, default_cuml_params)

    # Estimator persistence
    path = tmp_path + "/dbscan_tests"
    estimator_path = f"{path}/dbscan"
    default_dbscan.write().overwrite().save(estimator_path)
    loaded_dbscan = DBSCAN.load(estimator_path)
    assert_params(loaded_dbscan, default_spark_params, default_cuml_params)


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
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.slow
def test_dbscan_self(
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

    print("GPU_NUMBER", gpu_number)
    n_rows = data_shape[0]
    n_cols = data_shape[1]
    n_clusters = 8
    cluster_std = 1.0
    tolerance = 0.001

    eps = 5
    min_samples = 5
    metric = "euclidean"

    X, _ = make_blobs(
        n_rows, n_cols, n_clusters, cluster_std=cluster_std, random_state=0
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    from cuml import DBSCAN as cuDBSCAN

    cuml_dbscan = cuDBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, output_type="numpy", verbose=7
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
            verbose=7,
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

            label_rapids = label_arr[rid]
            label_cuml = sample_to_cluster[data]

            if label_rapids in cluster_dict:
                assert cluster_dict[label_rapids] == label_cuml
            else:
                cluster_dict[label_rapids] = label_cuml

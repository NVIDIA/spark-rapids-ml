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

from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spark_rapids_ml.feature import PCA, PCAModel

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    idfn,
    pyspark_supported_feature_types,
)


def test_fit(gpu_number: int) -> None:
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    topk = 1

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["features"])
        )
        gpu_pca = (
            PCA(num_workers=gpu_number, verbose=6).setInputCol("features").setK(topk)
        )
        gpu_model = gpu_pca.fit(df)

        assert gpu_model.getInputCol() == "features"

        assert len(gpu_model.mean) == 2
        assert gpu_model.mean[0] == pytest.approx(2.0, 0.001)
        assert gpu_model.mean[1] == pytest.approx(2.0, 0.001)

        assert len(gpu_model.pc) == 1
        assert len(gpu_model.pc[0]) == 2
        assert gpu_model.pc[0][0] == pytest.approx(0.707, 0.001)
        assert gpu_model.pc[0][1] == pytest.approx(0.707, 0.001)

        assert len(gpu_model.explained_variance) == 1
        assert gpu_model.explained_variance[0] == pytest.approx(2.0, 0.001)


def test_fit_rectangle(gpu_number: int) -> None:
    data = [[1.0, 1.0], [1.0, 3.0], [5.0, 1.0], [5.0, 3.0]]

    topk = 2

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["coordinates"])
        )

        gpu_pca = PCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)
        gpu_model = gpu_pca.fit(df)

        assert gpu_model.getInputCol() == "coordinates"

        assert len(gpu_model.mean) == 2
        assert gpu_model.mean[0] == pytest.approx(3.0, 0.001)
        assert gpu_model.mean[1] == pytest.approx(2.0, 0.001)

        assert len(gpu_model.pc) == 2

        first_pc = gpu_model.pc[0]
        assert len(first_pc) == 2
        assert first_pc[0] == pytest.approx(1.0, 0.001)
        assert first_pc[1] == pytest.approx(0.0, 0.001)

        second_pc = gpu_model.pc[1]
        assert len(second_pc) == 2
        assert second_pc[0] == pytest.approx(0.0, 0.001)
        assert second_pc[1] == pytest.approx(1.0, 0.001)

        assert len(gpu_model.explained_variance) == 2
        assert gpu_model.explained_variance[0] == pytest.approx(16.0 / 3, 0.001)
        assert gpu_model.explained_variance[1] == pytest.approx(4.0 / 3, 0.001)


def test_pca_basic(gpu_number: int, tmp_path: str) -> None:
    """
    Sparkcuml keeps the algorithmic parameters and their default values
    exactly the same as cuml.dask.decomposition.PCA,
    which follows scikit-learn convention.
    Please refer to https://docs.rapids.ai/api/cuml/stable/api.html#id45
    """

    default_pca = PCA()
    assert default_pca.getOrDefault("n_components") == 1
    assert default_pca.getOrDefault("svd_solver") == "auto"
    assert not default_pca.getOrDefault("verbose")
    assert not default_pca.getOrDefault("whiten")
    assert default_pca.getOrDefault("num_workers") == 1
    assert default_pca.get_num_workers() == 1

    n_components = 5
    svd_solver = "jacobi"
    verbose = True
    whiten = True
    num_workers = 5
    custom_pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        verbose=verbose,
        whiten=whiten,
        num_workers=num_workers,
    )

    def assert_pca_parameters(pca: PCA) -> None:
        assert pca.getOrDefault("n_components") == n_components
        assert pca.getOrDefault("svd_solver") == svd_solver
        assert pca.getOrDefault("verbose") == verbose
        assert pca.getOrDefault("whiten") == whiten
        assert pca.getOrDefault("num_workers") == num_workers
        assert pca.get_num_workers() == num_workers

    assert_pca_parameters(custom_pca)

    path = tmp_path + "/pca_tests"
    estimator_path = f"{path}/pca"
    custom_pca.write().overwrite().save(estimator_path)
    custom_pca_loaded = PCA.load(estimator_path)

    assert_pca_parameters(custom_pca_loaded)

    # Train a PCA model
    data = [[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [5.0, 1.0, 3.9], [5.0, 3.0, 2.9]]
    topk = 2

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["coordinates"])
        )

        gpu_pca = PCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)
        pca_model: PCAModel = gpu_pca.fit(df)

        model_path = f"{path}/pca_model"
        pca_model.write().overwrite().save(model_path)
        pca_model_loaded = PCAModel.load(model_path)

        def assert_pca_model(model: PCAModel, loaded_model: PCAModel) -> None:
            """
            Expect the model attributes are same
            """
            assert model.mean == loaded_model.mean
            assert model.singular_values == loaded_model.singular_values
            assert model.explained_variance == loaded_model.explained_variance
            assert model.pc == loaded_model.pc
            assert model.getOrDefault("n_components") == loaded_model.getOrDefault(
                "n_components"
            )
            assert model.dtype == loaded_model.dtype
            assert model.n_cols == model.n_cols
            assert model.n_cols == 3
            assert model.dtype == "float64"

        assert_pca_model(pca_model, pca_model_loaded)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
def test_pca(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
) -> None:

    X, _ = make_blobs(n_samples=data_shape[0], n_features=data_shape[1], random_state=0)

    from cuml import PCA as cuPCA

    n_components = 3

    cu_pca = cuPCA(n_components=n_components, output_type="numpy", verbose=7)
    cu_result = cu_pca.fit_transform(X)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )

        output_col = (
            "pca_features"
            if isinstance(features_col, str)
            else ["pca_feature_" + str(i) for i in range(n_components)]
        )

        spark_pca = (
            PCA(n_components=3).setInputCol(features_col).setOutputCol(output_col)
        )
        model = spark_pca.fit(train_df)
        assert array_equal(cu_pca.components_, model.pc, 1e-3, with_sign=False)
        assert array_equal(cu_pca.explained_variance_, model.explained_variance, 1e-3)
        assert array_equal(cu_pca.mean_, model.mean, 1e-3)
        assert array_equal(cu_pca.singular_values_, model.singular_values, 1e-3)
        transform_df = model.transform(train_df)

        if isinstance(output_col, str):
            spark_result = transform_df.collect()
            spark_result = [v[0] for v in spark_result]
        else:
            spark_result = transform_df.toPandas().to_numpy()
        assert array_equal(cu_result, spark_result, 1e-3, with_sign=False)


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_pca_numeric_type(gpu_number: int, data_type: str) -> None:
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
        pca = PCA(num_workers=gpu_number, inputCols=feature_cols)
        pca.fit(df)

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
from typing import List

import pytest
from pyspark.sql import SparkSession

from sparkcuml.decomposition import SparkCumlPCA, SparkCumlPCAModel


def test_fit(spark: SparkSession, gpu_number: int) -> None:
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    topk = 1

    rdd = spark.sparkContext.parallelize(data).map(lambda row: (row,))

    df = rdd.toDF(["features"])

    gpu_pca = SparkCumlPCA(num_workers=gpu_number).setInputCol("features").setK(topk)

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


def test_fit_rectangle(spark: SparkSession, gpu_number: int) -> None:
    data = [[1.0, 1.0], [1.0, 3.0], [5.0, 1.0], [5.0, 3.0]]

    topk = 2

    rdd = spark.sparkContext.parallelize(data).map(lambda row: (row,))

    df = rdd.toDF(["coordinates"])

    gpu_pca = SparkCumlPCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)

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


def test_pca_basic(spark: SparkSession, gpu_number: int, tmp_path: str) -> None:
    """
    Sparkcuml keeps the algorithmic parameters and their default values
    exactly the same as cuml.dask.decomposition.PCA,
    which follows scikit-learn convention.
    Please refer to https://docs.rapids.ai/api/cuml/stable/api.html#id45
    """

    default_pca = SparkCumlPCA()
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
    custom_pca = SparkCumlPCA(
        n_components=n_components,
        svd_solver=svd_solver,
        verbose=verbose,
        whiten=whiten,
        num_workers=num_workers,
    )

    def assert_pca_parameters(pca: SparkCumlPCA) -> None:
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
    custom_pca_loaded = SparkCumlPCA.load(estimator_path)

    assert_pca_parameters(custom_pca_loaded)

    # Train a PCA model
    data = [[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [5.0, 1.0, 3.9], [5.0, 3.0, 2.9]]
    topk = 2
    rdd = spark.sparkContext.parallelize(data).map(lambda row: (row,))
    df = rdd.toDF(["coordinates"])
    gpu_pca = SparkCumlPCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)
    pca_model: SparkCumlPCAModel = gpu_pca.fit(df)

    model_path = f"{path}/pca_model"
    pca_model.write().overwrite().save(model_path)
    pca_model_loaded = SparkCumlPCAModel.load(model_path)

    def assert_pca_model(
        model: SparkCumlPCAModel, loaded_model: SparkCumlPCAModel
    ) -> None:
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

    assert_pca_model(pca_model, pca_model_loaded)


def test_fit_compare_cuml(spark: SparkSession, gpu_number: int) -> None:
    import numpy

    data = numpy.random.rand(100, 30).astype(numpy.float64).tolist()
    topk = 5
    tolerance_float = 0.001

    from cuml import PCA

    cuml_pca = PCA(n_components=topk, output_type="numpy")

    import cudf

    gdf = cudf.DataFrame(data)
    cuml_pca.fit(gdf)

    rdd = spark.sparkContext.parallelize(data).map(lambda row: (row,))
    df = rdd.toDF(["features"])
    sparkcuml_pca = SparkCumlPCA(num_workers=gpu_number, n_components=topk).setInputCol(
        "features"
    )
    sparkcuml_model = sparkcuml_pca.fit(df)

    assert sparkcuml_pca.getOrDefault("num_workers") == gpu_number

    assert sparkcuml_model.mean == pytest.approx(
        cuml_pca.mean_.tolist(), tolerance_float
    )
    cuml_pc = cuml_pca.components_.tolist()
    assert len(sparkcuml_model.pc) == len(cuml_pc)
    for i in range(len(cuml_pc)):
        assert_pc_equal(sparkcuml_model.pc[i], cuml_pc[i], tolerance_float)
    assert sparkcuml_model.explained_variance == pytest.approx(
        cuml_pca.explained_variance_.tolist(), tolerance_float
    )


def assert_pc_equal(pc1: List[float], pc2: List[float], tolerance: float) -> None:
    pc2_opposite_dir = [-v for v in pc2]
    assert pc1 == pytest.approx(pc2, tolerance) or pc1 == pytest.approx(
        pc2_opposite_dir, tolerance
    )


def test_transform(spark: SparkSession) -> None:
    mean = [2.0, 2.0]
    pc = [[0.707, 0.707]]
    explained_variance = [2.0]
    singular_values = [2.0]
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    df = spark.sparkContext.parallelize(data).map(lambda row: (row,)).toDF(["features"])
    model = (
        SparkCumlPCAModel(mean, pc, explained_variance, singular_values)
        .setInputCol("features")
        .setOutputCol("pca_features")
    )

    projs = model.transform(df).collect()
    assert len(projs) == 3
    d1 = projs[0].asDict()
    d2 = projs[1].asDict()
    d3 = projs[2].asDict()
    assert "pca_features" in d1
    assert "pca_features" in d2
    assert "pca_features" in d3
    assert d1["pca_features"] == pytest.approx([-1.414], 0.001)
    assert d2["pca_features"] == pytest.approx([0], 0.001)
    assert d3["pca_features"] == pytest.approx([1.414], 0.001)

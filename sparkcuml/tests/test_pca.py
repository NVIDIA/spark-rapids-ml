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

import pytest
from pyspark.sql import SparkSession

from sparkcuml.decomposition import SparkCumlPCA


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


def test_pca_parameters(spark: SparkSession, gpu_number: int) -> None:
    """
    Sparkcuml keeps the algorithmic parameters and their default values
    exactly the same as cuml.dask.decomposition.PCA,
    which follows scikit-learn convention.
    Please refer to https://docs.rapids.ai/api/cuml/stable/api.html#id45
    """

    default_pca = SparkCumlPCA()
    assert default_pca.getOrDefault("n_components") == 1
    assert default_pca.getOrDefault("svd_solver") == "auto"
    assert default_pca.getOrDefault("verbose") == False
    assert default_pca.getOrDefault("whiten") == False

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

    assert custom_pca.getOrDefault("n_components") == n_components
    assert custom_pca.getOrDefault("svd_solver") == svd_solver
    assert custom_pca.getOrDefault("verbose") == verbose
    assert custom_pca.getOrDefault("whiten") == whiten

    assert custom_pca.getOrDefault("num_workers") == num_workers
    assert custom_pca.get_num_workers() == num_workers


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


def assert_pc_equal(pc1: list[float], pc2: list[float], tolerance: float) -> None:
    pc2_opposite_dir = [-v for v in pc2]
    assert pc1 == pytest.approx(pc2, tolerance) or pc1 == pytest.approx(
        pc2_opposite_dir, tolerance
    )

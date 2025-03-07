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

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pyspark
import pytest
from _pytest.logging import LogCaptureFixture
from packaging import version

if version.parse(pyspark.__version__) < version.parse("3.4.0"):
    from pyspark.sql.utils import IllegalArgumentException  # type: ignore
else:
    from pyspark.errors import IllegalArgumentException  # type: ignore

from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.clustering import KMeansModel as SparkKMeansModel
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

from spark_rapids_ml.clustering import KMeans, KMeansModel

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

KMeansType = TypeVar("KMeansType", Type[KMeans], Type[SparkKMeans])
KMeansModelType = TypeVar("KMeansModelType", Type[KMeansModel], Type[SparkKMeansModel])


def assert_centers_equal(
    a_clusters: List[List[float]], b_clusters: List[List[float]], tolerance: float
) -> None:
    assert len(a_clusters) == len(b_clusters)
    a_clusters = sorted(a_clusters, key=lambda l: l)
    b_clusters = sorted(b_clusters, key=lambda l: l)
    for i in range(len(a_clusters)):
        a_center = a_clusters[i]
        b_center = b_clusters[i]
        assert len(a_center) == len(b_center)
        assert a_center == pytest.approx(b_center, tolerance)


@pytest.mark.parametrize("default_params", [True, False])
def test_params(default_params: bool) -> None:
    from cuml import KMeans as CumlKMeans
    from pyspark.ml.clustering import KMeans as SparkKMeans

    spark_params = {
        param.name: value for param, value in SparkKMeans().extractParamMap().items()
    }

    cuml_params = get_default_cuml_parameters(
        cuml_classes=[CumlKMeans], excludes=["handle", "output_type", "convert_dtype"]
    )

    # Ensure internal cuml defaults match actual cuml defaults
    assert KMeans()._get_cuml_params_default() == cuml_params

    # Our algorithm overrides the following cuml parameters with their spark defaults:
    spark_default_overrides = {
        "n_clusters": spark_params["k"],
        "max_iter": spark_params["maxIter"],
        "init": spark_params["initMode"],
    }

    cuml_params.update(spark_default_overrides)

    if default_params:
        kmeans = KMeans()
        seed = kmeans.getSeed()  # get the random seed that Spark generates
        spark_params["seed"] = seed
        cuml_params["random_state"] = seed
    else:
        kmeans = KMeans(
            k=10,
            seed=42,
        )
        cuml_params["n_clusters"] = 10
        cuml_params["random_state"] = 42
        spark_params["k"] = 10
        spark_params["seed"] = 42

    # Ensure both Spark API params and internal cuml_params are set correctly
    assert_params(kmeans, spark_params, cuml_params)
    assert kmeans.cuml_params == cuml_params

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(KMeans)


def test_kmeans_params(
    gpu_number: int, tmp_path: str, caplog: LogCaptureFixture
) -> None:
    # Default constructor
    default_spark_params = {
        "initMode": "k-means||",
        "k": 2,
        "maxIter": 20,
    }
    default_cuml_params = {
        "n_clusters": 2,
        "max_iter": 20,
        "tol": 0.0001,
        "verbose": False,
        "init": "k-means||",
        "oversampling_factor": 2.0,
        "max_samples_per_batch": 32768,
        "num_workers": 1,
    }
    default_kmeans = KMeans()
    assert_params(default_kmeans, default_spark_params, default_cuml_params)

    # Spark Params constructor
    spark_params: Dict[str, Any] = {"k": 10, "maxIter": 100}
    spark_kmeans = KMeans(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update({"n_clusters": 10, "max_iter": 100})
    assert_params(spark_kmeans, expected_spark_params, expected_cuml_params)

    # cuml_params constructor
    cuml_params: Dict[str, Any] = {
        "n_clusters": 10,
        "max_iter": 100,
        "tol": 1e-1,
        "verbose": True,
        "random_state": 5,
        "init": "k-means||",
        "oversampling_factor": 3,
        "max_samples_per_batch": 45678,
    }
    cuml_kmeans = KMeans(**cuml_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update({"k": 10, "maxIter": 100})
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(cuml_params)
    assert_params(cuml_kmeans, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/kmeans_tests"
    estimator_path = f"{path}/kmeans"
    cuml_kmeans.write().overwrite().save(estimator_path)
    loaded_kmeans = KMeans.load(estimator_path)
    assert_params(loaded_kmeans, expected_spark_params, expected_cuml_params)

    # conflicting params
    conflicting_params: Dict[str, Any] = {
        "k": 2,
        "n_clusters": 10,
    }
    with pytest.raises(ValueError, match="set one or the other"):
        conflicting_kmeans = KMeans(**conflicting_params)

    # make sure no warning when enabling float64 inputs
    kmeans_float32 = KMeans(float32_inputs=False)
    assert "float32_inputs to False" not in caplog.text
    assert not kmeans_float32._float32_inputs


def test_kmeans_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [
        ({"k": 17}, {"n_clusters": 17}),
        ({"initMode": "random"}, {"init": "random"}),
        ({"tol": 0.0132}, {"tol": 0.0132}),
        ({"maxIter": 27}, {"max_iter": 27}),
        ({"seed": 11}, {"random_state": 11}),
        ({"verbose": True}, {"verbose": True}),
    ]

    for pair in param_list:
        _test_est_copy(KMeans, pair[0], pair[1])


def test_kmeans_basic(
    gpu_number: int, tmp_path: str, caplog: LogCaptureFixture
) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]]

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data, gpu_number)
            .map(lambda row: (row,))
            .toDF(["features"])
        )
        kmeans = (
            KMeans(num_workers=gpu_number, n_clusters=2)
            .setFeaturesCol("features")
            .setSeed(0)
        )

        kmeans.setTol(0.0)

        warn_msg = (
            "tol=0 is not supported in cuml yet. "
            + "It will be mapped to smallest positive float, i.e. numpy.finfo('float32').tiny."
        )

        assert warn_msg in caplog.text

        def assert_kmeans_model(model: KMeansModel) -> None:
            assert len(model.cluster_centers_) == 2
            sorted_centers = sorted(model.cluster_centers_, key=lambda p: p)
            assert sorted_centers[0] == pytest.approx([1.0, 1.5], 0.001)
            assert sorted_centers[1] == pytest.approx([3.5, 2.5], 0.001)
            assert model.dtype == "float32"
            assert model.n_cols == 2

        def assert_cuml_spark_model(
            model: KMeansModel, spark_model: SparkKMeansModel
        ) -> None:
            lhs = model.clusterCenters()
            rhs = spark_model.clusterCenters()
            assert len(lhs) == len(rhs)
            for i in range(len(lhs)):
                comp = lhs[i] == rhs[i]
                assert comp.all()

        kmeans_model = kmeans.fit(df)
        assert_kmeans_model(model=kmeans_model)

        assert isinstance(kmeans_model.cpu(), SparkKMeansModel)
        assert_cuml_spark_model(kmeans_model, kmeans_model.cpu())

        # Model persistence
        path = tmp_path + "/kmeans_tests"
        model_path = f"{path}/kmeans_model"
        kmeans_model.write().overwrite().save(model_path)
        kmeans_model_loaded = KMeansModel.load(model_path)
        assert_kmeans_model(model=kmeans_model_loaded)

        assert isinstance(kmeans_model_loaded.cpu(), SparkKMeansModel)
        assert_cuml_spark_model(kmeans_model_loaded, kmeans_model_loaded.cpu())

        # test transform function
        label_df = kmeans_model.transform(df)
        assert ["features", "prediction"] == sorted(label_df.columns)

        o_col = kmeans_model.getPredictionCol()
        labels = [row[o_col] for row in label_df.collect()]

        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[1] != labels[2]
        assert labels[2] == labels[3]

        # without raising exception for cuml model predict
        kmeans_model.predict(Vectors.dense(1.0, 1.0))

        # without raising exception for cpu transform
        kmeans_model.cpu().transform(
            df.select(array_to_vector(col("features")).alias("features"))
        ).collect()


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_kmeans_numeric_type(gpu_number: int, data_type: str) -> None:
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
        kmeans = KMeans(num_workers=gpu_number, featuresCols=feature_cols, n_clusters=2)
        kmeans.fit(df)


@pytest.mark.xfail
@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.slow
def test_kmeans(
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

    n_rows = data_shape[0]
    n_cols = data_shape[1]
    n_clusters = 4
    tol = 1.0e-20
    seed = 42  # This does not guarantee deterministic centers in 25.02.
    cluster_std = 1.0
    tolerance = 0.001

    X, _ = make_blobs(
        n_rows, n_cols, n_clusters, cluster_std=cluster_std, random_state=0
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    from cuml import KMeans as cuKMeans

    cuml_kmeans = cuKMeans(
        n_clusters=n_clusters,
        output_type="numpy",
        tol=tol,
        random_state=seed,
        verbose=6,
    )

    import cudf

    gdf = cudf.DataFrame(X)
    cuml_kmeans.fit(gdf)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )

        kmeans = KMeans(
            num_workers=gpu_number, n_clusters=n_clusters, tol=tol, seed=seed, verbose=6
        ).setFeaturesCol(features_col)

        kmeans_model = kmeans.fit(df)

        cuml_cluster_centers = cuml_kmeans.cluster_centers_.tolist()
        assert_centers_equal(
            kmeans_model.cluster_centers_,
            cuml_cluster_centers,
            tolerance,
        )

        # test transform function

        sid_ordered = sorted(
            range(n_clusters), key=lambda idx: kmeans_model.cluster_centers_[idx]
        )
        cid_ordered = sorted(
            range(n_clusters), key=lambda idx: cuml_cluster_centers[idx]
        )
        s2c = dict(
            zip(sid_ordered, cid_ordered)
        )  # map spark-rapids-ml center id to cuml center id

        labelDf = kmeans_model.transform(df)
        o_col = kmeans_model.getPredictionCol()
        slabels = [row[o_col] for row in labelDf.collect()]

        clabels = cuml_kmeans.predict(gdf).tolist()

        assert len(slabels) == len(clabels)
        to_clabels = [s2c[v] for v in slabels]
        assert to_clabels == clabels


@pytest.mark.compat
@pytest.mark.parametrize(
    "kmeans_types", [(SparkKMeans, SparkKMeansModel), (KMeans, KMeansModel)]
)
def test_kmeans_spark_compat(
    gpu_number: int,
    kmeans_types: Tuple[KMeansType, KMeansModelType],
    tmp_path: str,
) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

    # based on https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.PCA.htm
    _KMeans, _KMeansModel = kmeans_types

    with CleanSparkSession() as spark:
        from pyspark.ml.linalg import Vectors

        data = [
            (Vectors.dense([0.0, 0.0]), 2.0),
            (Vectors.dense([1.0, 1.0]), 2.0),
            (Vectors.dense([9.0, 8.0]), 2.0),
            (Vectors.dense([8.0, 9.0]), 2.0),
        ]
        df = spark.createDataFrame(data, ["features", "weighCol"])

        import pyspark
        from packaging import version

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            kmeans = _KMeans(k=2)
        else:
            kmeans = _KMeans(k=2, solver="auto", maxBlockSizeInMB=0)  # type: ignore # only spark >= 3.4 supports solver and maxblockSize

        kmeans.setSeed(1)
        kmeans.setMaxIter(10)
        kmeans.setInitMode("k-means||")
        if isinstance(kmeans, SparkKMeans):
            kmeans.setWeightCol("weighCol")
        else:
            with pytest.raises(ValueError):
                kmeans.setWeightCol("weighCol")

        assert kmeans.getMaxIter() == 10
        assert kmeans.getK() == 2
        assert kmeans.getSeed() == 1
        assert kmeans.getInitMode() == "k-means||"

        kmeans.clear(kmeans.maxIter)
        assert kmeans.getMaxIter() == 20

        model = kmeans.fit(df)

        assert model.getDistanceMeasure() == "euclidean"

        model.setPredictionCol("newPrediction")
        assert model.getPredictionCol() == "newPrediction"

        example = df.head()
        if example:
            model.predict(example.features)

        centers = model.clusterCenters()
        # [array([0.5, 0.5]), array([8.5, 8.5])]
        assert len(centers) == 2

        sorted_centers = sorted([x.tolist() for x in centers])
        expected_centers = [[0.5, 0.5], [8.5, 8.5]]
        assert sorted_centers == expected_centers

        transformed = model.transform(df).select("features", "newPrediction")
        rows = transformed.collect()
        # [Row(features=DenseVector([0.0, 0.0]), newPrediction=0),
        #  Row(features=DenseVector([1.0, 1.0]), newPrediction=0),
        #  Row(features=DenseVector([9.0, 8.0]), newPrediction=1),
        #  Row(features=DenseVector([8.0, 9.0]), newPrediction=1)]

        assert rows[0].newPrediction == rows[1].newPrediction
        assert rows[2].newPrediction == rows[3].newPrediction

        if isinstance(model, SparkKMeansModel):
            assert model.hasSummary == True
            summary = model.summary
            assert summary.k == 2
            assert summary.clusterSizes == [2, 2]
            assert summary.trainingCost == 4.0
        else:
            assert model.hasSummary == False

        kmeans_path = tmp_path + "/kmeans"
        kmeans.save(kmeans_path)
        kmeans2 = _KMeans.load(kmeans_path)
        assert kmeans2.getK() == 2

        model_path = tmp_path + "/kmeans_model"
        model.save(model_path)
        model2 = _KMeansModel.load(model_path)
        assert model2.hasSummary == False

        assert all(model.clusterCenters()[0] == model2.clusterCenters()[0])
        # array([ True,  True], dtype=bool)

        assert all(model.clusterCenters()[1] == model2.clusterCenters()[1])
        # array([ True,  True], dtype=bool)

        assert model.transform(df).take(1) == model2.transform(df).take(1)
        # True


def test_parameters_validation() -> None:
    data = [
        ([1.0, 2.0], 1.0),
        ([3.0, 1.0], 0.0),
    ]

    with CleanSparkSession() as spark:
        features_col = "features"
        label_col = "label"
        schema = features_col + " array<float>, " + label_col + " float"
        df = spark.createDataFrame(data, schema=schema)
        with pytest.raises(IllegalArgumentException, match="k given invalid value -1"):
            KMeans(k=-1).fit(df)

        with pytest.raises(
            IllegalArgumentException, match="maxIter given invalid value -1"
        ):
            KMeans().setMaxIter(-1).fit(df)

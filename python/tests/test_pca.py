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

from typing import Any, Dict, Tuple, Type, TypeVar

import numpy as np
import pyspark
import pytest
from _pytest.logging import LogCaptureFixture
from packaging import version

if version.parse(pyspark.__version__) < version.parse("3.4.0"):
    from pyspark.sql.utils import IllegalArgumentException  # type: ignore
else:
    from pyspark.errors import IllegalArgumentException  # type: ignore

from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCAModel as SparkPCAModel
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseMatrix, Vectors, VectorUDT
from pyspark.sql.functions import col
from pyspark.sql.types import StructField, StructType
from sklearn.datasets import make_blobs

from spark_rapids_ml.feature import PCA, PCAModel

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)

PCAType = TypeVar("PCAType", Type[SparkPCA], Type[PCA])
PCAModelType = TypeVar("PCAModelType", Type[SparkPCAModel], Type[PCAModel])


@pytest.mark.parametrize("default_params", [True, False])
def test_params(default_params: bool, caplog: LogCaptureFixture) -> None:
    from cuml import PCA as CumlPCA
    from pyspark.ml.feature import PCA as SparkPCA

    spark_params = {
        param.name: value for param, value in SparkPCA().extractParamMap().items()
    }
    # Ignore output col, as it is linked to the object id by default (e.g., 'PCA_ac9c581af6b3__output')
    spark_params.pop("outputCol", None)

    cuml_params = get_default_cuml_parameters(
        cuml_classes=[CumlPCA],
        excludes=[
            "copy",
            "handle",
            "iterated_power",
            "output_type",
            "random_state",
            "tol",
        ],
    )

    # Ensure internal cuml defaults match actual cuml defaults
    assert cuml_params == PCA()._get_cuml_params_default()

    if default_params:
        pca = PCA()
    else:
        pca = PCA(k=4)
        cuml_params["n_components"] = 4
        spark_params["k"] = 4

    # Ensure both Spark API params and internal cuml_params are set correctly
    assert_params(pca, spark_params, cuml_params)
    assert cuml_params == pca.cuml_params

    # make sure no warning when enabling float64 inputs
    pca_float32 = PCA(float32_inputs=False)
    assert "float32_inputs to False" not in caplog.text
    assert not pca_float32._float32_inputs

    # setter/getter
    from .test_common_estimator import _test_input_setter_getter

    _test_input_setter_getter(PCA)


def test_pca_copy() -> None:
    from .test_common_estimator import _test_est_copy

    param_list = [
        ({"k": 42}, {"n_components": 42}),
        ({"verbose": 42},),
    ]

    for pair in param_list:
        spark_param = pair[0]
        cuml_param = spark_param if len(pair) == 1 else pair[1]
        _test_est_copy(PCA, spark_param, cuml_param)


def test_fit(gpu_number: int) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

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

        assert len(gpu_model.mean_) == 2
        assert gpu_model.mean_[0] == pytest.approx(2.0, 0.001)
        assert gpu_model.mean_[1] == pytest.approx(2.0, 0.001)

        assert len(gpu_model.components_) == 1
        assert len(gpu_model.components_[0]) == 2
        assert gpu_model.components_[0][0] == pytest.approx(0.707, 0.001)
        assert gpu_model.components_[0][1] == pytest.approx(0.707, 0.001)

        assert len(gpu_model.explained_variance_ratio_) == 1
        assert gpu_model.explained_variance_ratio_[0] == pytest.approx(1.0, 0.001)


def test_fit_rectangle(gpu_number: int) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

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

        assert len(gpu_model.mean_) == 2
        assert gpu_model.mean_[0] == pytest.approx(3.0, 0.001)
        assert gpu_model.mean_[1] == pytest.approx(2.0, 0.001)

        assert len(gpu_model.components_) == 2

        first_pc = gpu_model.components_[0]
        assert len(first_pc) == 2
        assert first_pc[0] == pytest.approx(1.0, 0.001)
        assert first_pc[1] == pytest.approx(0.0, 0.001)

        second_pc = gpu_model.components_[1]
        assert len(second_pc) == 2
        assert second_pc[0] == pytest.approx(0.0, 0.001)
        assert second_pc[1] == pytest.approx(1.0, 0.001)

        assert len(gpu_model.explained_variance_ratio_) == 2
        assert gpu_model.explained_variance_ratio_[0] == pytest.approx(0.8, 0.001)
        assert gpu_model.explained_variance_ratio_[1] == pytest.approx(0.2, 0.001)


def test_pca_params(gpu_number: int, tmp_path: str, caplog: LogCaptureFixture) -> None:
    # Default constructor
    default_spark_params = {
        "k": None,
    }
    default_cuml_params = {
        "num_workers": 1,
        "n_components": None,
        "svd_solver": "auto",
        "verbose": False,
        "whiten": False,
    }
    default_pca = PCA()
    assert_params(default_pca, default_spark_params, default_cuml_params)

    # Spark Params constructor
    spark_params: Dict[str, Any] = {"k": 2}
    spark_pca = PCA(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)  # type: ignore
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update({"n_components": 2})
    assert_params(spark_pca, expected_spark_params, expected_cuml_params)

    # cuml_params constructor
    cuml_params: Dict[str, Any] = {
        "n_components": 5,
        "num_workers": 5,
        "svd_solver": "jacobi",
        "verbose": True,
        "whiten": True,
    }
    cuml_pca = PCA(**cuml_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update({"k": 5})  # type: ignore
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(cuml_params)  # type: ignore
    assert_params(cuml_pca, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/pca_tests"
    estimator_path = f"{path}/pca"
    cuml_pca.write().overwrite().save(estimator_path)
    custom_pca_loaded = PCA.load(estimator_path)
    assert_params(custom_pca_loaded, expected_spark_params, expected_cuml_params)

    # Conflicting params
    conflicting_params: Dict[str, Any] = {
        "k": 1,
        "n_components": 2,
    }
    with pytest.raises(ValueError, match="set one or the other"):
        conflicting_pca = PCA(**conflicting_params)


def test_pca_basic(gpu_number: int, tmp_path: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

    # Train a PCA model
    data = [[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [5.0, 1.0, 3.9], [5.0, 3.0, 2.9]]
    topk = 2
    path = tmp_path + "/pca_tests"

    with CleanSparkSession() as spark:
        df = (
            spark.sparkContext.parallelize(data)
            .map(lambda row: (row,))
            .toDF(["coordinates"])
        )

        gpu_pca = PCA(num_workers=gpu_number).setInputCol("coordinates").setK(topk)
        pca_model: PCAModel = gpu_pca.fit(df)
        assert isinstance(pca_model.cpu(), SparkPCAModel)

        model_path = f"{path}/pca_model"
        pca_model.write().overwrite().save(model_path)
        pca_model_loaded = PCAModel.load(model_path)
        assert isinstance(pca_model_loaded.cpu(), SparkPCAModel)

        def assert_cuml_model(model: PCAModel, loaded_model: PCAModel) -> None:
            """
            Expect the model attributes are same
            """
            assert model.mean_ == loaded_model.mean_
            assert model.singular_values_ == loaded_model.singular_values_
            assert (
                model.explained_variance_ratio_
                == loaded_model.explained_variance_ratio_
            )
            assert model.components_ == loaded_model.components_
            assert (
                model.cuml_params["n_components"]
                == loaded_model.cuml_params["n_components"]
            )
            assert model.dtype == loaded_model.dtype
            assert model.n_cols == model.n_cols
            assert model.n_cols == 3
            assert model.dtype == "float32"

        assert_cuml_model(pca_model, pca_model_loaded)

        def assert_cuml_spark_model(
            model: PCAModel, spark_model: SparkPCAModel
        ) -> None:
            """
            Expect the model attributes are same
            """
            assert model.pc == spark_model.pc
            assert model.explainedVariance == spark_model.explainedVariance
            assert model.getK() == spark_model.getK()
            assert model.getInputCol() == spark_model.getInputCol()
            assert model.getInputCol() == "coordinates"
            assert model.getOutputCol() == spark_model.getOutputCol()

        assert_cuml_spark_model(pca_model, pca_model.cpu())
        assert_cuml_spark_model(pca_model, pca_model_loaded.cpu())

        # cpu model transform without raising any exception
        pca_model.cpu().transform(
            df.select(array_to_vector(col("coordinates")).alias("coordinates"))
        ).collect()


@pytest.mark.parametrize("data_type", ["byte", "short", "int", "long"])
def test_pca_numeric_type(gpu_number: int, data_type: str) -> None:
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
        pca = PCA(num_workers=gpu_number, inputCols=feature_cols)
        pca.fit(df)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.slow
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

    cu_pca = cuPCA(n_components=n_components, output_type="numpy", verbose=6)
    cu_model = cu_pca.fit(X)

    # Spark does not remove the mean from the transformed data
    # adding mean to input data to sanity-check the transformed mean approach in main class
    cu_result = cu_model.transform(X + np.array(cu_model.mean_, data_type))

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type, data_type, X, None
        )
        output_col = "pca_features"

        spark_pca = (
            PCA(n_components=3).setInputCol(features_col).setOutputCol(output_col)
        )

        model = spark_pca.fit(train_df)
        assert model.getK() == model.cpu().getK()
        assert model.getK() == 3
        assert model.getOutputCol() == model.cpu().getOutputCol()
        assert model.getOutputCol() == "pca_features"

        assert array_equal(cu_pca.components_, model.components_, 1e-3, with_sign=False)
        assert array_equal(
            cu_pca.explained_variance_ratio_, model.explained_variance_ratio_, 1e-3
        )
        assert array_equal(cu_pca.mean_, model.mean_, 1e-3)
        assert array_equal(cu_pca.singular_values_, model.singular_values_, 1e-3)
        transform_df = model.transform(train_df)

        spark_result = transform_df.collect()
        pred_result = [v.pca_features for v in spark_result]

        assert array_equal(cu_result, pred_result, 1e-2, with_sign=False)


@pytest.mark.compat
@pytest.mark.parametrize("pca_types", [(SparkPCA, SparkPCAModel), (PCA, PCAModel)])
def test_pca_spark_compat(
    pca_types: Tuple[PCAType, PCAModelType],
    tmp_path: str,
) -> None:
    # based on https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.PCA.htm
    _PCA, _PCAModel = pca_types

    with CleanSparkSession() as spark:
        data = [
            (Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),),
            (Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),),
        ]
        df = spark.createDataFrame(data, ["features"])

        pca = _PCA(k=2, inputCol="features")
        pca.setOutputCol("pca_features")
        assert pca.getInputCol() == "features"
        assert pca.getK() == 2
        assert pca.getOutputCol() == "pca_features"

        model = pca.fit(df)
        model.setOutputCol("output")
        assert model.getOutputCol() == "output"

        k = model.getK()
        output_df = model.transform(df)
        schema_fields = [
            StructField("features", VectorUDT(), True),
        ]
        if _PCA is SparkPCA:
            schema_fields.append(
                StructField(
                    "output", VectorUDT(), True, metadata={"ml_attr": {"num_attrs": k}}
                )
            )
        else:
            schema_fields.append(StructField("output", VectorUDT(), True))

        schema_gnd = StructType(schema_fields)  # type
        assert output_df.schema == schema_gnd

        output = output_df.collect()[0].output
        expected_output = [1.6485728230883814, -4.0132827005162985]
        assert array_equal(output, expected_output, with_sign=False)

        variance = model.explainedVariance.toArray()
        expected_variance = [0.7943932532230531, 0.20560674677694699]
        assert array_equal(variance, expected_variance)

        pc = model.pc
        expected_pc = DenseMatrix(
            5,
            2,
            [
                -0.4486,
                0.133,
                -0.1252,
                0.2165,
                -0.8477,
                -0.2842,
                -0.0562,
                0.7636,
                -0.5653,
                -0.1156,
            ],
            False,
        )
        assert array_equal(pc.toArray(), expected_pc.toArray(), with_sign=False)

        pcaPath = tmp_path + "/pca"
        pca.save(pcaPath)
        loadedPca = _PCA.load(pcaPath)
        assert loadedPca.getK() == pca.getK()

        modelPath = tmp_path + "/pca-model"
        model.save(modelPath)
        loadedModel = _PCAModel.load(modelPath)
        assert loadedModel.pc == model.pc
        assert loadedModel.explainedVariance == model.explainedVariance
        assert loadedModel.transform(df).take(1) == model.transform(df).take(1)


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
            PCA(k=-1).fit(df)

        with pytest.raises(IllegalArgumentException, match="k given invalid value -1"):
            PCA().setK(-1).fit(df)

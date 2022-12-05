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
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pyspark
import pytest
from cuml import LinearRegression as cuLinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import array
from sklearn.datasets import make_regression

from sparkcuml.linear_model.linear_regression import SparkLinearRegression
from sparkcuml.tests.sparksession import CleanSparkSession


def _make_regression_dataset_uncached(
    nrows: int, ncols: int, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    return make_regression(**kwargs, n_samples=nrows, n_features=ncols, random_state=0)


@lru_cache(4)
def _make_regression_dataset_from_cache(
    nrows: int, ncols: int, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    return _make_regression_dataset_uncached(nrows, ncols, **kwargs)


def make_regression_dataset(
    datatype: np.dtype, nrows: int, ncols: int, **kwargs: Any
) -> Any:
    if nrows * ncols < 1e8:  # Keep cache under 4 GB
        dataset = _make_regression_dataset_from_cache(nrows, ncols, **kwargs)
    else:
        dataset = _make_regression_dataset_uncached(nrows, ncols, **kwargs)

    return map(lambda arr: arr.astype(datatype), dataset)


def test_linear_regression_basic(gpu_number: int, tmp_path: str) -> None:
    lr = SparkLinearRegression()
    assert lr.getOrDefault("algorithm") == "eig"
    assert lr.getOrDefault("fit_intercept")
    assert not lr.getOrDefault("normalize")

    def assert_params(linear_reg: SparkLinearRegression) -> None:
        assert linear_reg.getOrDefault("algorithm") == "svd"
        assert not linear_reg.getOrDefault("fit_intercept")
        assert linear_reg.getOrDefault("normalize")

    lr = SparkLinearRegression(algorithm="svd", fit_intercept=False, normalize=True)
    assert_params(lr)

    # Estimator persistence
    path = tmp_path + "/linear_regression_tests"
    estimator_path = f"{path}/linear_regression"
    lr.write().overwrite().save(estimator_path)
    lr_loaded = SparkLinearRegression.load(estimator_path)

    assert_params(lr_loaded)


def dtype_to_pyspark_type(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    elif dtype == np.int32:
        return "int"
    elif dtype == np.int16:
        return "short"
    else:
        raise RuntimeError("Unsupported type")


pyspark_feature_types = ["array", "multi_cols"]


def create_pyspark_dataframe(
    spark: SparkSession,
    feature_type: str,
    dtype: np.dtype,
    features: np.ndarray,
    label: Optional[np.ndarray] = None,
) -> Tuple[pyspark.sql.DataFrame, Union[str, List[str]], Optional[str]]:
    """Construct a dataframe based on features and label data."""
    assert feature_type in pyspark_feature_types

    m, n = features.shape

    pyspark_type = dtype_to_pyspark_type(dtype)
    feature_cols: Union[str, List[str]] = [f"c{i}" for i in range(n)]
    schema = [f"{c} {pyspark_type}" for c in feature_cols]
    label_col = None

    if label is not None:
        label_col = "label_col"
        schema.append(f"{label_col} {pyspark_type}")
        df = spark.createDataFrame(
            np.concatenate((features, label.reshape(m, 1)), axis=1).tolist(),
            ",".join(schema),
        )
    else:
        df = spark.createDataFrame(features.tolist(), ",".join(schema))

    if feature_type == "array":
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)
        feature_cols = "features"

    return df, feature_cols, label_col


# @lru_cache(4) TODO fixme: TypeError: Unhashable Type” Numpy.Ndarray
def train_with_cuml_linear_regression(
    X: np.ndarray, y: np.ndarray
) -> cuLinearRegression:
    lr = cuLinearRegression(output_type="numpy")
    lr.fit(X, y)
    return lr


def array_equal(
    lhs: Union[np.ndarray, List[float]],
    rhs: Union[np.ndarray, List[float]],
    unit_tol: float = 1e-4,
    total_tol: float = 1e-4,
    with_sign: bool = True,
) -> bool:
    a = np.asarray(lhs)
    b = np.asarray(rhs)
    if len(a) == 0 and len(b) == 0:
        return True

    if not with_sign:
        a, b = np.abs(a), np.abs(b)
    res = (np.sum(np.abs(a - b) > unit_tol)) / a.size < total_tol
    return res


def idfn(val: Any) -> str:
    """Provide an API to provide display names for data type generators."""
    return str(val)


@pytest.mark.parametrize("feature_type", pyspark_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
@pytest.mark.parametrize("max_record_batch", [100, 10000])
def test_linear_regression(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
) -> None:
    m, n = data_shape
    X, y = make_regression_dataset(data_type, m, n)

    cu_lr = train_with_cuml_linear_regression(X, y)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )
        assert label_col is not None
        slr = SparkLinearRegression(num_workers=gpu_number)
        slr.setFeaturesCol(features_col)
        slr.setLabelCol(label_col)
        slr_model = slr.fit(df)

        assert array_equal(cu_lr.coef_, slr_model.coef, 1e-3, 1e-3)

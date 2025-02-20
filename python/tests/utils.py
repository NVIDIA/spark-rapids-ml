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

from collections import namedtuple
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import pyspark
from pyspark.ml import Estimator, Model
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import array
from pyspark.sql.types import Row
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from spark_rapids_ml.params import _CumlParams
from spark_rapids_ml.utils import _get_default_params_from_func, dtype_to_pyspark_type

FeatureTypes = namedtuple("FeatureTypes", ("vector", "array", "multi_cols"))
feature_types = FeatureTypes("vector", "array", "multi_cols")

pyspark_supported_feature_types = feature_types._fields
cuml_supported_data_types = [np.float32, np.float64]

CumlParams = TypeVar("CumlParams", bound=_CumlParams)


def idfn(val: Any) -> str:
    """Provide an API to provide display names for data type generators."""
    return str(val)


def _make_regression_dataset_uncached(
    nrows: int, ncols: int, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create regression dataset.

    return X_train, X_test, y_train, y_test
    """
    X, y = make_regression(**kwargs, n_samples=nrows, n_features=ncols, random_state=0)
    return train_test_split(X, y, train_size=0.8, random_state=10)


@lru_cache(4)
def _make_regression_dataset_from_cache(
    nrows: int, ncols: int, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache the dataset"""
    return _make_regression_dataset_uncached(nrows, ncols, **kwargs)


def make_regression_dataset(
    datatype: np.dtype, nrows: int, ncols: int, **kwargs: Any
) -> Iterator[np.ndarray]:
    """Create regression dataset"""
    if nrows * ncols < 1e8:  # Keep cache under 4 GB
        dataset = _make_regression_dataset_from_cache(nrows, ncols, **kwargs)
    else:
        dataset = _make_regression_dataset_uncached(nrows, ncols, **kwargs)

    return map(lambda arr: arr.astype(datatype), dataset)


def create_pyspark_dataframe(
    spark: SparkSession,
    feature_type: str,
    dtype: np.dtype,
    data: np.ndarray,
    label: Optional[np.ndarray] = None,
    label_dtype: Optional[np.dtype] = None,  # type: ignore
) -> Tuple[pyspark.sql.DataFrame, Union[str, List[str]], Optional[str]]:
    """Construct a dataframe based on features and label data."""
    assert feature_type in pyspark_supported_feature_types

    # in case cp.ndarray get passed in
    if not isinstance(data, np.ndarray):
        data = data.get()
    if label is not None and not isinstance(label, np.ndarray):
        label = label.get()

    m, n = data.shape

    pyspark_type = dtype_to_pyspark_type(dtype)
    feature_cols: Union[str, List[str]] = [f"c{i}" for i in range(n)]
    schema = [f"{c} {pyspark_type}" for c in feature_cols]
    label_col = None

    if label is not None:
        label_dtype = dtype if label_dtype is None else label_dtype
        label = label.astype(label_dtype)
        label_pyspark_type = dtype_to_pyspark_type(label_dtype)

        label_col = "label_col"
        schema.append(f"{label_col} {label_pyspark_type}")

        pdf = pd.DataFrame(data, dtype=dtype, columns=feature_cols)
        pdf[label_col] = label.astype(label_dtype)
        df = spark.createDataFrame(
            pdf,
            ",".join(schema),
        )
    else:
        df = spark.createDataFrame(data.tolist(), ",".join(schema))

    if feature_type == feature_types.array:
        # avoid calling df.withColumn here because runtime slowdown is observed when df has many columns (e.g. 3000).
        from pyspark.sql.functions import col

        selected_col = [array(*feature_cols).alias("features")]
        if label_col:
            selected_col.append(col(label_col).alias(label_col))
        df = df.select(selected_col)

        feature_cols = "features"
    elif feature_type == feature_types.vector:
        df = (
            VectorAssembler()
            .setInputCols(feature_cols)  # type: ignore
            .setOutputCol("features")
            .transform(df)
            .drop(*feature_cols)
        )
        feature_cols = "features"
    else:
        # When df has many columns (e.g. 3000), and was created by calling spark.createDataFrame on a pandas DataFrame,
        # calling df.withColumn can lead to noticeable runtime slowdown.
        # Using select here can significantly reduce the runtime and improve the performance.
        df = df.select("*")

    return df, feature_cols, label_col


def array_equal(
    lhs: Union[np.ndarray, List[float]],
    rhs: Union[np.ndarray, List[float]],
    unit_tol: float = 1e-4,
    total_tol: float = 0,
    with_sign: bool = True,
) -> bool:
    a = np.asarray(lhs)
    b = np.asarray(rhs)
    if len(a) == 0 and len(b) == 0:
        return True

    if not with_sign:
        a, b = np.abs(a), np.abs(b)
    res = (np.sum(np.abs(a - b) > unit_tol)) / a.size <= total_tol
    return res


def assert_params(
    instance: CumlParams, spark_params: Dict[str, Any], cuml_params: Dict[str, Any]
) -> None:
    for key in spark_params:
        if instance.hasParam(key):
            if instance.isDefined(key):
                actual = instance.getOrDefault(key)
                expected = spark_params[key]
                assert (
                    actual == expected
                ), f"Value of '{key}' Param was {actual}, expected {expected}."
            elif spark_params[key] != None:
                assert False, f"Value of {key} Param is undefined."
    for key in cuml_params:
        if key in instance.cuml_params:
            actual = instance.cuml_params[key]
            expected = cuml_params[key]
            assert (
                actual == expected
            ), f"Value of '{key}' cuml_param was {actual}, expected {expected}."


@lru_cache(4)
def _make_classification_dataset_from_cache(
    nrows: int, ncols: int, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache the dataset"""
    return _make_classification_dataset_uncached(nrows, ncols, **kwargs)


def _make_classification_dataset_uncached(
    nrows: int, ncols: int, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create classification dataset.

    return X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        **kwargs, n_samples=nrows, n_features=ncols, random_state=0
    )
    return train_test_split(X, y, train_size=0.8, random_state=10)


def make_classification_dataset(
    datatype: np.dtype, nrows: int, ncols: int, **kwargs: Any
) -> Iterator[np.ndarray]:
    """Create classification dataset"""
    if nrows * ncols < 1e8:  # Keep cache under 4 GB
        dataset = _make_classification_dataset_from_cache(nrows, ncols, **kwargs)
    else:
        dataset = _make_classification_dataset_uncached(nrows, ncols, **kwargs)

    return map(lambda arr: arr.astype(datatype), dataset)


def get_default_cuml_parameters(
    cuml_classes: List[type], excludes: List[str] = []
) -> Dict[str, Any]:
    params = {}
    for cuml_cls in cuml_classes:
        params.update(_get_default_params_from_func(cuml_cls, excludes))
    return params


def get_toy_model(EstimatorCLS: Callable, spark: SparkSession) -> Model:
    data = [
        Row(id=0, label=1.0, weight=1.0, features=Vectors.dense([0.0, 0.0, 1.0])),
        Row(id=1, label=1.0, weight=1.0, features=Vectors.dense([0.0, 1.0, 0.0])),
        Row(id=2, label=0.0, weight=1.0, features=Vectors.dense([1.0, 0.0, 0.0])),
        Row(id=3, label=0.0, weight=1.0, features=Vectors.dense([2.0, 0.0, -1.0])),
    ]
    train_df = spark.createDataFrame(data)

    if "spark_rapids_ml" in EstimatorCLS.__module__:
        est = EstimatorCLS(num_workers=1)
    else:
        est = EstimatorCLS()

    if est.hasParam("inputCol"):
        est.setInputCol("features")
    elif est.hasParam("featuresCol"):
        est.setFeaturesCol("features")
    else:
        assert False, "an Estimator must contain inputCol or featuresCol"

    if est.hasParam("labelCol"):
        est.setLabelCol("label")

    if est.hasParam("idCol"):
        est.setIdCol("id")

    model = est.fit(train_df)
    return model

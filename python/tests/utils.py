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

from collections import namedtuple
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pyspark
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import array
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
) -> Tuple[pyspark.sql.DataFrame, Union[str, List[str]], Optional[str]]:
    """Construct a dataframe based on features and label data."""
    assert feature_type in pyspark_supported_feature_types

    m, n = data.shape

    pyspark_type = dtype_to_pyspark_type(dtype)
    feature_cols: Union[str, List[str]] = [f"c{i}" for i in range(n)]
    schema = [f"{c} {pyspark_type}" for c in feature_cols]
    label_col = None

    if label is not None:
        label_col = "label_col"
        schema.append(f"{label_col} {pyspark_type}")
        df = spark.createDataFrame(
            np.concatenate((data, label.reshape(m, 1)), axis=1).tolist(),
            ",".join(schema),
        )
    else:
        df = spark.createDataFrame(data.tolist(), ",".join(schema))

    if feature_type == feature_types.array:
        df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)
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

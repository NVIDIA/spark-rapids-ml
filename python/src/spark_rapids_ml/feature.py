#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import itertools
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import cudf

import numpy as np
import pandas as pd
from pyspark.ml.common import _py2java
from pyspark.ml.feature import PCAModel as SparkPCAModel
from pyspark.ml.feature import _PCAParams
from pyspark.ml.linalg import DenseMatrix, DenseVector
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
)

from .core import (
    CumlInputType,
    CumlT,
    _CumlEstimator,
    _CumlModelWithColumns,
    param_alias,
)
from .params import P, _CumlClass, _CumlParams
from .utils import (
    PartitionDescriptor,
    _get_spark_session,
    dtype_to_pyspark_type,
    java_uid,
)


class PCAClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {"k": "n_components"}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_components": None,
            "svd_solver": "auto",
            "verbose": False,
            "whiten": False,
        }


class _PCACumlParams(_CumlParams, _PCAParams, HasInputCols):
    """
    Shared Spark Params for PCA and PCAModel.
    """

    def setInputCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`inputCol` or :py:attr:`inputCols`. Used when input vectors are stored in a single column.
        """
        if isinstance(value, str):
            self.set_params(inputCol=value)
        else:
            self.set_params(inputCols=value)
        return self

    def setInputCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`inputCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self.set_params(inputCols=value)

    def setOutputCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`outputCol`
        """
        return self.set_params(outputCol=value)


class PCA(PCAClass, _CumlEstimator, _PCACumlParams):
    """
    PCA algorithm learns principal component vectors to project high-dimensional vectors
    into low-dimensional vectors, while preserving the similarity of the vectors. PCA
    has been used in dimensionality reduction, clustering, and data visualization on large
    datasets. This class provides GPU acceleration for pyspark distributed PCA.

    Parameters
    ----------
    k: int
        the number of components, or equivalently the dimension that all vectors will be projected to.
    inputCol: str
        the name of the column that contains input vectors. inputCol should be set when input vectors
        are stored in a single column of a dataframe.

    inputCols: List[str]
        the names of feature columns that form input vectors. inputCols should be set when input vectors
        are stored as multiple feature columns of a dataframe.

    outputCol: str
        the name of the column that stores output vectors. outputCol should be set when users expect to
        store output vectors in a single column.


    Examples
    --------
    >>> from spark_rapids_ml.feature import PCA
    >>> data = [([1.0, 1.0],),
    ...         ([2.0, 2.0],),
    ...         ([3.0, 3.0],),]
    >>> df = spark.createDataFrame(data, ["features"])
    >>> gpu_pca = PCA(k=1, inputCol="features")
    >>> gpu_pca.setOutputCol("pca_features")
    PCA...
    >>> gpu_model = gpu_pca.fit(df)
    >>> gpu_model.getK()
    1
    >>> print(gpu_model.mean)
    [2.0, 2.0]
    >>> print(gpu_model.pc)
    DenseMatrix([[0.70710678],
                 [0.70710678]])
    >>> print(gpu_model.explainedVariance)
    [1.0]
    >>> gpu_pca.save("/tmp/pca")

    >>> # vector column input
    >>> from pyspark.ml.linalg import Vectors
    >>> data = [(Vectors.dense([1.0, 1.0]),),
    ...         (Vectors.dense([2.0, 2.0]),),
    ...         (Vectors.dense([3.0, 3.0]),),]
    >>> df = spark.createDataFrame(data, ["features"])
    >>> gpu_pca = PCA(k=1).setInputCol("features")
    >>> gpu_pca.getInputCol()
    'features'
    >>> gpu_model = gpu_pca.fit(df)

    >>> # multi-column input
    >>> data = [(1.0, 1.0),
    ...         (2.0, 2.0),
    ...         (3.0, 3.0),]
    >>> df = spark.createDataFrame(data, ["f1", "f2"])
    >>> gpu_pca = PCA(k=1).setInputCols(["f1", "f2"])
    >>> gpu_pca.getInputCols()
    ['f1', 'f2']
    >>> gpu_model = gpu_pca.fit(df)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setK(self, value: int) -> "PCA":
        """
        Sets the value of :py:attr:`k`.
        """
        return self.set_params(k=value)

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        def _cuml_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.decomposition.pca_mg import PCAMG as CumlPCAMG

            pca_object = CumlPCAMG(
                handle=params[param_alias.handle],
                output_type="cudf",
                **params[param_alias.cuml_init],
            )

            pdesc = PartitionDescriptor.build(
                params[param_alias.part_sizes], params[param_alias.num_cols]
            )
            pca_object.fit(
                [x for x, _, _ in dfs],
                pdesc.m,
                pdesc.n,
                pdesc.parts_rank_size,
                pdesc.rank,
                _transform=False,
            )

            cpu_mean = pca_object.mean_.to_arrow().to_pylist()
            cpu_pc = pca_object.components_.to_numpy().tolist()
            cpu_explained_variance = (
                pca_object.explained_variance_ratio_.to_numpy().tolist()
            )
            cpu_singular_values = pca_object.singular_values_.to_numpy().tolist()

            return {
                "mean_": [cpu_mean],
                "components_": [cpu_pc],
                "explained_variance_ratio_": [cpu_explained_variance],
                "singular_values_": [cpu_singular_values],
                "n_cols": params[param_alias.num_cols],
                "dtype": pca_object.dtype.name,
            }

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("mean_", ArrayType(DoubleType(), False), False),
                StructField(
                    "components_", ArrayType(ArrayType(DoubleType()), False), False
                ),
                StructField(
                    "explained_variance_ratio_", ArrayType(DoubleType(), False), False
                ),
                StructField("singular_values_", ArrayType(DoubleType(), False), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "PCAModel":
        return PCAModel.from_row(result)


class PCAModel(PCAClass, _CumlModelWithColumns, _PCACumlParams):
    """Applies dimensionality reduction on an input DataFrame.

    Note: Input vectors must be zero-centered to ensure PCA work properly.
    Spark PCA does not automatically remove the mean of the input data, so use the
    :py:class::`~pyspark.ml.feature.StandardScaler` to center the input data before
    invoking transform.

    Examples
    --------
    >>> from spark_rapids_ml.feature import PCA
    >>> data = [([-1.0, -1.0],),
    ...         ([0.0, 0.0],),
    ...         ([1.0, 1.0],),]
    >>> df = spark.createDataFrame(data, ["features"])
    >>> gpu_pca = PCA(k=1).setInputCol("features").setOutputCol("pca_features")
    >>> gpu_model = gpu_pca.fit(df)
    >>> reduced_df = gpu_model.transform(df)
    >>> reduced_df.show()
    +---------------------+
    |         pca_features|
    +---------------------+
    | [-1.414213562373095]|
    |                [0.0]|
    |  [1.414213562373095]|
    +---------------------+
    """

    def __init__(
        self,
        mean_: List[float],
        components_: List[List[float]],
        explained_variance_ratio_: List[float],
        singular_values_: List[float],
        n_cols: int,
        dtype: str,
    ):
        super().__init__(
            n_cols=n_cols,
            dtype=dtype,
            mean_=mean_,
            components_=components_,
            explained_variance_ratio_=explained_variance_ratio_,
            singular_values_=singular_values_,
        )

        self.mean_ = mean_
        self.components_ = components_
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.singular_values_ = singular_values_
        self._pca_ml_model: Optional[SparkPCAModel] = None

        self.set_params(n_components=len(components_))

    @property
    def mean(self) -> List[float]:
        """
        Returns the mean of the input vectors.
        """
        return self.mean_

    @property
    def pc(self) -> DenseMatrix:
        """
        Returns a principal components Matrix.
        Each column is one principal component.
        """
        num_rows = len(self.components_)
        num_cols = self.n_cols
        values = list(itertools.chain.from_iterable(self.components_))
        # DenseMatrix is column major, so flip rows/cols
        return DenseMatrix(num_cols, num_rows, values, False)  # type: ignore

    @property
    def explainedVariance(self) -> DenseVector:
        """
        Returns a vector of proportions of variance
        explained by each principal component.
        """
        return DenseVector(self.explained_variance_ratio_)

    def cpu(self) -> SparkPCAModel:
        """Return the PySpark ML PCAModel"""
        if self._pca_ml_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            java_pc = _py2java(sc, self.pc)
            java_explainedVariance = _py2java(sc, self.explainedVariance)
            java_model = sc._jvm.org.apache.spark.ml.feature.PCAModel(
                java_uid(sc, "pca"), java_pc, java_explainedVariance
            )
            self._pca_ml_model = SparkPCAModel(java_model)
            self._copyValues(self._pca_ml_model)

        return self._pca_ml_model

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union["cudf.DataFrame", np.ndarray]], pd.DataFrame],
    ]:
        cuml_alg_params = self.cuml_params.copy()

        n_cols = self.n_cols
        dype = self.dtype
        components = self.components_
        mean = self.mean_
        singular_values = self.singular_values_

        def _construct_pca() -> CumlT:
            """
            Returns the instance of PCAMG which will be passed to _transform_internal
            to do the transform.
            -------

            """
            from cuml.decomposition.pca_mg import PCAMG as CumlPCAMG

            pca = CumlPCAMG(output_type="numpy", **cuml_alg_params)

            # Compatible with older cuml versions (before 23.02)
            pca._n_components = pca.n_components
            pca.n_components_ = pca.n_components

            from spark_rapids_ml.utils import cudf_to_cuml_array

            pca.n_cols = n_cols
            pca.dtype = np.dtype(dype)

            # TBD: figure out why PCA warns regardless of array order here and for singular values
            pca.components_ = cudf_to_cuml_array(
                np.array(components, order="F").astype(pca.dtype)
            )
            pca.mean_ = cudf_to_cuml_array(np.array(mean, order="F").astype(pca.dtype))
            pca.singular_values_ = cudf_to_cuml_array(
                np.array(singular_values, order="F").astype(pca.dtype)
            )
            return pca

        transformed_mean = np.matmul(
            np.array(self.mean_, self.dtype),
            np.array(self.components_, self.dtype).T,
        )

        def _transform_internal(
            pca_object: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:
            res = pca_object.transform(df)
            # Spark does not remove the mean from the transformed data,
            # but cuML does, so need to add the mean back to match Spark results
            res += transformed_mean

            return pd.Series(list(res))

        return _construct_pca, _transform_internal

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        assert self.dtype is not None

        pyspark_type = dtype_to_pyspark_type(self.dtype)
        return f"array<{pyspark_type}>"

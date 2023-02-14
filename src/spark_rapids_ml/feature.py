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

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cudf
import numpy as np
import pandas as pd
from pyspark.ml.feature import _PCAParams
from pyspark.ml.linalg import DenseMatrix, DenseVector
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
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

from spark_rapids_ml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimator,
    _CumlModel,
)
from spark_rapids_ml.params import _CumlClass, _CumlParams
from spark_rapids_ml.utils import PartitionDescriptor


class PCAClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml import PCA

        return [PCA]

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {"k": "n_components"}

    @classmethod
    def _param_excludes(cls) -> List[str]:
        return [
            "copy",
            "handle",
            "iterated_power",
            "output_type",
            "random_state",
            "tol",
        ]


class _PCACumlParams(_CumlParams, _PCAParams, HasInputCols, HasOutputCols):
    """
    Shared Spark Params for PCA and PCAModel.
    """

    def setInputCol(self, value: Union[str, List[str]]) -> "_PCACumlParams":
        """
        Sets the value of :py:attr:`inputCol` or :py:attr:`inputCols`.
        Used when input vectors are stored in a single column.
        
        Examples
        --------
        >>> from spark_rapids_ml.feature import PCA
        >>> data = [([1.0, 1.0],),
        ...         ([2.0, 2.0],),
        ...         ([3.0, 3.0],),]
        >>> df = spark.createDataFrame(data, ["features"])
        >>> gpu_pca = PCA(k=1).setInputCol("features")
        >>> gpu_pca.getInputCol()
        'features'
        >>> gpu_model = gpu_pca.fit(df)

        >>> from pyspark.ml.linalg import Vectors
        >>> data = [(Vectors.dense([1.0, 1.0]),),
        ...         (Vectors.dense([2.0, 2.0]),),
        ...         (Vectors.dense([3.0, 3.0]),),]
        >>> df = spark.createDataFrame(data, ["features"])
        >>> gpu_pca = PCA(k=1).setInputCol("features")
        >>> gpu_pca.getInputCol()
        'features'
        >>> gpu_model = gpu_pca.fit(df)
        """
        if isinstance(value, str):
            self.set_params(inputCol=value)
        else:
            self.set_params(inputCols=value)
        return self

    def setInputCols(self, value: List[str]) -> "_PCACumlParams":
        """
        Sets the value of :py:attr:`inputCols`.
        Used when input vectors are stored as multiple feature columns. 

        Examples
        --------
         >>> data = [(1.0, 1.0),
         ...         (2.0, 2.0),
         ...         (3.0, 3.0),]
         >>> df = spark.createDataFrame(data, ["f1", "f2"])
         >>> gpu_pca = PCA(k=1).setInputCols(["f1", "f2"])
         >>> gpu_pca.getInputCols() 
         ['f1', 'f2']
         >>> gpu_model = gpu_pca.fit(df)
        """
        return self.set_params(inputCols=value)

    def setOutputCol(self, value: Union[str, List[str]]) -> "_PCACumlParams":
        """
        Sets the value of :py:attr:`outputCol` or py:attr:`outputCols`
        """
        if isinstance(value, str):
            self.set_params(outputCol=value)
        else:
            self.set_params(outputCols=value)
        return self

    def setOutputCols(self, value: List[str]) -> "_PCACumlParams":
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self.set_params(outputCols=value)


class PCA(PCAClass, _CumlEstimator, _PCACumlParams):
    """
    PCA algorithm learns principal component vectors to project high-dimensional vectors
    into low-dimensional vectors, while preserving the similarity of the vectors. PCA 
    has been used in dimensionality reduction, clustering, and data visualization on large 
    datasets. This class provides GPU acceleration for pyspark distributed PCA.

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

    Parameters
    ----------
    k: int
        the number of components, or equivalently the dimension that all vectors will be projected to.   
    inputCol: str
        the name of the column that contains input vectors. inputCol should be set when input vectors are stored in a single column of a dataframe. 

    inputCols: List[str]
        the names of feature columns that form input vectors. inputCols should be set when input vectors are stored as multiple feature columns of a dataframe. 
        
    outputCol: str
        the name of the column that stores output vectors. outputCol should be set when users expect to store output vectors in a single column.  

    outputCols: List[str]
        the name of the feature columns that form output vectors. outputCols should be set when users expect to store output vectors as multiple feature columns.   

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
                handle=params["handle"],
                output_type="cudf",
                **params[INIT_PARAMETERS_NAME],
            )

            pdesc = PartitionDescriptor.build(params["part_sizes"], params["n"])
            pca_object.fit(
                [x for x, _ in dfs],
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
                "n_cols": params["n"],
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


class PCAModel(PCAClass, _CumlModel, _PCACumlParams):
    """Applies dimensionality reduction on an input DataFrame.

    Note: Input vectors must be zero-centered to ensure PCA work properly. 
    Spark PCA does not automatically remove the mean of the input data, so use the
    :py:class::`StandardScaler` to center the input data before invoking transform.

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

        self.set_params(n_components=len(components_))

    @property
    def mean(self) -> List[float]:
        return self.mean_

    @property
    def pc(self) -> DenseMatrix:
        numRows = len(self.components_)
        numCols = self.n_cols
        values = list(itertools.chain.from_iterable(self.components_))
        # DenseMatrix is column major, so flip rows/cols
        return DenseMatrix(numCols, numRows, values, False)  # type: ignore

    @property
    def components(self) -> List[List[float]]:
        return self.components_

    @property
    def explainedVariance(self) -> DenseVector:
        return DenseVector(self.explained_variance_ratio_)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        if self.isDefined(self.inputCol):
            input_column_name = self.getInputCol()
        else:
            input_column_name = self.getInputCols()[0]

        for field in input_schema:
            if field.name == input_column_name:
                # TODO: mypy throws error here since it doesn't know that dataType will be ArrayType which has elementType field
                input_data_type = field.dataType.elementType if self.isDefined(self.inputCol) else field.dataType  # type: ignore
                break

        if self.isDefined(self.outputCols):
            output_cols = self.getOutputCols()
            ret_schema = StructType(
                [
                    StructField(col_name, input_data_type, False)
                    for col_name in output_cols
                ]
            )
        else:
            ret_schema = StructType(
                [
                    StructField(
                        self.getOutputCol(), ArrayType(input_data_type, False), False
                    )
                ]
            )
        return ret_schema

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:

        cuml_alg_params = self.cuml_params.copy()

        transformed_mean = np.matmul(
            np.array(self.mean_, self.dtype),
            np.array(self.components_, self.dtype).T,
        )

        def _construct_pca() -> CumlT:
            """

            Returns the instance of PCAMG which will be passed to _transform_internal
            to do the transform.
            -------

            """
            from cuml.decomposition.pca_mg import PCAMG as CumlPCAMG

            pca = CumlPCAMG(output_type="cudf", **cuml_alg_params)
            pca._n_components = pca.n_components

            from spark_rapids_ml.utils import cudf_to_cuml_array

            pca.n_cols = self.n_cols
            pca.dtype = np.dtype(self.dtype)
            pca.components_ = cudf_to_cuml_array(
                np.array(self.components_).astype(pca.dtype)
            )
            pca.mean_ = cudf_to_cuml_array(np.array(self.mean_).astype(pca.dtype))
            pca.singular_values_ = cudf_to_cuml_array(
                np.array(self.singular_values_).astype(pca.dtype)
            )
            return pca

        def _transform_internal(
            pca_object: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:
            res = pca_object.transform(df).to_numpy()
            # if num_components is 1, a 1-d numpy array is returned
            # convert to 2d for correct downstream behavior

            if len(res.shape) == 1:
                res = np.expand_dims(res, 1)

            # Spark does not remove the mean from the transformed data,
            # but cuML does, so need to add the mean back to match Spark results
            res += transformed_mean

            if self.isDefined(self.outputCols):
                return pd.DataFrame(res, columns=self.getOutputCols())
            else:
                res = list(res)
                return pd.DataFrame({self.getOutputCol(): res})

        return _construct_pca, _transform_internal

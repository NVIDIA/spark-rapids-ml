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
from typing import Any, Callable, Dict, List, Tuple, Union

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
from spark_rapids_ml.params import _CumlClass
from spark_rapids_ml.utils import PartitionDescriptor


class PCAClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml import PCA

        return [PCA]

    @classmethod
    def _param_mapping(cls) -> Dict[str, str]:
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


class PCA(PCAClass, _CumlEstimator, _PCAParams, HasInputCols, HasOutputCols):
    """
    PCA algorithm projects high-dimensional vectors into low-dimensional vectors
    while preserving the similarity of the vectors. This class provides GPU accleration for pyspark mllib PCA.

    Examples
    --------
    >>> from spark_rapids_ml.feature import PCA
    >>> data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    >>> topk = 1
    >>> gpu_pca = PCA().setInputCol("features").setK(topk)
    >>> df = spark.sparkContext.parallelize(data).map(lambda row: (row,)).toDF(["features"])
    >>> gpu_model = gpu_pca.fit(df)
    >>> print(gpu_model.mean)
    [2.0, 2.0]
    >>> print(gpu_model.pc)
    DenseMatrix([[0.70710678],
             [0.70710678]])
    >>> print(gpu_model.explained_variance)
    [1.0]

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setK(self, value: int) -> "PCA":
        """
        Sets the value of :py:attr:`k`.
        """
        return self.set_params(k=value)

    def setInputCol(self, value: str) -> "PCA":
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self.set_params(inputCol=value)

    def setInputCols(self, value: List[str]) -> "PCA":
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self.set_params(inputCols=value)

    def setOutputCol(self, value: str) -> "PCA":
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self.set_params(outputCol=value)

    def setOutputCols(self, value: List[str]) -> "PCA":
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self.set_params(outputCols=value)

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


class PCAModel(PCAClass, _CumlModel, _PCAParams, HasInputCols, HasOutputCols):
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

    def setInputCol(self, value: str) -> "PCAModel":
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self.set_params(inputCol=value)

    def setInputCols(self, value: List[str]) -> "PCAModel":
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self.set_params(inputCols=value)

    def setOutputCol(self, value: str) -> "PCAModel":
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self.set_params(outputCol=value)

    def setOutputCols(self, value: List[str]) -> "PCAModel":
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self.set_params(outputCols=value)

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
            # TODO: Spark doesn't auto-normalize the inputs like CuML, so add mean back
            # df = df + np.array(self.mean_, self.dtype)
            res = pca_object.transform(df).to_numpy()
            # if num_components is 1, a 1-d numpy array is returned
            # convert to 2d for correct downstream behavior
            if len(res.shape) == 1:
                res = np.expand_dims(res, 1)

            if self.isDefined(self.outputCols):
                return pd.DataFrame(res, columns=self.getOutputCols())
            else:
                res = list(res)
                return pd.DataFrame({self.getOutputCol(): res})

        return _construct_pca, _transform_internal

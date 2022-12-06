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
from typing import Any, Callable, Dict, List, Tuple, Union

import cudf
import numpy as np
import pandas as pd
from pyspark import Row
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModel,
    _CumlModelSupervised,
    _set_pyspark_cuml_cls_param_attrs,
)
from sparkcuml.utils import PartitionDescriptor, cudf_to_cuml_array


class SparkLinearRegression(_CumlEstimatorSupervised):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setFeaturesCol(self, value: Union[str, List[str]]) -> "SparkLinearRegression":
        """
        Sets the value of `inputCol` or `inputCols`.
        """
        if isinstance(value, str):
            self.set_params(inputCol=value)
        else:
            self.set_params(inputCols=value)
        return self

    def getFeaturesCol(self) -> Union[str, List[str]]:
        if self.isDefined(self.inputCols):
            return self.getInputCols()
        elif self.isDefined(self.inputCol):
            return self.getInputCol()
        else:
            raise RuntimeError("features col is not set")

    def setLabelCol(self, value: str) -> "SparkLinearRegression":
        self._set(labelCol=value)  # type: ignore
        return self

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        def _linear_regression_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.linear_model.linear_regression_mg import LinearRegressionMG

            linear_regression = LinearRegressionMG(
                handle=params["handle"],
                output_type="cudf",
                **params[INIT_PARAMETERS_NAME],
            )

            pdesc = PartitionDescriptor.build(params["part_sizes"], params["n"])
            linear_regression.fit(
                dfs,
                pdesc.m,
                pdesc.n,
                pdesc.parts_rank_size,
                pdesc.rank,
            )

            return {
                "coef": [linear_regression.coef_.to_numpy().tolist()],
                "intercept": linear_regression.intercept_,
                "dtype": linear_regression.dtype.name,
                "n_cols": linear_regression.n_cols,
            }

        return _linear_regression_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("coef", ArrayType(DoubleType(), False), False),
                StructField("intercept", DoubleType(), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "SparkLinearRegressionModel":
        return SparkLinearRegressionModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> type:
        from cuml.linear_model.linear_regression import LinearRegression

        return LinearRegression

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        return ["handle", "output_type"]


class SparkLinearRegressionModel(_CumlModelSupervised):
    def __init__(
        self,
        coef: List[float],
        intercept: float,
        n_cols: int,
        dtype: str,
    ) -> None:
        super().__init__(dtype=dtype, n_cols=n_cols, coef=coef, intercept=intercept)
        self.coef = coef
        self.intercept = intercept
        cuml_params = SparkLinearRegression._get_cuml_params_default()
        self.set_params(**cuml_params)

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        coef = self.coef
        intercept = self.intercept
        n_cols = self.n_cols
        dtype = self.dtype

        def _construct_lr() -> CumlT:
            from cuml.linear_model.linear_regression_mg import LinearRegressionMG

            lr = LinearRegressionMG(output_type="numpy")
            lr.coef_ = cudf_to_cuml_array(np.array(coef).astype(dtype))
            lr.intercept_ = intercept
            lr.n_cols = n_cols
            lr.dtype = np.dtype(dtype)

            return lr

        def _predict(lr: CumlT, pdf: Union[cudf.DataFrame, np.ndarray]) -> pd.Series:
            ret = lr.predict(pdf)
            return pd.Series(ret)

        return _construct_lr, _predict


_set_pyspark_cuml_cls_param_attrs(SparkLinearRegression, SparkLinearRegressionModel)

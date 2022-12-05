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
from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModel,
    _set_pyspark_cuml_cls_param_attrs,
)
from sparkcuml.utils import PartitionDescriptor


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
            }

        return _linear_regression_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("coef", ArrayType(DoubleType(), False), False),
                StructField("intercept", DoubleType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "_CumlModel":
        return SparkLinearRegressionModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> type:
        from cuml.linear_model.linear_regression import LinearRegression

        return LinearRegression

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        return ["handle", "output_type"]


class SparkLinearRegressionModel(_CumlModel):
    def __init__(self, coef: List[float], intercept: float) -> None:
        super().__init__(coef=coef, intercept=intercept)
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
        pass

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        pass


_set_pyspark_cuml_cls_param_attrs(SparkLinearRegression, SparkLinearRegressionModel)

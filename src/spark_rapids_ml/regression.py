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
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.regression import _LinearRegressionParams
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from spark_rapids_ml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModelSupervised,
)
from spark_rapids_ml.params import _CumlClass
from spark_rapids_ml.utils import PartitionDescriptor, cudf_to_cuml_array


class LinearRegressionClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        # from cuml.dask.linear_model import LinearRegression
        from cuml.linear_model.linear_regression import LinearRegression
        from cuml.linear_model.ridge import Ridge
        from cuml.solvers import CD

        return [LinearRegression, Ridge, CD]

    @classmethod
    def _param_mapping(cls) -> Dict[str, str]:
        return {
            "elasticNetParam": "l1_ratio",
            "fitIntercept": "fit_intercept",
            "loss": "loss",
            "maxIter": "max_iter",
            "regParam": "alpha",
            "solver": "solver",
            "standardization": "normalize",
            "tol": "tol",
        }

    @classmethod
    def _param_value_mapping(cls) -> Dict[str, Dict[str, Union[str, None]]]:
        return {
            "loss": {"squaredError": "squared_loss", "huber": None},
            "solver": {"auto": "eig", "normal": "eig", "l-bfgs": None},
        }

    @classmethod
    def _param_excludes(cls) -> List[str]:
        return ["handle", "output_type"]


class LinearRegression(
    LinearRegressionClass,
    _CumlEstimatorSupervised,
    _LinearRegressionParams,
    HasInputCols,
):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setFeaturesCol(self, value: str) -> "LinearRegression":
        """
        Sets the value of :py:attr:`featuresCol`.
        """
        return self.set_params(featuresCol=value)

    def setInputCols(self, value: List[str]) -> "LinearRegression":
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self.set_params(inputCols=value)

    def setMaxIter(self, value: int) -> "LinearRegression":
        """
        Sets the value of :py:attr:`maxIter`.
        """
        return self.set_params(maxIter=value)

    def setRegParam(self, value: float) -> "LinearRegression":
        """
        Sets the value of :py:attr:`regParam`.
        """
        return self.set_params(regParam=value)

    def setElasticNetParam(self, value: float) -> "LinearRegression":
        """
        Sets the value of :py:attr:`elasticNetParam`.
        """
        return self.set_params(elasticNetParam=value)

    def setLabelCol(self, value: str) -> "LinearRegression":
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self.set_params(labelCol=value)

    def setLoss(self, value: str) -> "LinearRegression":
        """
        Sets the value of :py:attr:`loss`.
        """
        return self.set_params(loss=value)

    def setStandardization(self, value: str) -> "LinearRegression":
        """
        Sets the value of :py:attr:`standardization`.
        """
        return self.set_params(standardization=value)

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        def _linear_regression_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            init_parameters = params[INIT_PARAMETERS_NAME]

            if init_parameters["alpha"] == 0:
                # LR
                from cuml.linear_model.linear_regression_mg import (
                    LinearRegressionMG as CumlLinearRegression,
                )

                supported_params = [
                    "algorithm",
                    "fit_intercept",
                    "normalize",
                    "verbose",
                ]
            else:
                if init_parameters["l1_ratio"] == 0:
                    # LR + L2
                    from cuml.linear_model.ridge_mg import (
                        RidgeMG as CumlLinearRegression,
                    )

                    supported_params = [
                        "alpha",
                        "solver",
                        "fit_intercept",
                        "normalize",
                        "verbose",
                    ]

                else:
                    # LR + L1, or LR + L1 + L2
                    # Cuml uses Coordinate Descent algorithm to implement Lasso and ElasticNet
                    # So combine Lasso and ElasticNet here.
                    from cuml.solvers.cd_mg import CDMG as CumlLinearRegression

                    supported_params = [
                        "loss",
                        "alpha",
                        "l1_ratio",
                        "fit_intercept",
                        "normalize",
                        "tol",
                        "shuffle",
                        "verbose",
                    ]

            # filter only supported params
            init_parameters = {
                k: v for k, v in init_parameters.items() if k in supported_params
            }

            linear_regression = CumlLinearRegression(
                handle=params["handle"],
                output_type="cudf",
                **init_parameters,
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
                "coef_": [linear_regression.coef_.to_numpy().tolist()],
                "intercept_": linear_regression.intercept_,
                "dtype": linear_regression.dtype.name,
                "n_cols": linear_regression.n_cols,
            }

        return _linear_regression_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("coef_", ArrayType(DoubleType(), False), False),
                StructField("intercept_", DoubleType(), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "LinearRegressionModel":
        return LinearRegressionModel.from_row(result)


class LinearRegressionModel(
    LinearRegressionClass, _CumlModelSupervised, _LinearRegressionParams, HasInputCols
):
    def __init__(
        self,
        coef_: List[float],
        intercept_: float,
        n_cols: int,
        dtype: str,
    ) -> None:
        super().__init__(dtype=dtype, n_cols=n_cols, coef_=coef_, intercept_=intercept_)
        self.coef_ = coef_
        self.intercept_ = intercept_

    @property
    def coefficients(self) -> List[float]:
        return self.coef_

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        coef_ = self.coef_
        intercept_ = self.intercept_
        n_cols = self.n_cols
        dtype = self.dtype

        def _construct_lr() -> CumlT:
            from cuml.linear_model.linear_regression_mg import LinearRegressionMG

            lr = LinearRegressionMG(output_type="numpy")
            lr.coef_ = cudf_to_cuml_array(np.array(coef_).astype(dtype))
            lr.intercept_ = intercept_
            lr.n_cols = n_cols
            lr.dtype = np.dtype(dtype)

            return lr

        def _predict(lr: CumlT, pdf: Union[cudf.DataFrame, np.ndarray]) -> pd.Series:
            ret = lr.predict(pdf)
            return pd.Series(ret)

        return _construct_lr, _predict

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
from pyspark.ml.param.shared import HasElasticNetParam, HasRegParam
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
    _CumlModelSupervised,
    _set_pyspark_cuml_cls_param_attrs,
)
from sparkcuml.utils import (
    PartitionDescriptor,
    _get_default_params_from_func,
    cudf_to_cuml_array,
)

_lr_unsupported_params = ["handle", "output_type", "alpha", "l1_ratio"]


class _LinearRegressionParams(HasRegParam, HasElasticNetParam):
    """
    Spark wraps L1 and L2 into LinearRegression. It uses elasticNetParam and regParam
    to decide which regularization will be used.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._setDefault(  # type: ignore
            regParam=0.0,
            elasticNetParam=0.0,
        )


class SparkCumlLinearRegression(_CumlEstimatorSupervised, _LinearRegressionParams):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setFeaturesCol(
        self, value: Union[str, List[str]]
    ) -> "SparkCumlLinearRegression":
        """
        Sets the value of `inputCol` or `inputCols`.
        """
        if isinstance(value, str):
            self.set_params(inputCol=value)
        else:
            self.set_params(inputCols=value)
        return self

    def setRegParam(self, value: float) -> "SparkCumlLinearRegression":
        """
        Sets the value of :py:attr:`regParam`.
        """
        return self._set(regParam=value)  # type: ignore

    def setElasticNetParam(self, value: float) -> "SparkCumlLinearRegression":
        """
        Sets the value of :py:attr:`elasticNetParam`.
        """
        return self._set(elasticNetParam=value)  # type: ignore

    def getFeaturesCol(self) -> Union[str, List[str]]:
        if self.isDefined(self.inputCols):
            return self.getInputCols()
        elif self.isDefined(self.inputCol):
            return self.getInputCol()
        else:
            raise RuntimeError("features col is not set")

    def setLabelCol(self, value: str) -> "SparkCumlLinearRegression":
        self._set(labelCol=value)  # type: ignore
        return self

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:

        # alpha
        reg = self.getRegParam()

        # L1 ratio
        elastic_net = self.getElasticNetParam()

        def _linear_regression_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            init_parameters = params[INIT_PARAMETERS_NAME]

            if reg == 0:
                # LR
                from cuml.linear_model.linear_regression import (
                    LinearRegression as LREstimator,
                )
                from cuml.linear_model.linear_regression_mg import (
                    LinearRegressionMG as CumlLinearRegression,
                )

                other_params = []
            else:
                if elastic_net == 0:
                    # LR + L2
                    from cuml.linear_model.ridge import Ridge as LREstimator
                    from cuml.linear_model.ridge_mg import (
                        RidgeMG as CumlLinearRegression,
                    )

                    other_params = ["alpha"]

                else:
                    # LR + L1, or LR + L1 + L2
                    # Cuml uses Coordinate Descent algorithm to implement Lasso and ElasticNet
                    # So combine Lasso and ElasticNet here.
                    from cuml.solvers import CD as LREstimator
                    from cuml.solvers.cd_mg import CDMG as CumlLinearRegression

                    other_params = ["alpha", "l1_ratio"]

            init_parameters["alpha"] = reg
            init_parameters["l1_ratio"] = elastic_net
            param_names = list(
                _get_default_params_from_func(
                    LREstimator, _lr_unsupported_params
                ).keys()
            )
            param_names.extend(other_params)

            init_parameters = dict((k, init_parameters[k]) for k in param_names)

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

    def _create_pyspark_model(self, result: Row) -> "SparkCumlLinearRegressionModel":
        return SparkCumlLinearRegressionModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml.linear_model.linear_regression import LinearRegression
        from cuml.linear_model.ridge import Ridge
        from cuml.solvers import CD

        return [LinearRegression, Ridge, CD]

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        # "alpha" is replaced by regParam
        return _lr_unsupported_params


class SparkCumlLinearRegressionModel(_CumlModelSupervised):
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
        cuml_params = SparkCumlLinearRegression._get_cuml_params_default()
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


_set_pyspark_cuml_cls_param_attrs(
    SparkCumlLinearRegression, SparkCumlLinearRegressionModel
)

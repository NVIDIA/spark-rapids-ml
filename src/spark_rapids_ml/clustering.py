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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cudf
import numpy as np
import pandas as pd
from pyspark.ml.clustering import _KMeansParams
from pyspark.ml.linalg import Vector
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

from spark_rapids_ml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimator,
    _CumlModel,
    _CumlModelSupervised,
)
from spark_rapids_ml.params import HasFeaturesCols, _CumlClass, _CumlParams
from spark_rapids_ml.utils import get_logger


class KMeansClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml import KMeans

        return [KMeans]

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "distanceMeasure": None,
            "k": "n_clusters",
            "initSteps": "",
            "maxIter": "max_iter",
            "seed": "random_state",
            "tol": "tol",
            "weightCol": None,
        }

    @classmethod
    def _param_excludes(cls) -> List[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return [
            "handle",
            "output_type",
        ]


class _KMeansCumlParams(_CumlParams, _KMeansParams, HasFeaturesCols):
    """
    Shared Spark Params for KMeans and KMeansModel.
    """

    def __init__(self) -> None:
        super().__init__()
        # restrict default seed to max value of 32-bit signed integer for CuML
        self._setDefault(seed=hash(type(self).__name__) & 0x07FFFFFFF)

    def getFeaturesCol(self) -> Union[str, List[str]]:  # type: ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(self, value: str) -> "_KMeansCumlParams":
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self, value: List[str]) -> "_KMeansCumlParams":
        """
        Sets the value of :py:attr:`featuresCols`.
        """
        return self.set_params(featuresCols=value)

    def setPredictionCol(self, value: str) -> "_KMeansCumlParams":
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self.set_params(predictionCol=value)
        return self


class KMeans(KMeansClass, _CumlEstimator, _KMeansCumlParams):
    """
    KMeans algorithm partitions data points into a fixed number (denoted as k) of clusters.
    The algorithm initializes a set of k random centers then runs in iterations.
    In each iteration, KMeans assigns every point to its nearest center,
    then calculates a new set of k centers.

    Examples
    --------
    >>> from spark_rapids_ml.clustering import KMeans
    TODO
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setK(self, value: int) -> "KMeans":
        """
        Sets the value of :py:attr:`k`.
        """
        return self.set_params(k=value)

    def setMaxIter(self, value: int) -> "KMeans":
        """
        Sets the value of :py:attr:`maxIter`.
        """
        return self.set_params(maxIter=value)

    def setSeed(self, value: int) -> "KMeans":
        """
        Sets the value of :py:attr:`seed`.
        """
        if value > 0x07FFFFFFF:
            raise ValueError("CuML seed value must be a 32-bit integer.")
        return self.set_params(seed=value)

    def setWeightCol(self, value: str) -> "KMeans":
        """
        Sets the value of :py:attr:`weightCol`.
        """
        raise ValueError("'weightCol' is not supported by cuML.")

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:
        cls = self.__class__

        def _cuml_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans_object = CumlKMeansMG(
                handle=params["handle"],
                output_type="cudf",
                **params[INIT_PARAMETERS_NAME],
            )
            df_list = [x for (x, _) in dfs]
            if isinstance(df_list[0], pd.DataFrame):
                concated = pd.concat(df_list)
            else:
                # should be list of np.ndarrays here
                concated = np.concatenate(df_list)

            kmeans_object.fit(
                concated,
                sample_weight=None,
            )

            logger = get_logger(cls)
            # TBD: inertia is always 0 for some reason
            logger.info(
                f"iterations: {kmeans_object.n_iter_}, inertia: {kmeans_object.inertia_}"
            )

            return {
                "cluster_centers_": [
                    kmeans_object.cluster_centers_.to_numpy().tolist()
                ],
                "n_cols": params["n"],
                "dtype": kmeans_object.dtype.name,
            }

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "cluster_centers_", ArrayType(ArrayType(DoubleType()), False), False
                ),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "KMeansModel":
        return KMeansModel.from_row(result)


class KMeansModel(KMeansClass, _CumlModelSupervised, _KMeansCumlParams):
    def __init__(
        self,
        cluster_centers_: List[List[float]],
        n_cols: int,
        dtype: str,
    ):
        super(KMeansModel, self).__init__(
            n_cols=n_cols, dtype=dtype, cluster_centers_=cluster_centers_
        )

        self.cluster_centers_ = cluster_centers_

    def clusterCenters(self) -> List[List[float]]:
        return self.cluster_centers_

    @property
    def hasSummary(self) -> bool:
        return False

    def predict(self, value: Vector) -> int:
        raise NotImplementedError(
            "'predict' method is not supported, use 'transform' instead."
        )

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        ret_schema = "int"
        return ret_schema

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        cuml_alg_params = self.cuml_params.copy()

        cluster_centers_ = self.cluster_centers_
        output_col = self.getPredictionCol()
        dtype = self.dtype
        n_cols = self.n_cols

        def _construct_kmeans() -> CumlT:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans = CumlKMeansMG(output_type="cudf", **cuml_alg_params)
            from spark_rapids_ml.utils import cudf_to_cuml_array

            kmeans.n_cols = n_cols
            kmeans.dtype = np.dtype(dtype)
            kmeans.cluster_centers_ = cudf_to_cuml_array(
                np.array(cluster_centers_).astype(dtype), order="C"
            )
            return kmeans

        def _transform_internal(
            kmeans: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.Series:
            res = list(kmeans.predict(df, normalize_weights=False).to_numpy())
            return pd.Series(res)

        return _construct_kmeans, _transform_internal

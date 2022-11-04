from typing import Any, Callable, Dict, List, Union

import cudf
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import ArrayType, DoubleType, Row, StructField, StructType

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    _CumlEstimator,
    _CumlModel,
    _set_pyspark_cuml_cls_param_attrs,
)


class SparkCumlKMeans(_CumlEstimator):
    """
    KMeans algorithm partitions data points into a fixed number (denoted as k) of clusters.
    The algorithm initializes a set of k random centers then runs in iterations.
    In each iteration, KMeans assigns every point to its nearest center,
    then calculates a new set of k centers.

    Examples
    --------
    >>> from sparkcuml.cluster import SparkCumlKMeans
    TODO
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setK(self, value: int) -> "SparkCumlKMeans":
        """
        Sets the value of `n_clusters`.
        """
        self.set_params(n_clusters=value)
        return self

    def setFeaturesCol(self, value: str) -> "SparkCumlKMeans":
        """
        Sets the value of `inputCol`.
        """
        self.set_params(inputCol=value)
        return self

    def setPredictionCol(self, value: str) -> "SparkCumlKMeans":
        """
        Sets the value of `outputCol`.
        """
        self.set_params(outputCol=value)
        return self

    def setMaxIter(self, value: int) -> "SparkCumlKMeans":
        """
        Sets the value of `max_iter`.
        """
        self.set_params(max_iter=value)
        return self

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[List[cudf.DataFrame], Dict[str, Any]], Dict[str, Any]]:
        def _cuml_fit(
            df: List[cudf.DataFrame], params: Dict[str, Any]
        ) -> Dict[str, Any]:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans_object = CumlKMeansMG(
                handle=params["handle"],
                output_type="cudf",
                **params[INIT_PARAMETERS_NAME],
            )

            concated = cudf.concat(df)
            kmeans_object.fit(
                concated,
                sample_weight=None,
            )

            return {
                "cluster_centers_": [
                    kmeans_object.cluster_centers_.to_numpy().tolist()
                ],
            }

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "cluster_centers_", ArrayType(ArrayType(DoubleType()), False), False
                ),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "SparkCumlKMeansModel":
        return SparkCumlKMeansModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> type:
        from cuml import KMeans

        return KMeans

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return [
            "handle",
            "output_type",
        ]


class SparkCumlKMeansModel(_CumlModel):
    def __init__(
        self,
        cluster_centers_: List[List[float]],
    ):
        super().__init__()

        self.cluster_centers_ = cluster_centers_

        cumlParams = SparkCumlKMeans._get_cuml_params_default()
        self.set_params(**cumlParams)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        pass

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Callable[[cudf.DataFrame], pd.DataFrame]:
        pass


_set_pyspark_cuml_cls_param_attrs(SparkCumlKMeans, SparkCumlKMeansModel)

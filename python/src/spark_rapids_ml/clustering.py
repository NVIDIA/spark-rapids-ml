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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeansModel as SparkKMeansModel
from pyspark.ml.clustering import _KMeansParams
from pyspark.ml.linalg import Vector
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
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlEstimator,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    param_alias,
    transform_evaluate,
)
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .utils import (
    _ArrayOrder,
    _concat_and_free,
    _get_spark_session,
    get_logger,
    java_uid,
)


class KMeansClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "distanceMeasure": None,
            "initMode": "init",
            "k": "n_clusters",
            "initSteps": "",
            "maxIter": "max_iter",
            "seed": "random_state",
            "tol": "tol",
            "weightCol": None,
        }

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_clusters": 8,
            "max_iter": 300,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
            "init": "scalable-k-means++",
            "n_init": 1,
            "oversampling_factor": 2.0,
            "max_samples_per_batch": 32768,
        }


class _KMeansCumlParams(_CumlParams, _KMeansParams, HasFeaturesCols):
    """
    Shared Spark Params for KMeans and KMeansModel.
    """

    def __init__(self) -> None:
        super().__init__()
        # restrict default seed to max value of 32-bit signed integer for cuML
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

    def setFeaturesCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`. Used when input vectors are stored in a single column.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self.set_params(featuresCols=value)

    def setPredictionCol(self: P, value: str) -> P:
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
    then calculates a new set of k centers. KMeans often deals with large datasets.
    This class provides GPU acceleration for pyspark distributed KMeans.

    Parameters
    ----------
    k: int (default = 8)
        the number of centers. Set this parameter to enable KMeans to learn k centers from input vectors.

    maxIter: int (default = 300)
        the maximum iterations the algorithm will run to learn the k centers.
        More iterations help generate more accurate centers.

    seed: int (default = 1)
        the random seed used by the algorithm to initialize a set of k random centers to start with.

    tol: float (default = 1e-4)
        early stopping criterion if centers do not change much after an iteration.

    featuresCol: str
        the name of the column that contains input vectors. featuresCol should be set when input vectors are stored in a single column of a dataframe.

    featuresCols: List[str]
        the names of feature columns that form input vectors. featureCols should be set when input vectors are stored as multiple feature columns of a dataframe.

    predictionCol: str
        the name of the column that stores cluster indices of input vectors. predictionCol should be set when users expect to apply the transform function of a learned model.

    Examples
    --------
    >>> from spark_rapids_ml.clustering import KMeans
    >>> data = [([0.0, 0.0],),
    ...         ([1.0, 1.0],),
    ...         ([9.0, 8.0],),
    ...         ([8.0, 9.0],),]
    >>> df = spark.createDataFrame(data, ["features"])
    >>> df.show()
    +----------+
    |  features|
    +----------+
    |[0.0, 0.0]|
    |[1.0, 1.0]|
    |[9.0, 8.0]|
    |[8.0, 9.0]|
    +----------+
    >>> gpu_kmeans = KMeans(k=2).setFeaturesCol("features")
    >>> gpu_kmeans.setMaxIter(10)
    KMeans_5606dff6b4fa
    >>> gpu_model = gpu_kmeans.fit(df)
    >>> gpu_model.setPredictionCol("prediction")
    >>> gpu_model.clusterCenters()
    [[0.5, 0.5], [8.5, 8.5]]
    >>> transformed = gpu_model.transform(df)
    >>> transformed.show()
    +----------+----------+
    |  features|prediction|
    +----------+----------+
    |[0.0, 0.0]|         0|
    |[1.0, 1.0]|         0|
    |[9.0, 8.0]|         1|
    |[8.0, 9.0]|         1|
    +----------+----------+
    >>> gpu_kmeans.save("/tmp/kmeans")
    >>> gpu_model.save("/tmp/kmeans_model")

    >>> # vector column input
    >>> from spark_rapids_ml.clustering import KMeans
    >>> from pyspark.ml.linalg import Vectors
    >>> data = [(Vectors.dense([0.0, 0.0]),),
    ...         (Vectors.dense([1.0, 1.0]),),
    ...         (Vectors.dense([9.0, 8.0]),),
    ...         (Vectors.dense([8.0, 9.0]),),]
    >>> df = spark.createDataFrame(data, ["features"])
    >>> gpu_kmeans = KMeans(k=2).setFeaturesCol("features")
    >>> gpu_kmeans.getFeaturesCol()
    'features'
    >>> gpu_model = gpu_kmeans.fit(df)

    >>> # multi-column input
    >>> data = [(0.0, 0.0),
    ...         (1.0, 1.0),
    ...         (9.0, 8.0),
    ...         (8.0, 9.0),]
    >>> df = spark.createDataFrame(data, ["f1", "f2"])
    >>> gpu_kmeans = KMeans(k=2).setFeaturesCols(["f1", "f2"])
    >>> gpu_kmeans.getFeaturesCols()
    ['f1', 'f2']
    >>> gpu_kmeans = gpu_kmeans.fit(df)
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
            raise ValueError("cuML seed value must be a 32-bit integer.")
        return self.set_params(seed=value)

    def setTol(self, value: float) -> "KMeans":
        """
        Sets the value of :py:attr:`tol`.
        """
        return self.set_params(tol=value)

    def setWeightCol(self, value: str) -> "KMeans":
        """
        Sets the value of :py:attr:`weightCol`.
        """
        raise ValueError("'weightCol' is not supported by cuML.")

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
        cls = self.__class__

        array_order = self._fit_array_order()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            import cupy as cp
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans_object = CumlKMeansMG(
                handle=params[param_alias.handle],
                output_type="cudf",
                **params[param_alias.cuml_init],
            )
            df_list = [x for (x, _, _) in dfs]
            if isinstance(df_list[0], pd.DataFrame):
                concated = pd.concat(df_list)
            else:
                # features are either cp or np arrays here
                concated = _concat_and_free(df_list, order=array_order)

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
                "n_cols": params[param_alias.num_cols],
                "dtype": str(kmeans_object.dtype.name),
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


class KMeansModel(KMeansClass, _CumlModelWithPredictionCol, _KMeansCumlParams):
    """
    KMeans gpu model for clustering input vectors to learned k centers.
    Refer to the KMeans class for learning the k centers.
    """

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
        self._kmeans_spark_model: Optional[SparkKMeansModel] = None

    def cpu(self) -> SparkKMeansModel:
        """Return the PySpark ML KMeansModel"""
        if self._kmeans_spark_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            from pyspark.mllib.common import _py2java
            from pyspark.mllib.linalg import _convert_to_vector

            java_centers = _py2java(
                sc, [_convert_to_vector(c) for c in self.cluster_centers_]
            )
            java_mllib_model = sc._jvm.org.apache.spark.mllib.clustering.KMeansModel(
                java_centers
            )
            java_model = sc._jvm.org.apache.spark.ml.clustering.KMeansModel(
                java_uid(sc, "kmeans"), java_mllib_model
            )
            self._kmeans_spark_model = SparkKMeansModel(java_model)

        return self._kmeans_spark_model

    def clusterCenters(self) -> List[np.ndarray]:
        """Returns the list of cluster centers."""
        return [np.array(x) for x in self.cluster_centers_]

    @property
    def hasSummary(self) -> bool:
        """Indicates whether a training summary exists for this model instance."""
        return False

    def predict(self, value: Vector) -> int:
        """Predict label for the given features.
        cuML doesn't support predicting 1 single sample.
        Fall back to PySpark ML KMeansModel"""
        return self.cpu().predict(value)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        ret_schema = "int"
        return ret_schema

    def _transform_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        cuml_alg_params = self.cuml_params.copy()

        cluster_centers_ = self.cluster_centers_
        dtype = self.dtype
        n_cols = self.n_cols
        array_order = self._transform_array_order()

        def _construct_kmeans() -> CumlT:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans = CumlKMeansMG(output_type="cudf", **cuml_alg_params)
            from spark_rapids_ml.utils import cudf_to_cuml_array

            kmeans.n_cols = n_cols
            kmeans.dtype = np.dtype(dtype)
            kmeans.cluster_centers_ = cudf_to_cuml_array(
                np.array(cluster_centers_).astype(dtype), order=array_order
            )
            return kmeans

        def _transform_internal(
            kmeans: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.Series:
            res = list(kmeans.predict(df, normalize_weights=False).to_numpy())
            return pd.Series(res)

        return _construct_kmeans, _transform_internal, None

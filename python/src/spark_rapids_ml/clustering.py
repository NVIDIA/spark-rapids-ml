#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import pyspark
from pyspark import keyword_only
from pyspark.ml.clustering import KMeansModel as SparkKMeansModel
from pyspark.ml.clustering import _KMeansParams
from pyspark.ml.linalg import Vector
from pyspark.ml.param.shared import HasFeaturesCol, Param, Params, TypeConverters
from pyspark.sql import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    Row,
    StringType,
    StructField,
    StructType,
)

from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlCaller,
    _CumlEstimator,
    _CumlModel,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
)
from .metrics import EvalMetricInfo
from .params import HasFeaturesCols, HasIDCol, P, _CumlClass, _CumlParams
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
        param_map = {
            "distanceMeasure": None,
            "initMode": "init",
            "k": "n_clusters",
            "initSteps": "",
            "maxIter": "max_iter",
            "seed": "random_state",
            "tol": "tol",
            "weightCol": None,
            "solver": "",
            "maxBlockSizeInMB": "",
        }

        import pyspark
        from packaging import version

        if version.parse(pyspark.__version__) < version.parse("3.4.0"):
            param_map.pop("solver")
            param_map.pop("maxBlockSizeInMB")

        return param_map

    @classmethod
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[Any], Union[None, str, float, int]]]:
        def tol_value_mapper(x: float) -> float:
            if x == 0.0:
                logger = get_logger(cls)
                logger.warning(
                    "tol=0 is not supported in cuml yet. "
                    + "It will be mapped to smallest positive float, i.e. numpy.finfo('float32').tiny."
                )

                return np.finfo("float32").tiny.item()
            else:
                return x

        return {"tol": lambda x: tol_value_mapper(x)}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_clusters": 8,
            "max_iter": 300,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
            "init": "scalable-k-means++",
            "n_init": "warn",  # See https://github.com/rapidsai/cuml/pull/6142 - this needs to be updated to "auto" for cuml 25.04
            "oversampling_factor": 2.0,
            "max_samples_per_batch": 32768,
        }

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return pyspark.ml.clustering.KMeans


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
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`.
        """
        if isinstance(value, str):
            self._set_params(featuresCol=value)
        else:
            self._set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self._set_params(featuresCols=value)

    def setPredictionCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self._set_params(predictionCol=value)
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
    k: int (default = 2)
        the number of centers. Set this parameter to enable KMeans to learn k centers from input vectors.

    initMode: str (default = "k-means||")
        the algorithm to select initial centroids. It can be "k-means||" or "random".

    maxIter: int (default = 20)
        the maximum iterations the algorithm will run to learn the k centers.
        More iterations help generate more accurate centers.

    seed: int (default = None)
        the random seed used by the algorithm to initialize a set of k random centers to start with.

    tol: float (default = 1e-4)
        early stopping criterion if centers do not change much after an iteration.

    featuresCol: str or List[str]
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.

    predictionCol: str
        the name of the column that stores cluster indices of input vectors. predictionCol should be set when users expect to apply the transform function of a learned model.

    num_workers:
        Number of cuML workers, where each cuML worker corresponds to one Spark task
        running on one GPU. If not set, spark-rapids-ml tries to infer the number of
        cuML workers (i.e. GPUs in cluster) from the Spark environment.

    verbose:
    Logging level.
            * ``0`` - Disables all log messages.
            * ``1`` - Enables only critical messages.
            * ``2`` - Enables all messages up to and including errors.
            * ``3`` - Enables all messages up to and including warnings.
            * ``4 or False`` - Enables all messages up to and including information messages.
            * ``5 or True`` - Enables all messages up to and including debug messages.
            * ``6`` - Enables all messages up to and including trace messages.

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

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        predictionCol: str = "prediction",
        k: int = 2,
        initMode: str = "k-means||",
        tol: float = 0.0001,
        maxIter: int = 20,
        seed: Optional[int] = None,
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._set_params(**self._input_kwargs)

    def setInitMode(self, value: str) -> "KMeans":
        """
        Sets the value of :py:attr:`initMode`.
        """
        return self._set_params(initMode=value)

    def setK(self, value: int) -> "KMeans":
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set_params(k=value)

    def setMaxIter(self, value: int) -> "KMeans":
        """
        Sets the value of :py:attr:`maxIter`.
        """
        return self._set_params(maxIter=value)

    def setSeed(self, value: int) -> "KMeans":
        """
        Sets the value of :py:attr:`seed`.
        """
        if value > 0x07FFFFFFF:
            raise ValueError("cuML seed value must be a 32-bit integer.")
        return self._set_params(seed=value)

    def setTol(self, value: float) -> "KMeans":
        """
        Sets the value of :py:attr:`tol`.
        """
        return self._set_params(tol=value)

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
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
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
        return KMeansModel._from_row(result)


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
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        cuml_alg_params = self.cuml_params.copy()

        cluster_centers_ = self.cluster_centers_
        dtype = self.dtype
        n_cols = self.n_cols
        array_order = self._transform_array_order()

        def _construct_kmeans() -> CumlT:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans = CumlKMeansMG(output_type="cudf", **cuml_alg_params)
            from spark_rapids_ml.utils import cudf_to_cuml_array

            kmeans.n_features_in_ = n_cols
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


class DBSCANClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "algorithm": "brute",
            "verbose": False,
            "max_mbytes_per_batch": None,
        }

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None


class _DBSCANCumlParams(_CumlParams, HasFeaturesCol, HasFeaturesCols, HasIDCol):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault(
            eps=0.5,
            min_samples=5,
            metric="euclidean",
            algorithm="brute",
            max_mbytes_per_batch=None,
            idCol=alias.row_number,
        )

    eps = Param(
        Params._dummy(),
        "eps",
        (
            f"The maximum distance between 2 points such they reside in the same neighborhood."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    min_samples = Param(
        Params._dummy(),
        "min_samples",
        (
            f"The number of samples in a neighborhood such that this group can be considered as an important core point (including the point itself)."
        ),
        typeConverter=TypeConverters.toInt,
    )

    metric = Param(
        Params._dummy(),
        "metric",
        (
            f"The metric to use when calculating distances between points."
            f"Spark Rapids ML does not support the 'precomputed' mode from sklearn and cuML, please use those libraries instead."
        ),
        typeConverter=TypeConverters.toString,
    )

    algorithm = Param(
        Params._dummy(),
        "algorithm",
        (f"The algorithm to be used by for nearest neighbor computations."),
        typeConverter=TypeConverters.toString,
    )

    max_mbytes_per_batch = Param(
        Params._dummy(),
        "max_mbytes_per_batch",
        (
            f"Calculate batch size using no more than this number of megabytes for the pairwise distance computation."
            f"This enables the trade-off between runtime and memory usage for making the N^2 pairwise distance computations more tractable for large numbers of samples."
            f"If you are experiencing out of memory errors when running DBSCAN, you can set this value based on the memory size of your device."
        ),
        typeConverter=TypeConverters.toInt,
    )

    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column name.",
        typeConverter=TypeConverters.toString,
    )

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
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`.
        """
        if isinstance(value, str):
            self._set_params(featuresCol=value)
        else:
            self._set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self._set_params(featuresCols=value)

    def setPredictionCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self._set_params(predictionCol=value)
        return self

    def setIdCol(self: P, value: str) -> P:
        """
        Sets the value of `idCol`. If not set, an id column will be added with column name `unique_id`. The id column is used to specify dbscan vectors by associated id value.
        """
        self._set_params(idCol=value)
        return self


class DBSCAN(DBSCANClass, _CumlEstimator, _DBSCANCumlParams):
    """
    The Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a non-parametric
    data clustering algorithm based on data density. It groups points close to each other that form a dense cluster
    and mark the far-away points as noise and exclude them from all clusters.

    Parameters
    ----------
    featuresCol: str or List[str] (default = "features")
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.

    predictionCol: str (default = "prediction")
        the name of the column that stores cluster indices of input vectors. predictionCol should be set when users expect to apply the transform function of a learned model.

    num_workers:
        Number of cuML workers, where each cuML worker corresponds to one Spark task
        running on one GPU. If not set, spark-rapids-ml tries to infer the number of
        cuML workers (i.e. GPUs in cluster) from the Spark environment.

    eps: float (default = 0.5)
        The maximum distance between 2 points such they reside in the same neighborhood.

    min_samples: int (default = 5)
        The number of samples in a neighborhood such that this group can be considered as
        an important core point (including the point itself).

    metric: {'euclidean', 'cosine'}, default = 'euclidean'
        The metric to use when calculating distances between points.
        Spark Rapids ML does not support the 'precomputed' mode from sklearn and cuML, please use those libraries instead

    algorithm: {'brute', 'rbc'}, default = 'brute'
        The algorithm to be used by for nearest neighbor computations.

    verbose: int or boolean (default=False)
        Logging level.
            * ``0`` - Disables all log messages.
            * ``1`` - Enables only critical messages.
            * ``2`` - Enables all messages up to and including errors.
            * ``3`` - Enables all messages up to and including warnings.
            * ``4 or False`` - Enables all messages up to and including information messages.
            * ``5 or True`` - Enables all messages up to and including debug messages.
            * ``6`` - Enables all messages up to and including trace messages.

    max_mbytes_per_batch(optional): int
        Calculate batch size using no more than this number of megabytes for the pairwise distance computation.
        This enables the trade-off between runtime and memory usage for making the N^2 pairwise distance computations more tractable for large numbers of samples.
        If you are experiencing out of memory errors when running DBSCAN, you can set this value based on the memory size of your device.

    idCol: str (default = 'unique_id')
        The internal unique id column name for label matching, will not reveal in the output.
        Need to be set to a name that does not conflict with an existing column name in the original input data.

    Note: We currently do not support calculating and storing the indices of the core samples via the parameter calc_core_sample_indices=True.

    Examples
    ----------
    >>> from spark_rapids_ml.clustering import DBSCAN
    >>> data = [([0.0, 0.0],),
    ...        ([1.0, 1.0],),
    ...        ([9.0, 8.0],),
    ...        ([8.0, 9.0],),]
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
    >>> gpu_dbscan = DBSCAN(eps=3, metric="euclidean").setFeaturesCol("features")
    >>> gpu_model = gpu_dbscan.fit(df)
    >>> gpu_model.setPredictionCol("prediction")
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
    >>> gpu_dbscan.save("/tmp/dbscan")
    >>> gpu_model.save("/tmp/dbscan_model")

    >>> # vector column input
    >>> from spark_rapids_ml.clustering import DBSCAN
    >>> from pyspark.ml.linalg import Vectors
    >>> data = [(Vectors.dense([0.0, 0.0]),),
    ...        (Vectors.dense([1.0, 1.0]),),
    ...        (Vectors.dense([9.0, 8.0]),),
    ...        (Vectors.dense([8.0, 9.0]),),]
    >>> df = spark.createDataFrame(data, ["features"])
    >>> gpu_dbscan = DBSCAN(eps=3, metric="euclidean").setFeaturesCol("features")
    >>> gpu_dbscan.getFeaturesCol()
    'features'
    >>> gpu_model = gpu_dbscan.fit(df)


    >>> # multi-column input
    >>> data = [(0.0, 0.0),
    ...        (1.0, 1.0),
    ...        (9.0, 8.0),
    ...        (8.0, 9.0),]
    >>> df = spark.createDataFrame(data, ["f1", "f2"])
    >>> gpu_dbscan = DBSCAN(eps=3, metric="euclidean").setFeaturesCols(["f1", "f2"])
    >>> gpu_dbscan.getFeaturesCols()
    ['f1', 'f2']
    >>> gpu_model = gpu_dbscan.fit(df)
    """

    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: Union[str, List[str]] = "features",
        predictionCol: str = "prediction",
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "brute",
        max_mbytes_per_batch: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._set_params(**self._input_kwargs)

        max_records_per_batch_str = _get_spark_session().conf.get(
            "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
        )
        assert max_records_per_batch_str is not None
        self.max_records_per_batch = int(max_records_per_batch_str)
        self.BROADCAST_LIMIT = 8 << 30
        self.cuml_params["calc_core_sample_indices"] = False  # currently not supported

    def setEps(self: P, value: float) -> P:
        return self._set_params(eps=value)

    def getEps(self) -> float:
        return self.getOrDefault("eps")

    def setMinSamples(self: P, value: int) -> P:
        return self._set_params(min_samples=value)

    def getMinSamples(self) -> int:
        return self.getOrDefault("min_samples")

    def setMetric(self: P, value: str) -> P:
        return self._set_params(metric=value)

    def getMetric(self) -> str:
        return self.getOrDefault("metric")

    def setAlgorithm(self: P, value: str) -> P:
        return self._set_params(algorithm=value)

    def getAlgorithm(self) -> str:
        return self.getOrDefault("algorithm")

    def setMaxMbytesPerBatch(self: P, value: Optional[int]) -> P:
        return self._set_params(max_mbytes_per_batch=value)

    def getMaxMbytesPerBatch(self) -> Optional[int]:
        return self.getOrDefault("max_mbytes_per_batch")

    def _fit(self, dataset: DataFrame) -> _CumlModel:
        if self.getMetric() == "precomputed":
            raise ValueError(
                "Spark Rapids ML does not support the 'precomputed' mode from sklearn and cuML, please use those libraries instead"
            )

        # Create parameter-copied model without accessing the input dataframe
        # All information will be retrieved from Model and transform
        model = DBSCANModel(n_cols=0, dtype="")

        model._num_workers = self.num_workers
        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore

        return model

    def _create_pyspark_model(self, result: Row) -> _CumlModel:
        raise NotImplementedError("DBSCAN does not support model creation from Row")

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        raise NotImplementedError("DBSCAN does not fit and generate model")

    def _out_schema(self) -> Union[StructType, str]:
        raise NotImplementedError("DBSCAN does not output for fit and generate model")


class DBSCANModel(
    DBSCANClass, _CumlCaller, _CumlModelWithPredictionCol, _DBSCANCumlParams
):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
    ):
        super(DBSCANClass, self).__init__()
        super(_CumlModelWithPredictionCol, self).__init__(n_cols=n_cols, dtype=dtype)
        super(_DBSCANCumlParams, self).__init__()

        self._setDefault(
            idCol=alias.row_number,
        )

        self.BROADCAST_LIMIT = 8 << 30
        self._dbscan_spark_model = None

    def _pre_process_data(self, dataset: DataFrame) -> Tuple[  # type: ignore
        List[Column],
        Optional[List[str]],
        int,
        Union[Type[FloatType], Type[DoubleType]],
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = _CumlCaller._pre_process_data(self, dataset)

        # Must retain idCol for label matching
        if self.hasParam("idCol") and self.isDefined("idCol"):
            id_col_name = self.getOrDefault("idCol")
            select_cols.append(col(id_col_name))
        else:
            select_cols.append(col(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type

    def _out_schema(
        self, input_schema: StructType = StructType()
    ) -> Union[StructType, str]:
        return StructType(
            [
                StructField(self._get_prediction_name(), IntegerType(), False),
                StructField(self.getIdCol(), LongType(), False),
            ]
        )

    def _transform_array_order(self) -> _ArrayOrder:
        return "C"

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (True, True)

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        import cupy as cp
        import cupyx

        dtype = self.dtype
        n_cols = self.n_cols
        array_order = self._fit_array_order()
        pred_name = self._get_prediction_name()
        idCol_name = self.getIdCol()

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )

        idCol_bc = self.idCols_
        raw_data_bc = self.raw_data_
        data_size = self.data_size

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.cluster.dbscan_mg import DBSCANMG as CumlDBSCANMG
            from pyspark import BarrierTaskContext

            inputs = []  # type: ignore

            idCol = list(
                idCol_bc[0].value
                if len(idCol_bc) == 1
                else np.concatenate([chunk.value for chunk in idCol_bc])
            )

            for pdf_bc in raw_data_bc:
                features = pdf_bc.value

                # experiments indicate it is faster to convert to numpy array and then to cupy array than directly
                # invoking cupy array on the list
                if cuda_managed_mem_enabled:
                    features = cp.array(features)

                inputs.append(features)

            concated = _concat_and_free(inputs, order=array_order)

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()

            dbscan = CumlDBSCANMG(
                handle=params[param_alias.handle],
                output_type="cudf",
                **params[param_alias.cuml_init],
            )
            dbscan.n_cols = params[param_alias.num_cols]
            dbscan.dtype = np.dtype(dtype)

            # Set out_dtype tp 64bit to get larger indexType in cuML for avoiding overflow
            out_dtype = np.int32 if data_size < 2147000000 else np.int64
            res = list(dbscan.fit_predict(concated, out_dtype=out_dtype).to_numpy())

            # Only node 0 from cuML will contain the correct label output
            if partition_id == 0:
                return {
                    idCol_name: idCol,
                    pred_name: res,
                }
            else:
                return {
                    idCol_name: [],
                    pred_name: [],
                }

        return _cuml_fit

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        raise NotImplementedError("DBSCAN does not have a separate transform UDF")

    def _transform(self, dataset: DataFrame) -> DataFrame:
        logger = get_logger(self.__class__)

        spark = _get_spark_session()

        def _chunk_arr(
            arr: np.ndarray, BROADCAST_LIMIT: int = self.BROADCAST_LIMIT
        ) -> List[np.ndarray]:
            """Chunk an array, if oversized, into smaller arrays that can be broadcasted."""
            if arr.nbytes <= BROADCAST_LIMIT:
                return [arr]

            rows_per_chunk = BROADCAST_LIMIT // (arr.nbytes // arr.shape[0])
            num_chunks = (arr.shape[0] + rows_per_chunk - 1) // rows_per_chunk
            chunks = [
                arr[i * rows_per_chunk : (i + 1) * rows_per_chunk]
                for i in range(num_chunks)
            ]

            return chunks

        dataset = self._ensureIdCol(dataset)
        select_cols, multi_col_names, dimension, _ = self._pre_process_data(dataset)
        input_dataset = dataset.select(*select_cols)
        pd_dataset: pd.DataFrame = input_dataset.toPandas()

        if multi_col_names:
            raw_data = np.array(
                pd_dataset.drop(columns=[self.getIdCol()]),
                order=self._fit_array_order(),
            )
        else:
            raw_data = np.array(
                list(pd_dataset.drop(columns=[self.getIdCol()])[alias.data]),
                order=self._fit_array_order(),
            )

        self.data_size = len(raw_data) * len(raw_data[0])
        idCols: np.ndarray = np.array(pd_dataset[self.getIdCol()])

        # Set input metadata
        self.n_cols = len(raw_data[0])
        self.dtype = (
            type(raw_data[0][0][0]).__name__
            if isinstance(raw_data[0][0], List)
            or isinstance(raw_data[0][0], np.ndarray)
            else type(raw_data[0][0]).__name__
        )

        # Broadcast preprocessed input dataset and the idCol
        broadcast_raw_data = [
            spark.sparkContext.broadcast(chunk) for chunk in _chunk_arr(raw_data)
        ]

        broadcast_idCol = [
            spark.sparkContext.broadcast(chunk) for chunk in _chunk_arr(idCols)
        ]

        self.processed_input_cols = input_dataset.drop(self.getIdCol()).columns
        self.raw_data_ = broadcast_raw_data
        self.idCols_ = broadcast_idCol
        self.multi_col_names = multi_col_names

        idCol_name = self.getIdCol()

        default_num_partitions = dataset.rdd.getNumPartitions()

        rdd = self._call_cuml_fit_func(
            dataset=dataset,
            partially_collect=False,
            paramMaps=None,
        )
        rdd = rdd.repartition(default_num_partitions)

        pred_df = rdd.toDF()

        # JOIN the transformed label column into the original input dataset
        # and discard the internal idCol for row matching
        return dataset.join(pred_df, idCol_name).drop(idCol_name)

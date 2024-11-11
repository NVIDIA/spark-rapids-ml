#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import json
import os
from abc import ABCMeta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import pyspark
import scipy
from pandas import DataFrame as PandasDataFrame
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter, MLReader, MLWriter
from pyspark.sql import Column, DataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    Row,
    StructField,
    StructType,
)

from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlCommon,
    _CumlEstimator,
    _CumlEstimatorSupervised,
    _CumlModel,
    _CumlModelReader,
    _CumlModelWithColumns,
    _CumlModelWriter,
    _EvaluateFunc,
    _read_csr_matrix_from_unwrapped_spark_vec,
    _TransformFunc,
    _use_sparse_in_cuml,
    alias,
    param_alias,
)
from .metrics import EvalMetricInfo
from .params import (
    DictTypeConverters,
    HasEnableSparseDataOptim,
    HasFeaturesCols,
    P,
    _CumlClass,
    _CumlParams,
)
from .utils import (
    _ArrayOrder,
    _concat_and_free,
    _get_spark_session,
    _is_local,
    dtype_to_pyspark_type,
    get_logger,
)

if TYPE_CHECKING:
    import cudf
    from pyspark.ml._typing import ParamMap


class UMAPClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_neighbors": 15,
            "n_components": 2,
            "metric": "euclidean",
            "metric_kwds": None,
            "n_epochs": None,
            "learning_rate": 1.0,
            "init": "spectral",
            "min_dist": 0.1,
            "spread": 1.0,
            "set_op_mix_ratio": 1.0,
            "local_connectivity": 1.0,
            "repulsion_strength": 1.0,
            "negative_sample_rate": 5,
            "transform_queue_size": 4.0,
            "a": None,
            "b": None,
            "precomputed_knn": None,
            "random_state": None,
            "verbose": False,
            "build_algo": "auto",
            "build_kwds": None,
        }

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None


class _UMAPCumlParams(
    _CumlParams,
    HasFeaturesCol,
    HasFeaturesCols,
    HasLabelCol,
    HasOutputCol,
    HasEnableSparseDataOptim,
):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault(
            n_neighbors=15,
            n_components=2,
            metric="euclidean",
            metric_kwds=None,
            n_epochs=None,
            learning_rate=1.0,
            init="spectral",
            min_dist=0.1,
            spread=1.0,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            precomputed_knn=None,
            random_state=None,
            build_algo="auto",
            build_kwds=None,
            sample_fraction=1.0,
            outputCol="embedding",
        )

    n_neighbors = Param(
        Params._dummy(),
        "n_neighbors",
        (
            f"The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation."
            f" Larger values result in more global views of the manifold, while smaller values result in more local data being"
            f" preserved. In general values should be in the range 2 to 100."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    n_components = Param(
        Params._dummy(),
        "n_components",
        (
            f"The dimension of the space to embed into. This defaults to 2 to provide easy visualization, but can reasonably"
            f" be set to any integer value in the range 2 to 100."
        ),
        typeConverter=TypeConverters.toInt,
    )

    metric = Param(
        Params._dummy(),
        "metric",
        (
            f"Distance metric to use. Supported distances are ['l1', 'cityblock', 'taxicab', 'manhattan', 'euclidean', 'l2',"
            f" 'sqeuclidean', 'canberra', 'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger', 'hamming',"
            f" 'jaccard'] Metrics that take arguments (such as minkowski) can have arguments passed via the metric_kwds dictionary."
            f" Note: The 'jaccard' distance metric is only supported for sparse inputs."
        ),
        typeConverter=TypeConverters.toString,
    )

    metric_kwds = Param(
        Params._dummy(),
        "metric_kwds",
        (
            f"Additional keyword arguments for the metric function. If the metric function takes additional arguments, they"
            f" should be passed in this dictionary."
        ),
        typeConverter=DictTypeConverters._toDict,
    )

    n_epochs = Param(
        Params._dummy(),
        "n_epochs",
        (
            f"The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in"
            f" more accurate embeddings. If None is specified a value will be selected based on the size of the input dataset"
            f" (200 for large datasets, 500 for small)."
        ),
        typeConverter=TypeConverters.toInt,
    )

    learning_rate = Param(
        Params._dummy(),
        "learning_rate",
        "The initial learning rate for the embedding optimization.",
        typeConverter=TypeConverters.toFloat,
    )

    init = Param(
        Params._dummy(),
        "init",
        (
            f"How to initialize the low dimensional embedding. Options are: 'spectral': use a spectral embedding of the fuzzy"
            f" 1-skeleton, 'random': assign initial embedding positions at random."
        ),
        typeConverter=TypeConverters.toString,
    )

    min_dist = Param(
        Params._dummy(),
        "min_dist",
        (
            f"The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped"
            f" embedding where nearby points on the manifold are drawn closer together, while larger values will result in a"
            f" more even dispersal of points. The value should be set relative to the ``spread`` value, which determines the"
            f" scale at which embedded points will be spread out."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    spread = Param(
        Params._dummy(),
        "spread",
        (
            f"The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped"
            f" the embedded points are."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    set_op_mix_ratio = Param(
        Params._dummy(),
        "set_op_mix_ratio",
        (
            f"Interpolate between (fuzzy) union and intersection as the set operation used to combine local fuzzy simplicial"
            f" sets to obtain a global fuzzy simplicial sets. Both fuzzy set operations use the product t-norm. The value of"
            f" this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a"
            f" pure fuzzy intersection."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    local_connectivity = Param(
        Params._dummy(),
        "local_connectivity",
        (
            f"The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected"
            f" at a local level. The higher this value the more connected the manifold becomes locally. In practice this should"
            f" be not more than the local intrinsic dimension of the manifold."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    repulsion_strength = Param(
        Params._dummy(),
        "repulsion_strength",
        (
            f"Weighting applied to negative samples in low dimensional embedding optimization. Values higher than one will"
            f" result in greater weight being given to negative samples."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    negative_sample_rate = Param(
        Params._dummy(),
        "negative_sample_rate",
        (
            f"The number of negative samples to select per positive sample in the optimization process. Increasing this value"
            f" will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy."
        ),
        typeConverter=TypeConverters.toInt,
    )

    transform_queue_size = Param(
        Params._dummy(),
        "transform_queue_size",
        (
            f"For transform operations (embedding new points using a trained model), this will control how aggressively to"
            f" search for nearest neighbors. Larger values will result in slower performance but more accurate nearest neighbor"
            f" evaluation."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    a = Param(
        Params._dummy(),
        "a",
        (
            f"More specific parameters controlling the embedding. If None these values are set automatically as determined by"
            f" ``min_dist`` and ``spread``."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    b = Param(
        Params._dummy(),
        "b",
        (
            f"More specific parameters controlling the embedding. If None these values are set automatically as determined by"
            f" ``min_dist`` and ``spread``."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    precomputed_knn = Param(
        Params._dummy(),
        "precomputed_knn",
        (
            f"Either one of a tuple (indices, distances) of arrays of shape (n_samples, n_neighbors), a pairwise distances"
            f" dense array of shape (n_samples, n_samples) or a KNN graph sparse array (preferably CSR/COO). This feature"
            f" allows the precomputation of the KNN outside of UMAP and also allows the use of a custom distance function."
            f" This function should match the metric used to train the UMAP embeedings."
        ),
        typeConverter=TypeConverters.toListListFloat,
    )

    random_state = Param(
        Params._dummy(),
        "random_state",
        (
            f"The seed used by the random number generator during embedding initialization and during sampling used by the"
            f" optimizer. Unfortunately, achieving a high amount of parallelism during the optimization stage often comes at"
            f" the expense of determinism, since many floating-point additions are being made in parallel without a"
            f" deterministic ordering. This causes slightly different results across training sessions, even when the same"
            f" seed is used for random number generation. Setting a random_state will enable consistency of trained embeddings,"
            f" allowing for reproducible results to 3 digits of precision, but will do so at the expense of training time and"
            f" memory usage."
        ),
        typeConverter=TypeConverters.toInt,
    )

    build_algo = Param(
        Params._dummy(),
        "build_algo",
        (
            f"How to build the knn graph. Supported build algorithms are ['auto', 'brute_force_knn', 'nn_descent']. 'auto' chooses"
            f" to run with brute force knn if number of data rows is smaller than or equal to 50K. Otherwise, runs with nn descent."
        ),
        typeConverter=TypeConverters.toString,
    )

    build_kwds = Param(
        Params._dummy(),
        "build_kwds",
        (
            f"Build algorithm argument {{'nnd_graph_degree': 64, 'nnd_intermediate_graph_degree': 128, 'nnd_max_iterations': 20,"
            f" 'nnd_termination_threshold': 0.0001, 'nnd_return_distances': True, 'nnd_n_clusters': 1}} Note that nnd_n_clusters > 1"
            f" will result in batch-building with NN Descent."
        ),
        typeConverter=DictTypeConverters._toDict,
    )

    sample_fraction = Param(
        Params._dummy(),
        "sample_fraction",
        (
            f"The fraction of the dataset to be used for fitting the model. Since fitting is done on a single node, very large"
            f" datasets must be subsampled to fit within the node's memory and execute in a reasonable time. Smaller fractions"
            f" will result in faster training, but may result in sub-optimal embeddings."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    def getNNeighbors(self: P) -> float:
        """
        Gets the value of `n_neighbors`.
        """
        return self.getOrDefault("n_neighbors")

    def setNNeighbors(self: P, value: float) -> P:
        """
        Sets the value of `n_neighbors`.
        """
        return self._set_params(n_neighbors=value)

    def getNComponents(self: P) -> int:
        """
        Gets the value of `n_components`.
        """
        return self.getOrDefault("n_components")

    def setNComponents(self: P, value: int) -> P:
        """
        Sets the value of `n_components`.
        """
        return self._set_params(n_components=value)

    def getMetric(self: P) -> str:
        """
        Gets the value of `metric`.
        """
        return self.getOrDefault("metric")

    def setMetric(self: P, value: str) -> P:
        """
        Sets the value of `metric`.
        """
        return self._set_params(metric=value)

    def getMetricKwds(self: P) -> Optional[Dict[str, Any]]:
        """
        Gets the value of `metric_kwds`.
        """
        return self.getOrDefault("metric_kwds")

    def setMetricKwds(self: P, value: Dict[str, Any]) -> P:
        """
        Sets the value of `metric_kwds`.
        """
        return self._set_params(metric_kwds=value)

    def getNEpochs(self: P) -> int:
        """
        Gets the value of `n_epochs`.
        """
        return self.getOrDefault("n_epochs")

    def setNEpochs(self: P, value: int) -> P:
        """
        Sets the value of `n_epochs`.
        """
        return self._set_params(n_epochs=value)

    def getLearningRate(self: P) -> float:
        """
        Gets the value of `learning_rate`.
        """
        return self.getOrDefault("learning_rate")

    def setLearningRate(self: P, value: float) -> P:
        """
        Sets the value of `learning_rate`.
        """
        return self._set_params(learning_rate=value)

    def getInit(self: P) -> str:
        """
        Gets the value of `init`.
        """
        return self.getOrDefault("init")

    def setInit(self: P, value: str) -> P:
        """
        Sets the value of `init`.
        """
        return self._set_params(init=value)

    def getMinDist(self: P) -> float:
        """
        Gets the value of `min_dist`.
        """
        return self.getOrDefault("min_dist")

    def setMinDist(self: P, value: float) -> P:
        """
        Sets the value of `min_dist`.
        """
        return self._set_params(min_dist=value)

    def getSpread(self: P) -> float:
        """
        Gets the value of `spread`.
        """
        return self.getOrDefault("spread")

    def setSpread(self: P, value: float) -> P:
        """
        Sets the value of `spread`.
        """
        return self._set_params(spread=value)

    def getSetOpMixRatio(self: P) -> float:
        """
        Gets the value of `set_op_mix_ratio`.
        """
        return self.getOrDefault("set_op_mix_ratio")

    def setSetOpMixRatio(self: P, value: float) -> P:
        """
        Sets the value of `set_op_mix_ratio`.
        """
        return self._set_params(set_op_mix_ratio=value)

    def getLocalConnectivity(self: P) -> float:
        """
        Gets the value of `local_connectivity`.
        """
        return self.getOrDefault("local_connectivity")

    def setLocalConnectivity(self: P, value: float) -> P:
        """
        Sets the value of `local_connectivity`.
        """
        return self._set_params(local_connectivity=value)

    def getRepulsionStrength(self: P) -> float:
        """
        Gets the value of `repulsion_strength`.
        """
        return self.getOrDefault("repulsion_strength")

    def setRepulsionStrength(self: P, value: float) -> P:
        """
        Sets the value of `repulsion_strength`.
        """
        return self._set_params(repulsion_strength=value)

    def getNegativeSampleRate(self: P) -> int:
        """
        Gets the value of `negative_sample_rate`.
        """
        return self.getOrDefault("negative_sample_rate")

    def setNegativeSampleRate(self: P, value: int) -> P:
        """
        Sets the value of `negative_sample_rate`.
        """
        return self._set_params(negative_sample_rate=value)

    def getTransformQueueSize(self: P) -> float:
        """
        Gets the value of `transform_queue_size`.
        """
        return self.getOrDefault("transform_queue_size")

    def setTransformQueueSize(self: P, value: float) -> P:
        """
        Sets the value of `transform_queue_size`.
        """
        return self._set_params(transform_queue_size=value)

    def getA(self: P) -> float:
        """
        Gets the value of `a`.
        """
        return self.getOrDefault("a")

    def setA(self: P, value: float) -> P:
        """
        Sets the value of `a`.
        """
        return self._set_params(a=value)

    def getB(self: P) -> float:
        """
        Gets the value of `b`.
        """
        return self.getOrDefault("b")

    def setB(self: P, value: float) -> P:
        """
        Sets the value of `b`.
        """
        return self._set_params(b=value)

    def getPrecomputedKNN(self: P) -> List[List[float]]:
        """
        Gets the value of `precomputed_knn`.
        """
        return self.getOrDefault("precomputed_knn")

    def setPrecomputedKNN(self: P, value: List[List[float]]) -> P:
        """
        Sets the value of `precomputed_knn`.
        """
        return self._set_params(precomputed_knn=value)

    def getRandomState(self: P) -> int:
        """
        Gets the value of `random_state`.
        """
        return self.getOrDefault("random_state")

    def setRandomState(self: P, value: int) -> P:
        """
        Sets the value of `random_state`.
        """
        return self._set_params(random_state=value)

    def getBuildAlgo(self: P) -> str:
        """
        Gets the value of `build_algo`.
        """
        return self.getOrDefault("build_algo")

    def setBuildAlgo(self: P, value: str) -> P:
        """
        Sets the value of `build_algo`.
        """
        return self._set_params(build_algo=value)

    def getBuildKwds(self: P) -> Optional[Dict[str, Any]]:
        """
        Gets the value of `build_kwds`.
        """
        return self.getOrDefault("build_kwds")

    def setBuildKwds(self: P, value: Dict[str, Any]) -> P:
        """
        Sets the value of `build_kwds`.
        """
        return self._set_params(build_kwds=value)

    def getSampleFraction(self: P) -> float:
        """
        Gets the value of `sample_fraction`.
        """
        return self.getOrDefault("sample_fraction")

    def setSampleFraction(self: P, value: float) -> P:
        """
        Sets the value of `sample_fraction`.
        """
        return self._set_params(sample_fraction=value)

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
            self._set_params(featuresCol=value)
        else:
            self._set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self._set_params(featuresCols=value)

    def setLabelCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self._set_params(labelCol=value)

    def getOutputCol(self: P) -> str:
        """
        Gets the value of :py:attr:`outputCol`. Contains the embeddings of the input data.
        """
        return self.getOrDefault("outputCol")

    def setOutputCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`outputCol`. Contains the embeddings of the input data.
        """
        return self._set_params(outputCol=value)


class UMAP(UMAPClass, _CumlEstimatorSupervised, _UMAPCumlParams):
    """
    Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique
    used for low-dimensional data visualization and general non-linear dimension reduction.
    The algorithm finds a low dimensional embedding of the data that approximates an underlying manifold.
    The fit() method constructs a KNN-graph representation of an input dataset and then optimizes a
    low dimensional embedding, and is performed on a single node. The transform() method transforms an input dataset
    into the optimized embedding space, and is performed distributedly.

    Parameters
    ----------
    n_neighbors : float (optional, default=15)
        The size of local neighborhood (in terms of number of neighboring sample points) used for
        manifold approximation. Larger values result in more global views of the manifold, while
        smaller values result in more local data being preserved. In general values should be in the range 2 to 100.

    n_components : int (optional, default=2)
        The dimension of the space to embed into. This defaults to 2 to provide easy visualization,
        but can reasonably be set to any integer value in the range 2 to 100.

    metric : str (optional, default='euclidean')
        Distance metric to use. Supported distances are ['l1', 'cityblock', 'taxicab', 'manhattan', 'euclidean',
        'l2', 'sqeuclidean', 'canberra', 'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']. Metrics that take arguments (such as minkowski) can have arguments passed via the
        metric_kwds dictionary.

    metric_kwds : dict (optional, default=None)
        Additional keyword arguments for the metric function. If the metric function takes additional arguments,
        they should be passed in this dictionary.

    n_epochs : int (optional, default=None)
        The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result
        in more accurate embeddings. If None is specified a value will be selected based on the size of the input dataset
        (200 for large datasets, 500 for small).

    learning_rate : float (optional, default=1.0)
        The initial learning rate for the embedding optimization.

    init : str (optional, default='spectral')
        How to initialize the low dimensional embedding. Options are:
          'spectral': use a spectral embedding of the fuzzy 1-skeleton
          'random': assign initial embedding positions at random.

    min_dist : float (optional, default=0.1)
        The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped
        embedding where nearby points on the manifold are drawn closer together, while larger values will result in a
        more even dispersal of points. The value should be set relative to the ``spread`` value, which determines the
        scale at which embedded points will be spread out.

    spread : float (optional, default=1.0)
        The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped
        the embedded points are.

    set_op_mix_ratio : float (optional, default=1.0)
        Interpolate between (fuzzy) union and intersection as the set operation used to combine local fuzzy simplicial
        sets to obtain a global fuzzy simplicial sets. Both fuzzy set operations use the product t-norm. The value of
        this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a
        pure fuzzy intersection.

    local_connectivity : int (optional, default=1)
        The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected
        at a local level. The higher this value the more connected the manifold becomes locally. In practice this should
        be not more than the local intrinsic dimension of the manifold.

    repulsion_strength : float (optional, default=1.0)
        Weighting applied to negative samples in low dimensional embedding optimization. Values higher than one will
        result in greater weight being given to negative samples.

    negative_sample_rate : int (optional, default=5)
        The number of negative samples to select per positive sample in the optimization process. Increasing this value
        will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.

    transform_queue_size : float (optional, default=4.0)
        For transform operations (embedding new points using a trained model), this will control how aggressively to
        search for nearest neighbors. Larger values will result in slower performance but more accurate nearest neighbor
        evaluation.

    a : float (optional, default=None)
        More specific parameters controlling the embedding. If None these values are set automatically as determined
        by ``min_dist`` and ``spread``.

    b : float (optional, default=None)
        More specific parameters controlling the embedding. If None these values are set automatically as determined
        by ``min_dist`` and ``spread``.

    precomputed_knn : array / sparse array / tuple - device or host (optional, default=None)
        Either one of a tuple (indices, distances) of arrays of shape (n_samples, n_neighbors), a pairwise distances
        dense array of shape (n_samples, n_samples) or a KNN graph sparse array (preferably CSR/COO). This feature
        allows the precomputation of the KNN outside of UMAP and also allows the use of a custom distance function.
        This function should match the metric used to train the UMAP embeedings.

    random_state : int, RandomState instance (optional, default=None)
        The seed used by the random number generator during embedding initialization and during sampling used by the
        optimizer. Unfortunately, achieving a high amount of parallelism during the optimization stage often comes at
        the expense of determinism, since many floating-point additions are being made in parallel without a deterministic
        ordering. This causes slightly different results across training sessions, even when the same seed is used for
        random number generation. Setting a random_state will enable consistency of trained embeddings, allowing for
        reproducible results to 3 digits of precision, but will do so at the expense of training time and memory usage.

    verbose :
        Logging level.
            * ``0`` - Disables all log messages.
            * ``1`` - Enables only critical messages.
            * ``2`` - Enables all messages up to and including errors.
            * ``3`` - Enables all messages up to and including warnings.
            * ``4 or False`` - Enables all messages up to and including information messages.
            * ``5 or True`` - Enables all messages up to and including debug messages.
            * ``6`` - Enables all messages up to and including trace messages.

    build_algo : str (optional, default='auto')
        How to build the knn graph. Supported build algorithms are ['auto', 'brute_force_knn', 'nn_descent']. 'auto' chooses
        to run with brute force knn if number of data rows is smaller than or equal to 50K. Otherwise, runs with nn descent.

    build_kwds : dict (optional, default=None)
        Build algorithm argument {'nnd_graph_degree': 64, 'nnd_intermediate_graph_degree': 128, 'nnd_max_iterations': 20,
        'nnd_termination_threshold': 0.0001, 'nnd_return_distances': True, 'nnd_n_clusters': 1} Note that nnd_n_clusters > 1
        will result in batch-building with NN Descent.

    sample_fraction : float (optional, default=1.0)
        The fraction of the dataset to be used for fitting the model. Since fitting is done on a single node, very large
        datasets must be subsampled to fit within the node's memory and execute in a reasonable time. Smaller fractions
        will result in faster training, but may result in sub-optimal embeddings.

    featuresCol: str or List[str]
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.

    labelCol: str (optional)
        The name of the column that contains labels. If provided, supervised fitting will be performed, where labels
        will be taken into account when optimizing the embedding.

    outputCol: str (optional)
        The name of the column that contains embeddings. If not provided, the default name of "embedding" will be used.

    num_workers:
        Number of cuML workers, where each cuML worker corresponds to one Spark task
        running on one GPU. If not set, spark-rapids-ml tries to infer the number of
        cuML workers (i.e. GPUs in cluster) from the Spark environment.

    Examples
    --------
    >>> from spark_rapids_ml.umap import UMAP
    >>> from cuml.datasets import make_blobs
    >>> import cupy as cp

    >>> X, _ = make_blobs(500, 5, centers=42, cluster_std=0.1, dtype=np.float32, random_state=10)
    >>> feature_cols = [f"c{i}" for i in range(X.shape[1])]
    >>> schema = [f"{c} {"float"}" for c in feature_cols]
    >>> df = spark.createDataFrame(X.tolist(), ",".join(schema))
    >>> df = df.withColumn("features", array(*feature_cols)).drop(*feature_cols)
    >>> df.show(10, False)

    +--------------------------------------------------------+
    |features                                                |
    +--------------------------------------------------------+
    |[1.5578103, -9.300072, 9.220654, 4.5838223, -3.2613218] |
    |[9.295866, 1.3326015, -4.6483326, 4.43685, 6.906736]    |
    |[1.1148645, 0.9800974, -9.67569, -8.020592, -3.748023]  |
    |[-4.6454153, -8.095899, -4.9839406, 7.954683, -8.15784] |
    |[-6.5075264, -5.538241, -6.740191, 3.0490158, 4.1693997]|
    |[7.9449835, 4.142317, 6.207676, 3.202615, 7.1319785]    |
    |[-0.3837125, 6.826891, -4.35618, -9.582829, -1.5456663] |
    |[2.5012932, 4.2080708, 3.5172815, 2.5741744, -6.291008] |
    |[9.317718, 1.3419528, -4.832837, 4.5362573, 6.9357944]  |
    |[-6.65039, -5.438729, -6.858565, 2.9733503, 3.99863]    |
    +--------------------------------------------------------+

    only showing top 10 rows

    >>> umap_estimator = UMAP(sample_fraction=0.5, num_workers=3).setFeaturesCol("features")
    >>> umap_model = umap_estimator.fit(df)
    >>> output = umap_model.transform(df).toPandas()
    >>> embedding = cp.asarray(output["embedding"].to_list())
    >>> print("First 10 embeddings:")
    >>> print(embedding[:10])

    First 10 embeddings:
    [[  5.378397    6.504756 ]
    [ 12.531521   13.946098 ]
    [ 11.990916    6.049594 ]
    [-14.175631    7.4849815]
    [  7.065363  -16.75355  ]
    [  1.8876278   1.0889664]
    [  0.6557462  17.965862 ]
    [-16.220764   -6.4817486]
    [ 12.476492   13.80965  ]
    [  6.823325  -16.71719  ]]

    """

    @pyspark.keyword_only
    def __init__(
        self,
        *,
        n_neighbors: Optional[float] = 15,
        n_components: Optional[int] = 15,
        metric: str = "euclidean",
        metric_kwds: Optional[Dict[str, Any]] = None,
        n_epochs: Optional[int] = None,
        learning_rate: Optional[float] = 1.0,
        init: Optional[str] = "spectral",
        min_dist: Optional[float] = 0.1,
        spread: Optional[float] = 1.0,
        set_op_mix_ratio: Optional[float] = 1.0,
        local_connectivity: Optional[float] = 1.0,
        repulsion_strength: Optional[float] = 1.0,
        negative_sample_rate: Optional[int] = 5,
        transform_queue_size: Optional[float] = 1.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        precomputed_knn: Optional[List[List[float]]] = None,
        random_state: Optional[int] = None,
        build_algo: Optional[str] = "auto",
        build_kwds: Optional[Dict[str, Any]] = None,
        sample_fraction: Optional[float] = 1.0,
        featuresCol: Optional[Union[str, List[str]]] = None,
        labelCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        num_workers: Optional[int] = None,
        enable_sparse_data_optim: Optional[
            bool
        ] = None,  # will enable SparseVector inputs if first row is sparse (for any metric).
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not self._input_kwargs.get("float32_inputs", True):
            get_logger(self.__class__).warning(
                "This estimator does not support double precision inputs. Setting float32_inputs to False will be ignored."
            )
            self._input_kwargs.pop("float32_inputs")
        self._set_params(**self._input_kwargs)
        max_records_per_batch_str = _get_spark_session().conf.get(
            "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
        )
        assert max_records_per_batch_str is not None
        self.max_records_per_batch = int(max_records_per_batch_str)

    def _create_pyspark_model(self, result: Row) -> _CumlModel:
        raise NotImplementedError("UMAP does not support model creation from Row")

    def _fit(self, dataset: DataFrame) -> "UMAPModel":
        if self.getSampleFraction() < 1.0:
            data_subset = dataset.sample(
                withReplacement=False,
                fraction=self.getSampleFraction(),
                seed=self.cuml_params["random_state"],
            )
        else:
            data_subset = dataset

        input_num_workers = self.num_workers
        # Force to single partition, single worker
        self._num_workers = 1
        if data_subset.rdd.getNumPartitions() != 1:
            data_subset = data_subset.coalesce(1)

        df_output = self._call_cuml_fit_func_dataframe(
            dataset=data_subset,
            partially_collect=False,
            paramMaps=None,
        )

        pdf_output: PandasDataFrame = df_output.toPandas()

        if self._sparse_fit:
            embeddings = np.array(
                list(
                    pd.concat(
                        [pd.Series(x) for x in pdf_output["embedding_"]],
                        ignore_index=True,
                    )
                ),
                dtype=np.float32,
            )
            pdf_output["raw_data_"] = pdf_output.apply(
                lambda row: scipy.sparse.csr_matrix(
                    (row["data"], row["indices"], row["indptr"]),
                    shape=row["shape"],
                ).astype(np.float32),
                axis=1,
            )
            raw_data = scipy.sparse.vstack(pdf_output["raw_data_"], format="csr")
        else:
            embeddings = np.vstack(pdf_output["embedding_"]).astype(np.float32)
            raw_data = np.vstack(pdf_output["raw_data_"]).astype(np.float32)  # type: ignore

        del pdf_output

        model = UMAPModel(
            embedding_=embeddings,
            raw_data_=raw_data,
            sparse_fit=self._sparse_fit,
            n_cols=self._n_cols,
            dtype="float32",  # UMAP only supports float
        )

        model._num_workers = input_num_workers

        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore

        return model

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
        array_order = self._fit_array_order()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.manifold import UMAP as CumlUMAP

            umap_object = CumlUMAP(
                **params[param_alias.cuml_init],
            )

            df_list = [x for (x, _, _) in dfs]
            if isinstance(df_list[0], pd.DataFrame):
                concated = pd.concat(df_list)
            else:
                concated = _concat_and_free(df_list, order=array_order)

            if dfs[0][1] is not None:
                # If labels are provided, call supervised fit
                label_list = [x for (_, x, _) in dfs]
                if isinstance(label_list[0], pd.DataFrame):
                    labels = pd.concat(label_list)
                else:
                    labels = _concat_and_free(label_list, order=array_order)
                umap_model = umap_object.fit(concated, y=labels)
            else:
                # Call unsupervised fit
                umap_model = umap_object.fit(concated)

            embedding = umap_model.embedding_
            del umap_model

            return {"embedding": embedding, "raw_data": concated}

        return _cuml_fit

    def _call_cuml_fit_func_dataframe(
        self,
        dataset: DataFrame,
        partially_collect: bool = True,
        paramMaps: Optional[Sequence["ParamMap"]] = None,
    ) -> DataFrame:
        """
        Fits a model to the input dataset. This replaces _call_cuml_fit_func() to omit barrier stages and return a dataframe
        rather than an RDD.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            input dataset

        Returns
        -------
        output : :py:class:`pyspark.sql.DataFrame`
            fitted model attributes
        """

        cls = self.__class__

        select_cols, multi_col_names, dimension, _ = self._pre_process_data(dataset)
        self._n_cols = dimension

        dataset = dataset.select(*select_cols)

        is_local = _is_local(_get_spark_session().sparkContext)

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )
        if cuda_managed_mem_enabled:
            get_logger(cls).info("CUDA managed memory enabled.")

        # parameters passed to subclass
        params: Dict[str, Any] = {
            param_alias.cuml_init: self.cuml_params,
        }

        params[param_alias.fit_multiple_params] = []

        cuml_fit_func = self._get_cuml_fit_func(dataset, None)

        array_order = self._fit_array_order()

        cuml_verbose = self.cuml_params.get("verbose", False)

        use_sparse_array = _use_sparse_in_cuml(dataset)
        self._sparse_fit = use_sparse_array  # param stored internally by cuml model
        if self.cuml_params.get("metric") == "jaccard" and not use_sparse_array:
            raise ValueError("Metric 'jaccard' not supported for dense inputs.")

        chunk_size = self.max_records_per_batch

        def _train_udf(pdf_iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            from pyspark import TaskContext

            logger = get_logger(cls)
            logger.info("Initializing cuml context")

            import cupy as cp
            import cupyx

            if cuda_managed_mem_enabled:
                import rmm
                from rmm.allocators.cupy import rmm_cupy_allocator

                rmm.reinitialize(managed_memory=True)
                cp.cuda.set_allocator(rmm_cupy_allocator)

            _CumlCommon._initialize_cuml_logging(cuml_verbose)

            context = TaskContext.get()

            # set gpu device
            _CumlCommon._set_gpu_device(context, is_local)

            # handle the input
            # inputs = [(X, Optional(y)), (X, Optional(y))]
            logger.info("Loading data into python worker memory")
            inputs: List[Any] = []
            sizes: List[int] = []

            for pdf in pdf_iter:
                sizes.append(pdf.shape[0])
                if multi_col_names:
                    features = np.array(pdf[multi_col_names], order=array_order)
                elif use_sparse_array:
                    # sparse vector input
                    features = _read_csr_matrix_from_unwrapped_spark_vec(pdf)
                else:
                    # dense input
                    features = np.array(list(pdf[alias.data]), order=array_order)
                if cuda_managed_mem_enabled and not use_sparse_array:
                    features = cp.array(features)

                label = pdf[alias.label] if alias.label in pdf.columns else None
                row_number = (
                    pdf[alias.row_number] if alias.row_number in pdf.columns else None
                )
                inputs.append((features, label, row_number))

            if cuda_managed_mem_enabled and use_sparse_array:
                concated_nnz = sum(triplet[0].nnz for triplet in inputs)  # type: ignore
                if concated_nnz > np.iinfo(np.int32).max:
                    logger.warn(
                        f"The number of non-zero values of a partition exceeds the int32 index dtype. \
                        cupyx csr_matrix currently does not support int64 indices (https://github.com/cupy/cupy/issues/3513); \
                        keeping as scipy csr_matrix to avoid overflow."
                    )
                else:
                    inputs = [
                        (cupyx.scipy.sparse.csr_matrix(row[0]), row[1], row[2])
                        for row in inputs
                    ]

            # call the cuml fit function
            # *note*: cuml_fit_func may delete components of inputs to free
            # memory.  do not rely on inputs after this call.
            embedding, raw_data = cuml_fit_func(inputs, params).values()

            logger.info("Cuml fit complete")

            num_sections = (len(embedding) + chunk_size - 1) // chunk_size

            for i in range(num_sections):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(embedding))
                if use_sparse_array:
                    csr_chunk = raw_data[start:end]
                    indices = csr_chunk.indices
                    indptr = csr_chunk.indptr
                    data = csr_chunk.data
                    yield pd.DataFrame(
                        data=[
                            {
                                "embedding_": embedding[start:end].tolist(),
                                "indices": indices.tolist(),
                                "indptr": indptr.tolist(),
                                "data": data.tolist(),
                                "shape": [end - start, dimension],
                            }
                        ]
                    )
                else:
                    yield pd.DataFrame(
                        {
                            "embedding_": embedding[start:end].tolist(),
                            "raw_data_": raw_data[start:end].tolist(),
                        }
                    )

        output_df = dataset.mapInPandas(_train_udf, schema=self._out_schema())

        return output_df

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self) -> Union[StructType, str]:
        if self._sparse_fit:
            return StructType(
                [
                    StructField(
                        "embedding_",
                        ArrayType(ArrayType(FloatType(), False), False),
                        False,
                    ),
                    StructField("indices", ArrayType(IntegerType(), False), False),
                    StructField("indptr", ArrayType(IntegerType(), False), False),
                    StructField("data", ArrayType(FloatType(), False), False),
                    StructField("shape", ArrayType(IntegerType(), False), False),
                ]
            )
        else:
            return StructType(
                [
                    StructField("embedding_", ArrayType(FloatType()), False),
                    StructField("raw_data_", ArrayType(FloatType()), False),
                ]
            )

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[
        List[Column], Optional[List[str]], int, Union[Type[FloatType], Type[DoubleType]]
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = super(
            _CumlEstimatorSupervised, self
        )._pre_process_data(dataset)

        if self.getLabelCol() in dataset.schema.names:
            select_cols.append(self._pre_process_label(dataset, feature_type))

        return select_cols, multi_col_names, dimension, feature_type


class UMAPModel(_CumlModelWithColumns, UMAPClass, _UMAPCumlParams):
    def __init__(
        self,
        embedding_: np.ndarray,
        raw_data_: Union[
            np.ndarray,
            scipy.sparse.csr_matrix,
        ],
        sparse_fit: bool,
        n_cols: int,
        dtype: str,
    ) -> None:
        super(UMAPModel, self).__init__(
            embedding_=embedding_,
            raw_data_=raw_data_,
            sparse_fit=sparse_fit,
            n_cols=n_cols,
            dtype=dtype,
        )
        self.embedding_ = embedding_
        self.raw_data_ = raw_data_
        self._sparse_fit = sparse_fit  # If true, raw data is a sparse CSR matrix
        self.BROADCAST_LIMIT = 8 << 30  # Spark broadcast limit: 8GiB

    @property
    def embedding(self) -> np.ndarray:
        """
        Returns the model embeddings.
        """
        return (
            self.embedding_
        )  # TBD: return a more Spark-like object, e.g. DenseMatrix?

    @property
    def rawData(self) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Returns the raw data used to fit the model. If the input data was sparse, this will be a scipy csr matrix.
        """
        return (
            self.raw_data_
        )  # TBD: return a more Spark-like object, e.g. DenseMatrix or SparseMatrix?

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        cuml_alg_params = self.cuml_params
        sparse_fit = self._sparse_fit
        n_cols = self.n_cols

        def _chunk_and_broadcast(
            sc: pyspark.SparkContext,
            arr: np.ndarray,
            BROADCAST_LIMIT: int,
        ) -> List[pyspark.broadcast.Broadcast]:
            """
            Broadcast the input array, chunking it into smaller arrays if it exceeds the broadcast limit.
            """
            if arr.nbytes < BROADCAST_LIMIT:
                return [sc.broadcast(arr)]

            rows_per_chunk = BROADCAST_LIMIT // (arr.nbytes // arr.shape[0])
            if rows_per_chunk == 0:
                raise ValueError(
                    f"Array cannot be chunked into broadcastable pieces: \
                        single row exceeds broadcast limit ({BROADCAST_LIMIT} bytes)"
                )
            num_chunks = (arr.shape[0] + rows_per_chunk - 1) // rows_per_chunk
            return [
                sc.broadcast(arr[i * rows_per_chunk : (i + 1) * rows_per_chunk])
                for i in range(num_chunks)
            ]

        spark = _get_spark_session()
        broadcast_embeddings = _chunk_and_broadcast(
            spark.sparkContext, self.embedding_, self.BROADCAST_LIMIT
        )

        if isinstance(self.raw_data_, scipy.sparse.csr_matrix):
            broadcast_raw_data = {
                "indices": _chunk_and_broadcast(
                    spark.sparkContext, self.raw_data_.indices, self.BROADCAST_LIMIT
                ),
                "indptr": _chunk_and_broadcast(
                    spark.sparkContext, self.raw_data_.indptr, self.BROADCAST_LIMIT
                ),
                "data": _chunk_and_broadcast(
                    spark.sparkContext, self.raw_data_.data, self.BROADCAST_LIMIT
                ),
            }  # NOTE: CSR chunks are not independently meaningful; do not use until recombined.
        else:
            broadcast_raw_data = _chunk_and_broadcast(
                spark.sparkContext, self.raw_data_, self.BROADCAST_LIMIT
            )  # type: ignore

        def _construct_umap() -> CumlT:
            import cupy as cp
            from cuml.common import SparseCumlArray
            from cuml.common.sparse_utils import is_sparse
            from cuml.manifold import UMAP as CumlUMAP

            from .utils import cudf_to_cuml_array

            nonlocal broadcast_embeddings, broadcast_raw_data

            assert isinstance(broadcast_embeddings, list)
            embedding = (
                broadcast_embeddings[0].value
                if len(broadcast_embeddings) == 1
                else np.concatenate([chunk.value for chunk in broadcast_embeddings])
            )

            if sparse_fit:
                if not isinstance(broadcast_raw_data, dict):
                    raise ValueError("Expected raw data as a CSR dict for sparse fit.")
                indices = np.concatenate(
                    [chunk.value for chunk in broadcast_raw_data["indices"]]
                )
                indptr = np.concatenate(
                    [chunk.value for chunk in broadcast_raw_data["indptr"]]
                )
                data = np.concatenate(
                    [chunk.value for chunk in broadcast_raw_data["data"]]
                )
                raw_data = scipy.sparse.csr_matrix(
                    (data, indices, indptr), shape=(len(indptr) - 1, n_cols)
                )
            else:
                if not isinstance(broadcast_raw_data, list):
                    raise ValueError(
                        "Expected raw data as list (of lists) for dense fit."
                    )
                raw_data = (
                    broadcast_raw_data[0].value
                    if len(broadcast_raw_data) == 1
                    else np.concatenate([chunk.value for chunk in broadcast_raw_data])
                )

            del broadcast_embeddings
            del broadcast_raw_data

            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
                raw_data = raw_data.astype(np.float32)

            if is_sparse(raw_data):
                raw_data_cuml = SparseCumlArray(
                    raw_data,
                )
            else:
                raw_data_cuml = cudf_to_cuml_array(
                    raw_data,
                    order="C",
                )

            internal_model = CumlUMAP(**cuml_alg_params)
            internal_model.embedding_ = cp.array(embedding).data
            internal_model._raw_data = raw_data_cuml
            internal_model.sparse_fit = sparse_fit

            return internal_model

        def _transform_internal(
            umap: CumlT,
            df: Union[pd.DataFrame, np.ndarray, scipy.sparse._csr.csr_matrix],
        ) -> pd.DataFrame:

            embedding = umap.transform(df)

            # Input is either numpy array or pandas dataframe
            emb_list = [
                (
                    embedding[i, :]
                    if isinstance(embedding, np.ndarray)
                    else embedding.iloc[i, :]
                )
                for i in range(embedding.shape[0])
            ]

            return pd.Series(emb_list)

        return _construct_umap, _transform_internal, None

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        assert self.dtype is not None
        pyspark_type = dtype_to_pyspark_type(self.dtype)
        return f"array<{pyspark_type}>"

    def write(self) -> MLWriter:
        return _CumlModelWriterNumpy(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlModelReaderNumpy(cls)


class _CumlModelWriterNumpy(_CumlModelWriter):
    """
    Override parent writer to save numpy objects of _CumlModel to the file
    """

    def saveImpl(self, path: str) -> None:
        DefaultParamsWriter.saveMetadata(
            self.instance,
            path,
            self.sc,
            extraMetadata={
                "_cuml_params": self.instance._cuml_params,
                "_num_workers": self.instance._num_workers,
                "_float32_inputs": self.instance._float32_inputs,
            },
        )
        data_path = os.path.join(path, "data")
        model_attributes = self.instance._get_model_attributes()

        if not os.path.exists(data_path):
            os.makedirs(data_path)
        assert model_attributes is not None

        for key in ["embedding_", "raw_data_"]:
            array = model_attributes[key]
            if isinstance(array, scipy.sparse.csr_matrix):
                npz_path = os.path.join(data_path, f"{key}csr_.npz")
                scipy.sparse.save_npz(npz_path, array)
            else:
                npz_path = os.path.join(data_path, f"{key}.npz")
                np.savez_compressed(npz_path, array)
            model_attributes[key] = npz_path

        metadata_file_path = os.path.join(data_path, "metadata.json")
        model_attributes_str = json.dumps(model_attributes)
        self.sc.parallelize([model_attributes_str], 1).saveAsTextFile(
            metadata_file_path
        )


class _CumlModelReaderNumpy(_CumlModelReader):
    """
    Override parent reader to instantiate numpy objects of _CumlModel from file
    """

    def load(self, path: str) -> "_CumlEstimator":
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        data_path = os.path.join(path, "data")
        metadata_file_path = os.path.join(data_path, "metadata.json")

        model_attr_str = self.sc.textFile(metadata_file_path).collect()[0]
        model_attr_dict = json.loads(model_attr_str)

        for key in ["embedding_", "raw_data_"]:
            npz_path = model_attr_dict[key]
            if npz_path.endswith("csr_.npz"):
                model_attr_dict[key] = scipy.sparse.load_npz(npz_path)
            else:
                with np.load(npz_path) as data:
                    model_attr_dict[key] = data["arr_0"]

        instance = self.model_cls(**model_attr_dict)
        DefaultParamsReader.getAndSetParams(instance, metadata)
        instance._cuml_params = metadata["_cuml_params"]
        instance._num_workers = metadata["_num_workers"]
        instance._float32_inputs = metadata["_float32_inputs"]
        return instance

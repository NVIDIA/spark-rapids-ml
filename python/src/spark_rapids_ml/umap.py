#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
from pyspark import RDD
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol
from pyspark.sql import Column, DataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    Row,
    StructField,
    StructType,
)

from spark_rapids_ml.core import FitInputType, _CumlModel

from .common.cuml_context import CumlContext
from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlCommon,
    _CumlEstimatorSupervised,
    _CumlModel,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
    transform_evaluate,
)
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .utils import (
    _ArrayOrder,
    _concat_and_free,
    _get_spark_session,
    _is_local,
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
        }


class _UMAPCumlParams(_CumlParams, HasFeaturesCol, HasFeaturesCols, HasLabelCol):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault()

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

    def setLabelCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self.set_params(labelCol=value)


class UMAP(UMAPClass, _CumlEstimatorSupervised, _UMAPCumlParams):

    """
    Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique
    used for low-dimensional data visualization and general non-linear dimension reduction.
    The algorithm finds a low dimensional embedding of the data that approximates an underlying manifold.
    The fit() method constructs a KNN-graph representation of an input dataset and then optimizes a
    low dimensional embedding, and is performed locally. The transform() method transforms an input dataset
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
        If set to a non-zero value, this will ensure reproducible results during fit(). Note that transform() is
        inherently stochastic and may yield slightly varied embedding results.

    verbose :
        Logging level.
            * ``0`` - Disables all log messages.
            * ``1`` - Enables only critical messages.
            * ``2`` - Enables all messages up to and including errors.
            * ``3`` - Enables all messages up to and including warnings.
            * ``4 or False`` - Enables all messages up to and including information messages.
            * ``5 or True`` - Enables all messages up to and including debug messages.
            * ``6`` - Enables all messages up to and including trace messages.

    sample_fraction : float (optional, default=1.0)
        The fraction of the dataset to be used for fitting the model. Since fitting is done locally, very large datasets
        must be subsampled to fit within local memory and execute in a reasonable time. Smaller fractions will result in
        faster training, but may result in sub-optimal embeddings.

    featuresCol: str
        The name of the column that contains input vectors. featuresCol should be set when input vectors are stored
        in a single column of a dataframe.

    featuresCols: List[str]
        The names of the columns that contain input vectors. featuresCols should be set when input vectors are stored
        in multiple columns of a dataframe.

    labelCol: str (optional)
        The name of the column that contains labels. If provided, supervised fitting will be performed, where labels
        will be taken into account when optimizing the embedding.

    Examples
    --------
    >>> from spark_rapids_ml.umap import UMAP
    >>> from cuml.datasets import make_blobs
    >>> X, _ = make_blobs(1000, 10, centers=42, cluster_std=0.1, dtype=np.float32, random_state=10)
    >>> df = spark.createDataFrame(X, ["features"])
    >>> df.show()
    # TODO: show DF
    >>> local_model = UMAP(sample_fraction=0.5)
    >>> local_model.setFeaturesCol("features")
    >>> distributed_model = local_model.fit(df)
    >>> embeddings = distributed_model.transform(df)
    >>> embeddings.show()
    # TODO: show output DF

    """

    def __init__(self, sample_fraction: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)
        self.sample_fraction = sample_fraction

    def _create_pyspark_model(self, result: Row) -> _CumlModel:
        raise NotImplementedError("UMAP does not support model creation from Row")

    def _fit(self, dataset: DataFrame) -> "UMAPModel":
        if self.sample_fraction < 1.0:
            data_subset = dataset.sample(
                withReplacement=False,
                fraction=self.sample_fraction,
                seed=self.cuml_params["random_state"],
            )
        else:
            data_subset = dataset

        input_num_workers = self.num_workers
        # Force to single partition, single worker
        self._num_workers = 1
        if data_subset.rdd.getNumPartitions() != 1:
            data_subset = data_subset.repartition(1)

        pipelined_rdd = self._call_cuml_fit_func(
            dataset=data_subset,
            partially_collect=False,
            paramMaps=None,
        )
        rows = pipelined_rdd.collect()
        # Collect and concatenate row-by-row fit results
        embeddings = [row["embedding_"] for row in rows]
        raw_data = [row["raw_data_"] for row in rows]
        del rows

        model = UMAPModel(
            embedding_=embeddings,
            raw_data_=raw_data,
            n_cols=len(raw_data[0]),
            dtype=str(np.array(raw_data[0][0]).dtype),
        )
        model._num_workers = input_num_workers

        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore

        return model

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_fit_func(  # type: ignore
        self, dataset: DataFrame
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
        """
        This class overrides the parent function with a different return signature.
        """
        pass

    def _get_cuml_fit_generator_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]], Generator[Dict[str, Any], None, None]
    ]:
        array_order = self._fit_array_order()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Generator[Dict[str, Any], None, None]:
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
                local_model = umap_object.fit(concated, y=labels)
            else:
                # Call unsupervised fit
                local_model = umap_object.fit(concated)

            embedding = local_model.embedding_
            del local_model

            for embedding, raw_data in zip(embedding, concated):
                yield pd.DataFrame(
                    data=[
                        {
                            "embedding_": embedding.tolist(),
                            "raw_data_": raw_data.tolist(),
                        }
                    ]
                )

        return _cuml_fit

    def _fit_return_by_row(self) -> bool:
        return True

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "embedding_",
                    ArrayType(FloatType(), False),
                    False,
                ),
                StructField(
                    "raw_data_",
                    ArrayType(FloatType(), False),
                    False,
                ),
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


class UMAPModel(_CumlModel, UMAPClass, _UMAPCumlParams):
    def __init__(
        self,
        embedding_: List[List[float]],
        raw_data_: List[List[float]],
        n_cols: int,
        dtype: str,
    ) -> None:
        super(UMAPModel, self).__init__(
            embedding_=embedding_,
            raw_data_=raw_data_,
            n_cols=n_cols,
            dtype=dtype,
        )
        self.embedding_ = embedding_
        self.raw_data_ = raw_data_

    @property
    def embedding(self) -> List[List[float]]:
        return self.embedding_

    @property
    def raw_data(self) -> List[List[float]]:
        return self.raw_data_

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        def _construct_umap() -> CumlT:
            raise NotImplementedError("TODO")

        def _transform_internal(
            umap: CumlT,
            df: Union[pd.DataFrame, np.ndarray],
        ) -> pd.Series:
            raise NotImplementedError("TODO")

        return _construct_umap, _transform_internal, None

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        raise NotImplementedError("TODO")

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

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
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
import pyspark
from pandas import DataFrame as PandasDataFrame
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter, MLReader, MLWriter
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

from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlCommon,
    _CumlEstimator,
    _CumlEstimatorSupervised,
    _CumlModel,
    _CumlModelReader,
    _CumlModelWriter,
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


class _UMAPCumlParams(
    _CumlParams, HasFeaturesCol, HasFeaturesCols, HasLabelCol, HasOutputCol
):
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

    def getOutputCol(self) -> str:
        """
        Gets the value of :py:attr:`outputCol`. Contains the embeddings of the input data.
        """
        return self.getOrDefault("outputCol")

    def setOutputCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`outputCol`. Contains the embeddings of the input data.
        """
        return self.set_params(outputCol=value)


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

    sample_fraction : float (optional, default=1.0)
        The fraction of the dataset to be used for fitting the model. Since fitting is done on a single node, very large
        datasets must be subsampled to fit within the node's memory and execute in a reasonable time. Smaller fractions
        will result in faster training, but may result in sub-optimal embeddings.

    featuresCol: str
        The name of the column that contains input vectors. featuresCol should be set when input vectors are stored
        in a single column of a dataframe.

    featuresCols: List[str]
        The names of the columns that contain input vectors. featuresCols should be set when input vectors are stored
        in multiple columns of a dataframe.

    labelCol: str (optional)
        The name of the column that contains labels. If provided, supervised fitting will be performed, where labels
        will be taken into account when optimizing the embedding.

    outputCol: str (optional)
        The name of the column that contains embeddings. If not provided, the default name of "embedding" will be used.

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

    def __init__(self, sample_fraction: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)
        self.sample_fraction = sample_fraction
        self.maxRecordsPerBatch = 10000
        self.setOutputCol("embedding")

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
            data_subset = data_subset.coalesce(1)

        maxRecordsPerBatch_str = _get_spark_session().conf.get(
            "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
        )
        assert maxRecordsPerBatch_str is not None
        self.maxRecordsPerBatch = int(maxRecordsPerBatch_str)

        df_output = self._call_cuml_fit_func_dataframe(
            dataset=data_subset,
            partially_collect=False,
            paramMaps=None,
        )

        pdf_output: PandasDataFrame = df_output.toPandas()

        # Collect and concatenate row-by-row fit results
        embeddings = np.array(
            list(
                pd.concat(
                    [pd.Series(x) for x in pdf_output["embedding_"]], ignore_index=True
                )
            )
        )
        raw_data = np.array(
            list(
                pd.concat(
                    [pd.Series(x) for x in pdf_output["raw_data_"]], ignore_index=True
                )
            )
        )
        del pdf_output

        spark = _get_spark_session()
        broadcast_embeddings = spark.sparkContext.broadcast(embeddings)
        broadcast_raw_data = spark.sparkContext.broadcast(raw_data)

        model = UMAPModel(
            embedding_=broadcast_embeddings,
            raw_data_=broadcast_raw_data,
            n_cols=len(raw_data[0]),
            dtype=type(raw_data[0][0]).__name__,
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
        This class replaces the parent function with a different return signature. See fit_generator_func below.
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
                umap_model = umap_object.fit(concated, y=labels)
            else:
                # Call unsupervised fit
                umap_model = umap_object.fit(concated)

            embedding = umap_model.embedding_
            del umap_model

            chunkSize = self.maxRecordsPerBatch
            num_sections = (len(embedding) + chunkSize - 1) // chunkSize

            for i in range(num_sections):
                start = i * chunkSize
                end = min((i + 1) * chunkSize, len(embedding))

                yield pd.DataFrame(
                    data=[
                        {
                            "embedding_": embedding[start:end].tolist(),
                            "raw_data_": concated[start:end].tolist(),
                        }
                    ]
                )

        return _cuml_fit

    def _call_cuml_fit_func_dataframe(
        self,
        dataset: DataFrame,
        partially_collect: bool = True,
        paramMaps: Optional[Sequence["ParamMap"]] = None,
    ) -> DataFrame:
        """
        Fits a model to the input dataset. This overrides the parent function to omit barrier stages and return a dataframe rather than an RDD.

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

        select_cols, multi_col_names, _, _ = self._pre_process_data(dataset)

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

        cuml_fit_func = self._get_cuml_fit_generator_func(dataset, None)  # type: ignore

        array_order = self._fit_array_order()

        cuml_verbose = self.cuml_params.get("verbose", False)

        def _train_udf(pdf_iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            from pyspark import TaskContext

            logger = get_logger(cls)
            logger.info("Initializing cuml context")

            import cupy as cp

            if cuda_managed_mem_enabled:
                import rmm
                from rmm.allocators.cupy import rmm_cupy_allocator

                rmm.reinitialize(managed_memory=True)
                cp.cuda.set_allocator(rmm_cupy_allocator)

            _CumlCommon.initialize_cuml_logging(cuml_verbose)

            context = TaskContext.get()

            # set gpu device
            _CumlCommon.set_gpu_device(context, is_local)

            # handle the input
            # inputs = [(X, Optional(y)), (X, Optional(y))]
            logger.info("Loading data into python worker memory")
            inputs = []
            sizes = []
            for pdf in pdf_iter:
                sizes.append(pdf.shape[0])
                if multi_col_names:
                    features = np.array(pdf[multi_col_names], order=array_order)
                else:
                    features = np.array(list(pdf[alias.data]), order=array_order)
                # experiments indicate it is faster to convert to numpy array and then to cupy array than directly
                # invoking cupy array on the list
                if cuda_managed_mem_enabled:
                    features = cp.array(features)

                label = pdf[alias.label] if alias.label in pdf.columns else None
                row_number = (
                    pdf[alias.row_number] if alias.row_number in pdf.columns else None
                )
                inputs.append((features, label, row_number))

            # call the cuml fit function
            # *note*: cuml_fit_func may delete components of inputs to free
            # memory.  do not rely on inputs after this call.
            result = cuml_fit_func(inputs, params)
            logger.info("Cuml fit complete")

            for row in result:
                yield row

        output_df = dataset.mapInPandas(_train_udf, schema=self._out_schema())

        return output_df

    def _use_fit_generator(self) -> bool:
        return True

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "embedding_",
                    ArrayType(ArrayType(FloatType(), False), False),
                    False,
                ),
                StructField(
                    "raw_data_",
                    ArrayType(ArrayType(FloatType(), False), False),
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
        embedding_: Union[pyspark.broadcast.Broadcast, np.ndarray],
        raw_data_: Union[pyspark.broadcast.Broadcast, np.ndarray],
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
    def embedding(self) -> np.ndarray:
        if isinstance(self.embedding_, np.ndarray):
            return self.embedding_
        return self.embedding_.value

    @property
    def raw_data(self) -> np.ndarray:
        if isinstance(self.raw_data_, np.ndarray):
            return self.raw_data_
        return self.raw_data_.value

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        cuml_alg_params = self.cuml_params

        def _construct_umap() -> CumlT:
            import cupy as cp
            from cuml.common import SparseCumlArray
            from cuml.common.sparse_utils import is_sparse
            from cuml.manifold import UMAP as CumlUMAP

            from .utils import cudf_to_cuml_array

            embedding = self.embedding
            raw_data = self.raw_data
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
                raw_data = raw_data.astype(np.float32)

            if is_sparse(raw_data):
                raw_data_cuml = SparseCumlArray(raw_data, convert_format=False)
            else:
                raw_data_cuml = cudf_to_cuml_array(
                    raw_data,
                    order="C",
                )

            internal_model = CumlUMAP(**cuml_alg_params)
            internal_model.embedding_ = cp.array(embedding).data
            internal_model._raw_data = raw_data_cuml

            return internal_model

        def _transform_internal(
            umap: CumlT,
            df: Union[pd.DataFrame, np.ndarray],
        ) -> pd.Series:
            embedding = umap.transform(df)

            is_df_np = isinstance(df, np.ndarray)
            is_emb_np = isinstance(embedding, np.ndarray)

            # Input is either numpy array or pandas dataframe
            input_list = [
                df[i, :] if is_df_np else df.iloc[i, :] for i in range(df.shape[0])  # type: ignore
            ]
            emb_list = [
                embedding[i, :] if is_emb_np else embedding.iloc[i, :]
                for i in range(embedding.shape[0])
            ]

            result = pd.DataFrame(
                {
                    "features": input_list,
                    self.getOutputCol(): emb_list,
                }
            )

            return result

        return _construct_umap, _transform_internal, None

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (False, False)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        return StructType(
            [
                StructField("features", ArrayType(FloatType(), False), False),
                StructField(self.getOutputCol(), ArrayType(FloatType(), False), False),
            ]
        )

    def get_model_attributes(self) -> Optional[Dict[str, Any]]:
        """Override parent method to bring broadcast variables to driver before JSON serialization."""
        if not isinstance(self.embedding_, np.ndarray):
            self._model_attributes["embedding_"] = self.embedding_.value
        if not isinstance(self.raw_data_, np.ndarray):
            self._model_attributes["raw_data_"] = self.raw_data_.value
        return self._model_attributes

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
            },
        )
        data_path = os.path.join(path, "data")
        model_attributes = self.instance.get_model_attributes()

        if not os.path.exists(data_path):
            os.makedirs(data_path)
        assert model_attributes is not None
        for key, value in model_attributes.items():
            if isinstance(value, np.ndarray):
                array_path = os.path.join(data_path, f"{key}.npy")
                np.save(array_path, value)
                model_attributes[key] = array_path

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

        for key, value in model_attr_dict.items():
            if isinstance(value, str) and value.endswith(".npy"):
                model_attr_dict[key] = np.load(value)

        instance = self.model_cls(**model_attr_dict)
        DefaultParamsReader.getAndSetParams(instance, metadata)
        instance._cuml_params = metadata["_cuml_params"]
        instance._num_workers = metadata["_num_workers"]
        return instance

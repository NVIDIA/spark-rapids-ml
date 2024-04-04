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

from abc import ABCMeta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pyspark import keyword_only
from pyspark.ml.param.shared import HasFeaturesCol, Param, Params, TypeConverters
from pyspark.sql import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    Row,
    StructField,
    StructType,
)

from .core import (
    FitInputType,
    _ConstructFunc,
    _CumlCaller,
    _CumlEstimator,
    _CumlModel,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _read_csr_matrix_from_unwrapped_spark_vec,
    _TransformFunc,
    _use_sparse_in_cuml,
    alias,
    param_alias,
)
from .metrics import EvalMetricInfo
from .params import HasFeaturesCols, HasIDCol, P, _CumlClass, _CumlParams
from .utils import _ArrayOrder, _concat_and_free, _get_spark_session, get_logger

if TYPE_CHECKING:
    import cudf
    from pyspark.ml._typing import ParamMap


class DBSCANClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "verbose": False,
            "max_mbytes_per_batch": None,
            "calc_core_sample_indices": True,
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
            max_mbytes_per_batch=None,
            calc_core_sample_indices=True,
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
            f"The input will be modified temporarily when cosine distance is used and the restored input matrix might not match completely due to numerical rounding."
        ),
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

    calc_core_sample_indices = Param(
        Params._dummy(),
        "calc_core_sample_indices",
        (
            f"Indicates whether the indices of the core samples should be calculated."
            f"Setting this to False will avoid unnecessary kernel launches"
        ),
        typeConverter=TypeConverters.toBoolean,
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
        Sets the value of `idCol`. If not set, an id column will be added with column name `unique_id`. The id column is used to specify nearest neighbor vectors by associated id value.
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

    eps: float (default = 0.5)
        The maximum distance between 2 points such they reside in the same neighborhood.

    min_samples: int (default = 5)
        The number of samples in a neighborhood such that this group can be considered as
        an important core point (including the point itself).

    metric: {'euclidean', 'cosine'}, default = 'euclidean'
        The metric to use when calculating distances between points.
        Spark Rapids ML does not support the 'precomputed' mode from sklearn and cuML, please use those libraries instead
        The input will be modified temporarily when cosine distance is used and the restored input matrix might not match completely due to numerical rounding.

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

    calc_core_sample_indices(optional): boolean (default = True)
        Indicates whether the indices of the core samples should be calculated.
        Setting this to False will avoid unnecessary kernel launches

    idCol: str (default = 'unique_id')
        The internal unique id column name for label matching, will not reveal in the output.
        Need to be set to a name that does not conflict with an existing column name in the original input data.

    Examples
    ----------
    >>> from spark_rapids_ml.dbscan import DBSCAN
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
    >>> from spark_rapids_ml.dbscan import DBSCAN
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
        featuresCol: str = "features",
        predictionCol: str = "prediction",
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        max_mbytes_per_batch: Optional[int] = None,
        calc_core_sample_indices: bool = True,
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

        self.verbose = verbose

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

    def setMaxMbytesPerBatch(self: P, value: Optional[int]) -> P:
        return self._set_params(max_mbytes_per_batch=value)

    def getMaxMbytesPerBatch(self) -> Optional[int]:
        return self.getOrDefault("max_mbytes_per_batch")

    def setCalcCoreSampleIndices(self: P, value: bool) -> P:
        return self._set_params(calc_core_sample_indices=value)

    def getCalcCoreSampleIndices(self) -> bool:
        return self.getOrDefault("calc_core_sample_indices")

    def _fit(self, dataset: DataFrame) -> _CumlModel:
        if self.getMetric() == "precomputed":
            raise ValueError(
                "Spark Rapids ML does not support the 'precomputed' mode from sklearn and cuML, please use those libraries instead"
            )

        # Create parameter-copied model without assess the input dataframe
        # All information will be retrieved from Model and transform
        model = DBSCANModel(verbose=self.verbose, n_cols=0, dtype="")

        model._num_workers = self.num_workers
        self._copyValues(model)

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
        raise NotImplementedError("DBSCAN does not can not fit and generate model")

    def _out_schema(self) -> Union[StructType, str]:
        return StructType()


class DBSCANModel(
    DBSCANClass, _CumlCaller, _CumlModelWithPredictionCol, _DBSCANCumlParams
):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
        verbose: Union[int, bool],
    ):
        super(DBSCANClass, self).__init__()
        super(_CumlModelWithPredictionCol, self).__init__(n_cols=n_cols, dtype=dtype)
        super(_DBSCANCumlParams, self).__init__()

        self._setDefault(
            idCol=alias.row_number,
        )

        self.verbose = verbose
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
            select_cols.append(col(id_col_name).alias(alias.row_number))
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
        logger = get_logger(self.__class__)

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )

        inputs = []  # type: ignore

        idCol = list(
            self.idCols_[0].value
            if len(self.idCols_) == 1
            else np.concatenate([chunk.value for chunk in self.idCols_])
        )

        for pdf_bc in self.raw_data_:
            pdf = pd.DataFrame(data=pdf_bc.value, columns=self.processed_input_cols)

            if self.multi_col_names:
                features = np.array(pdf[self.multi_col_names], order=array_order)
            elif self.use_sparse_array:
                # sparse vector
                features = _read_csr_matrix_from_unwrapped_spark_vec(pdf)
            else:
                # dense vector
                features = np.array(list(pdf[alias.data]), order=array_order)

            # experiments indicate it is faster to convert to numpy array and then to cupy array than directly
            # invoking cupy array on the list
            if cuda_managed_mem_enabled:
                features = (
                    cp.array(features)
                    if self.use_sparse_array is False
                    else cupyx.scipy.sparse.csr_matrix(features)
                )

            inputs.append(features)

        if isinstance(inputs[0], pd.DataFrame):
            concated = pd.concat(inputs)
        else:
            # features are either cp or np arrays here
            concated = _concat_and_free(inputs, order=array_order)

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.cluster.dbscan_mg import DBSCANMG as CumlDBSCANMG
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()

            logger = get_logger(self.__class__)

            dbscan = CumlDBSCANMG(
                handle=params[param_alias.handle],
                output_type="cudf",
                eps=self.getOrDefault("eps"),
                min_samples=self.getOrDefault("min_samples"),
                metric=self.getOrDefault("metric"),
                max_mbytes_per_batch=self.getOrDefault("max_mbytes_per_batch"),
                calc_core_sample_indices=self.getOrDefault("calc_core_sample_indices"),
                verbose=self.verbose,
            )
            dbscan.n_cols = params[param_alias.num_cols]
            dbscan.dtype = np.dtype(dtype)

            res = list(dbscan.fit_predict(concated).to_numpy())

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
        raise NotImplementedError(
            "DBSCAN does not can not have a separate transform UDF"
        )

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
        use_sparse_array = _use_sparse_in_cuml(dataset)
        input_dataset = dataset.select(*select_cols)
        pd_dataset: pd.DataFrame = input_dataset.toPandas()
        raw_data: np.ndarray = np.array(pd_dataset.drop(columns=[self.getIdCol()]))
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
        self.use_sparse_array = use_sparse_array

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

    def _get_model_attributes(self) -> Optional[Dict[str, Any]]:
        """
        Override parent method to bring broadcast variables to driver before JSON serialization.
        """

        self._model_attributes["verbose"] = self.verbose

        return self._model_attributes

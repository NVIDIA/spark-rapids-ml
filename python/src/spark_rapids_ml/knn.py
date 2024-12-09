#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import asyncio
import inspect
import math
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from pyspark import keyword_only
from pyspark.broadcast import Broadcast
from pyspark.ml import Estimator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasLabelCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.util import MLReader, MLWriter
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    LongType,
    Row,
    StructField,
    StructType,
)

from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlCaller,
    _CumlEstimator,
    _CumlEstimatorSupervised,
    _CumlModel,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
)
from .metrics import EvalMetricInfo
from .params import DictTypeConverters, HasIDCol, P, _CumlClass, _CumlParams
from .utils import (
    _concat_and_free,
    _get_class_or_callable_name,
    _get_default_params_from_func,
    _get_spark_session,
    get_logger,
)


class NearestNeighborsClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {"k": "n_neighbors"}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {"n_neighbors": 5, "verbose": False, "batch_size": 2000000}

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None


class _NearestNeighborsCumlParams(
    _CumlParams, HasInputCol, HasLabelCol, HasInputCols, HasIDCol
):
    """
    Shared Spark Params for NearestNeighbor and NearestNeighborModel.
    """

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(idCol=None)

    k = Param(
        Params._dummy(),
        "k",
        "The number nearest neighbors to retrieve. Must be >= 1.",
        typeConverter=TypeConverters.toInt,
    )

    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column name.",
        typeConverter=TypeConverters.toString,
    )

    def setK(self: P, value: int) -> P:
        """
        Sets the value of `k`.
        """
        self._set_params(k=value)
        return self

    def getK(self: P) -> int:
        """
        Get the value of `k`.
        """
        return self.getOrDefault("k")

    def _getIdColOrDefault(self) -> str:
        """
        Gets the value of `idCol`.
        """

        res = self.getIdCol()
        if res is None:
            res = alias.row_number
        return res

    def setInputCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`inputCol` or :py:attr:`inputCols`.
        """
        if isinstance(value, str):
            self._set_params(inputCol=value)
        else:
            self._set_params(inputCols=value)
        return self

    def setInputCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`inputCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self._set_params(inputCols=value)

    def setIdCol(self: P, value: str) -> P:
        """
        Sets the value of `idCol`. If not set, an id column will be added with column name `unique_id`. The id column is used to specify nearest neighbor vectors by associated id value.
        """
        self._set_params(idCol=value)
        return self

    def _ensureIdCol(self, df: DataFrame) -> DataFrame:
        """
        Ensure an id column exists in the input dataframe. Add the column if not exists.
        Overwritten for knn assumption on error for not setting idCol and duplicate exists.
        """

        id_col_name = self.getIdCol()
        if id_col_name is None:
            if alias.row_number in df.columns:
                raise ValueError(
                    f"Trying to create an id column with default name {alias.row_number}. But a column with the same name already exists."
                )
            else:
                get_logger(self.__class__).info(
                    f"idCol not set. Spark Rapids ML will create one with default name {alias.row_number}."
                )
                df_withid = df.select(
                    monotonically_increasing_id().alias(alias.row_number), "*"
                )
                return df_withid
        else:
            if id_col_name in df.columns:
                return df
            else:
                get_logger(self.__class__).info(
                    f"column {id_col_name} does not exists in the input dataframe. Spark Rapids ML will create the {id_col_name} column."
                )
                df_withid = df.select(
                    monotonically_increasing_id().alias(alias.row_number), "*"
                )
                return df_withid


class NearestNeighbors(
    NearestNeighborsClass, _CumlEstimatorSupervised, _NearestNeighborsCumlParams
):
    """
    NearestNeighbors retrieves the exact k nearest neighbors in item vectors for each
    query vector. The main methods accept distributed CPU dataframes as inputs,
    leverage GPUs to accelerate computation, and take care of communication and
    aggregation automatically. However, it should be noted that only the euclidean
    distance (also known as L2 distance) is supported in the current implementations
    and the feature data type must be of the float type. All other data types will
    be converted into float during computation.

    Parameters
    ----------
    k: int (default = 5)
        the default number nearest neighbors to retrieve for each query.

    inputCol: str or List[str]
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.

    idCol: str (default = None)
        the name of the column in a dataframe that uniquely identifies each vector. idCol should be set
        if such a column exists in the dataframe. If idCol is not set, a column with the name `unique_id`
        will be automatically added to the dataframe and used as unique identifier for each vector.

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
    >>> from spark_rapids_ml.knn import NearestNeighbors
    >>> data = [(0, [1.0, 1.0]),
    ...         (1, [2.0, 2.0]),
    ...         (2, [3.0, 3.0]),]
    >>> data_df = spark.createDataFrame(data, schema="id int, features array<float>")
    >>> query = [(3, [1.0, 1.0]),
    ...          (4, [3.0, 3.0]),]
    >>> query_df = spark.createDataFrame(query, schema="id int, features array<float>")
    >>> topk = 2
    >>> gpu_knn = NearestNeighbors().setInputCol("features").setIdCol("id").setK(topk)
    >>> gpu_model = gpu_knn.fit(data_df)
    >>> (data_df, query_df, knn_df) = gpu_model.kneighbors(query_df)
    >>> knn_df.show()
    +--------+-------+----------------+
    |query_id|indices|       distances|
    +--------+-------+----------------+
    |       3| [0, 1]|[0.0, 1.4142135]|
    |       4| [2, 1]|[0.0, 1.4142135]|
    +--------+-------+----------------+
    >>> data_df.show()
    +---+----------+
    | id|  features|
    +---+----------+
    |  0|[1.0, 1.0]|
    |  1|[2.0, 2.0]|
    |  2|[3.0, 3.0]|
    +---+----------+
    >>> query_df.show()
    +---+----------+
    | id|  features|
    +---+----------+
    |  3|[1.0, 1.0]|
    |  4|[3.0, 3.0]|
    +---+----------+
    >>> knnjoin_df = gpu_model.exactNearestNeighborsJoin(query_df, distCol="EuclideanDistance")
    >>> knnjoin_df.show()
    +---------------+---------------+-----------------+
    |        item_df|       query_df|EuclideanDistance|
    +---------------+---------------+-----------------+
    |{1, [2.0, 2.0]}|{3, [1.0, 1.0]}|        1.4142135|
    |{0, [1.0, 1.0]}|{3, [1.0, 1.0]}|              0.0|
    |{2, [3.0, 3.0]}|{4, [3.0, 3.0]}|              0.0|
    |{1, [2.0, 2.0]}|{4, [3.0, 3.0]}|        1.4142135|
    +---------------+---------------+-----------------+

    >>> # vector column input
    >>> from spark_rapids_ml.knn import NearestNeighbors
    >>> from pyspark.ml.linalg import Vectors
    >>> data = [(0, Vectors.dense([1.0, 1.0]),),
    ...         (1, Vectors.dense([2.0, 2.0]),),
    ...         (2, Vectors.dense([3.0, 3.0]),)]
    >>> data_df = spark.createDataFrame(data, ["id", "features"])
    >>> query = [(3, Vectors.dense([1.0, 1.0]),),
    ...          (4, Vectors.dense([3.0, 3.0]),)]
    >>> query_df = spark.createDataFrame(query, ["id", "features"])
    >>> topk = 2
    >>> gpu_knn = NearestNeighbors().setInputCol("features").setIdCol("id").setK(topk)
    >>> gpu_model = gpu_knn.fit(data_df)

    >>> # multi-column input
    >>> from spark_rapids_ml.knn import NearestNeighbors
    >>> data = [(0, 1.0, 1.0),
    ...         (1, 2.0, 2.0),
    ...         (2, 3.0, 3.0),]
    >>> data_df = spark.createDataFrame(data, schema="id int, f1 float, f2 float")
    >>> query = [(3, 1.0, 1.0),
    ...          (4, 3.0, 3.0),]
    >>> query_df = spark.createDataFrame(query, schema="id int, f1 float, f2 float")
    >>> topk = 2
    >>> gpu_knn = NearestNeighbors().setInputCols(["f1", "f2"]).setIdCol("id").setK(topk)
    >>> gpu_model = gpu_knn.fit(data_df)
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: Optional[int] = None,
        inputCol: Optional[Union[str, List[str]]] = None,
        idCol: Optional[str] = None,
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ) -> None:
        if not self._input_kwargs.get("float32_inputs", True):
            get_logger(self.__class__).warning(
                "This estimator does not support double precision inputs. Setting float32_inputs to False will be ignored."
            )
            self._input_kwargs.pop("float32_inputs")

        super().__init__()
        self._set_params(**self._input_kwargs)
        self._label_isdata = 0
        self._label_isquery = 1
        self._set_params(labelCol=alias.label)

    def _create_pyspark_model(self, result: Row) -> "NearestNeighborsModel":
        return NearestNeighborsModel._from_row(result)

    def _fit(self, item_df: DataFrame) -> "NearestNeighborsModel":
        self._item_df_withid = self._ensureIdCol(item_df)

        self._processed_item_df = self._item_df_withid.withColumn(
            alias.label, lit(self._label_isdata)
        )

        # TODO: should test this at scale to see if/when we hit limits
        model = self._create_pyspark_model(
            Row(
                item_df_withid=self._item_df_withid,
                processed_item_df=self._processed_item_df,
                label_isdata=self._label_isdata,
                label_isquery=self._label_isquery,
            )
        )
        model._num_workers = self._num_workers
        model._float32_inputs = self._float32_inputs
        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore
        return model

    def _out_schema(self) -> Union[StructType, str]:  # type: ignore
        """
        This class overrides _fit and will not call _out_schema.
        """
        pass

    def _get_cuml_fit_func(self, dataset: DataFrame) -> Callable[  # type: ignore
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        """
        This class overrides _fit and will not call _get_cuml_fit_func.
        """
        pass

    def write(self) -> MLWriter:
        """Unsupported."""
        raise NotImplementedError(
            "NearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def read(cls) -> MLReader:
        """Unsupported."""
        raise NotImplementedError(
            "NearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    def save(self, path: str) -> None:
        """Unsupported."""
        raise NotImplementedError(
            "NearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def load(cls, path: str) -> MLReader:
        """Unsupported."""
        raise NotImplementedError(
            "NearestNeighbors does not support saving/loading, just re-create the estimator."
        )


class _NNModelBase(_CumlModel, _NearestNeighborsCumlParams):

    def _transform(self, dataset: DataFrame) -> DataFrame:
        raise NotImplementedError(
            f"{self.__class__} does not provide a transform function. Use 'kneighbors' instead."
        )

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        raise NotImplementedError(
            "'_CumlModel._get_cuml_transform_func' method is not implemented. Use 'kneighbors' instead."
        )

    @abstractmethod
    def kneighbors(
        self, query_df: DataFrame, sort_knn_df_by_query_id: bool = True
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        raise NotImplementedError()

    def _nearest_neighbors_join(
        self,
        query_df: DataFrame,
        distCol: str = "distCol",
    ) -> DataFrame:

        id_col_name = self._getIdColOrDefault()

        # call kneighbors then prepare return results
        (item_df_withid, query_df_withid, knn_df) = self.kneighbors(
            query_df, sort_knn_df_by_query_id=False
        )

        from pyspark.sql.functions import arrays_zip, col, explode, struct

        knn_pair_df = knn_df.select(
            f"query_{id_col_name}",
            explode(arrays_zip("indices", "distances")).alias("zipped"),
        ).select(
            f"query_{id_col_name}",
            col("zipped.indices").alias(f"item_{id_col_name}"),
            col("zipped.distances").alias(distCol),
        )

        item_df_struct = item_df_withid.select(struct("*").alias("item_df"))
        query_df_struct = query_df_withid.select(struct("*").alias("query_df"))

        knnjoin_df = item_df_struct.join(
            knn_pair_df,
            item_df_struct[f"item_df.{id_col_name}"]
            == knn_pair_df[f"item_{id_col_name}"],
        )
        knnjoin_df = knnjoin_df.join(
            query_df_struct,
            knnjoin_df[f"query_{id_col_name}"]
            == query_df_struct[f"query_df.{id_col_name}"],
        )

        if self.isSet(self.idCol):
            knnjoin_df = knnjoin_df.select("item_df", "query_df", distCol)
        else:
            knnjoin_df = knnjoin_df.select(
                knnjoin_df["item_df"].dropFields(id_col_name).alias("item_df"),
                knnjoin_df["query_df"].dropFields(id_col_name).alias("query_df"),
                distCol,
            )

        return knnjoin_df

    def write(self) -> MLWriter:
        """Unsupported."""
        raise NotImplementedError(
            f"{self.__class__} does not support saving/loading, just re-fit the estimator to re-create a model."
        )

    @classmethod
    def read(cls) -> MLReader:
        """Unsupported."""
        raise NotImplementedError(
            f"{cls} does not support saving/loading, just re-fit the estimator to re-create a model."
        )

    def save(self, path: str) -> None:
        """Unsupported."""
        raise NotImplementedError(
            f"{self.__class__} does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def load(cls, path: str) -> MLReader:
        """Unsupported."""
        raise NotImplementedError(
            f"{cls} does not support saving/loading, just re-create the estimator."
        )


class NearestNeighborsModel(_CumlCaller, _NNModelBase, NearestNeighborsClass):
    def __init__(
        self,
        item_df_withid: DataFrame,
        processed_item_df: DataFrame,
        label_isdata: int,
        label_isquery: int,
    ):
        super().__init__()
        self._item_df_withid = item_df_withid
        self._processed_item_df = processed_item_df
        self._label_isdata = label_isdata
        self._label_isquery = label_isquery

    def _out_schema(self) -> Union[StructType, str]:  # type: ignore
        return StructType(
            [
                StructField(
                    f"query_{self._getIdColOrDefault()}",
                    ArrayType(LongType(), False),
                    False,
                ),
                StructField(
                    "indices", ArrayType(ArrayType(LongType(), False), False), False
                ),
                StructField(
                    "distances", ArrayType(ArrayType(DoubleType(), False), False), False
                ),
            ]
        )

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        """Enable ucx over NCCL"""
        return (True, True)

    def _pre_process_data(  # type: ignore
        self, dataset: DataFrame
    ) -> Tuple[
        List[Column], Optional[List[str]], int, Union[Type[FloatType], Type[DoubleType]]
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = super()._pre_process_data(dataset)

        # if input format is vectorUDT, convert data type from float64 to float32
        input_col, _ = self._get_input_columns()
        if input_col is not None and isinstance(
            dataset.schema[input_col].dataType, VectorUDT
        ):
            select_cols[0] = vector_to_array(col(input_col), dtype="float32").alias(
                alias.data
            )

        select_cols.append(col(alias.label))

        id_col_name = self._getIdColOrDefault()
        select_cols.append(col(id_col_name).alias(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type

    def kneighbors(
        self, query_df: DataFrame, sort_knn_df_by_query_id: bool = True
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Return the exact nearest neighbors for each query in query_df. The data
        vectors (or equivalently item vectors) should be provided through the fit
        function (see Examples in the spark_rapids_ml.knn.NearestNeighbors). The
        distance measure here is euclidean distance and the number of target exact
        nearest neighbors can be set through setK(). The function currently only
        supports float32 type and will convert other data types into float32.

        Parameters
        ----------
        query_df: pyspark.sql.DataFrame
            query vectors where each row corresponds to one query. The query_df can be in the
            format of a single array column, a single vector column, or multiple float columns.

        sort_knn_df_by_query_id: bool (default=True)
            whether to sort the returned dataframe knn_df by query_id

        Returns
        -------
        query_df: pyspark.sql.DataFrame
            the query_df itself if it has an id column set through setIdCol(). If not,
            a monotonically increasing id column will be added.

        item_df: pyspark.sql.DataFrame
            the item_df (or equivalently data_df) itself if it has an id column set
            through setIdCol(). If not, a monotonically increasing id column will be added.

        knn_df: pyspark.sql.DataFrame
            the result k nearest neighbors (knn) dataframe that has three
            columns (id, indices, distances). Each row of knn_df corresponds to the knn
            result of a query vector, identified by the id column. The indices/distances
            column stores the ids/distances of knn item_df vectors.
        """

        query_default_num_partitions = query_df.rdd.getNumPartitions()

        query_df_withid = self._ensureIdCol(query_df)

        processed_query_df = query_df_withid.withColumn(
            alias.label, lit(self._label_isquery)
        )

        def select_cols_for_cuml_fit(df_origin: DataFrame) -> DataFrame:
            cols_for_nns = [self._getIdColOrDefault(), alias.label]
            input_col, input_cols = self._get_input_columns()
            if input_col is not None:
                cols_for_nns.append(input_col)
            else:
                assert input_cols is not None
                cols_for_nns += input_cols

            return df_origin.select(cols_for_nns)

        df_item_for_nns = select_cols_for_cuml_fit(self._processed_item_df)
        df_query_for_nns = select_cols_for_cuml_fit(processed_query_df)
        union_df = df_item_for_nns.union(df_query_for_nns)

        pipelinedrdd = self._call_cuml_fit_func(union_df, partially_collect=False)
        pipelinedrdd = pipelinedrdd.repartition(query_default_num_partitions)  # type: ignore

        query_id_col_name = f"query_{self._getIdColOrDefault()}"
        id_col_type = dict(union_df.dtypes)[self._getIdColOrDefault()]
        knn_rdd = pipelinedrdd.flatMap(
            lambda row: list(
                zip(row[query_id_col_name], row["indices"], row["distances"])
            )
        )
        knn_df = knn_rdd.toDF(
            schema=f"{query_id_col_name} {id_col_type}, indices array<{id_col_type}>, distances array<float>"
        )

        knn_df_returned = (
            knn_df
            if sort_knn_df_by_query_id is False
            else knn_df.sort(query_id_col_name)
        )

        return (self._item_df_withid, query_df_withid, knn_df_returned)

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        label_isdata = self._label_isdata
        label_isquery = self._label_isquery
        id_col_name = self._getIdColOrDefault()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            rank = context.partitionId()

            from cuml.neighbors.nearest_neighbors_mg import NearestNeighborsMG as cumlNN

            nn_object = cumlNN(
                handle=params[param_alias.handle],
                n_neighbors=params[param_alias.cuml_init]["n_neighbors"],
                output_type="numpy",
                verbose=params[param_alias.cuml_init]["verbose"],
            )

            item_list = []
            query_list = []
            item_row_number = []
            query_row_number = []

            for x_array, label_array, row_number_array in dfs:
                item_filter = label_array == label_isdata
                query_filter = label_array == label_isquery

                item_list.append(x_array[item_filter])
                query_list.append(x_array[query_filter])

                item_row_number += row_number_array[item_filter].tolist()  # type: ignore
                query_row_number += row_number_array[query_filter].tolist()  # type: ignore

            if isinstance(item_list[0], pd.DataFrame):
                item = [pd.concat(item_list)]
                query = [pd.concat(query_list)]
            else:
                # do not use item_list or query_list after this, as elements are freed
                item = [_concat_and_free(item_list)]
                query = [_concat_and_free(query_list)]

            item_row_number = [item_row_number]
            query_row_number = [query_row_number]

            item_size: List[int] = [len(chunk) for chunk in item]
            query_size: List[int] = [len(chunk) for chunk in query]
            assert len(item_size) == len(query_size)
            import json

            async def do_allGather() -> List[str]:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    context.allGather,
                    json.dumps((rank, item_size, query_size, item_row_number)),
                )
                return result

            messages = params[param_alias.loop].run_until_complete(
                asyncio.ensure_future(do_allGather())
            )

            rank_stats = [json.loads(msg) for msg in messages]

            item_parts_to_ranks = []
            query_parts_to_ranks = []
            for m_rank, m_item_size, m_query_size, _ in rank_stats:
                item_parts_to_ranks += [(m_rank, size) for size in m_item_size]
                query_parts_to_ranks += [(m_rank, size) for size in m_query_size]
            item_nrows = sum(pair[1] for pair in item_parts_to_ranks)
            query_nrows = sum(pair[1] for pair in query_parts_to_ranks)

            res_tuple: Tuple[List[np.ndarray], List[np.ndarray]] = nn_object.kneighbors(
                index=item,
                index_parts_to_ranks=item_parts_to_ranks,
                index_nrows=item_nrows,
                query=query,
                query_parts_to_ranks=query_parts_to_ranks,
                query_nrows=query_nrows,
                ncols=params[param_alias.num_cols],
                rank=rank,
                n_neighbors=params[param_alias.cuml_init]["n_neighbors"],
                convert_dtype=False,  # only np.float32 is supported in cuml. Should set to True for all other types
            )

            distances: List[np.ndarray] = res_tuple[0]
            indices: List[np.ndarray] = res_tuple[1]

            distances = [ary.tolist() for ary in distances]
            indices = [ary.tolist() for ary in indices]

            # id mapping
            id2row: Dict[int, int] = {}
            count = 0
            for _, _, _, m_item_row_number in rank_stats:
                for chunk in m_item_row_number:
                    chunk_id2row = [(count + i, chunk[i]) for i in range(len(chunk))]
                    id2row.update(chunk_id2row)
                    count += len(chunk)

            transformed_indices = []
            for two_d in indices:
                res = []
                for row in two_d:
                    res.append([id2row[cuid] for cuid in row])
                transformed_indices.append(res)

            return {
                f"query_{id_col_name}": query_row_number,
                "indices": transformed_indices,
                "distances": distances,
            }

        return _cuml_fit

    def exactNearestNeighborsJoin(
        self,
        query_df: DataFrame,
        distCol: str = "distCol",
    ) -> DataFrame:
        """
        This function returns the k exact nearest neighbors (knn) in item_df of each query vector in query_df.
        item_df is the dataframe passed to the fit function of the NearestNeighbors estimator.
        Note that the knn relationship is asymmetric with respect to the input datasets (e.g., if x is a knn of y
        , y is not necessarily a knn of x).

        Parameters
        ----------
        query_df: pyspark.sql.DataFrame
            the query_df dataframe. Each row represents a query vector.

        distCol: str
            the name of the output distance column

        Returns
        -------
        knnjoin_df: pyspark.sql.DataFrame
            the result dataframe that has three columns (item_df, query_df, distCol).
            item_df column is of struct type that includes as fields all the columns of input item dataframe.
            Similarly, query_df column is of struct type that includes as fields all the columns of input query dataframe.
            distCol is the distance column. A row in knnjoin_df is in the format (v1, v2, dist(v1, v2)),
            where item_vector v1 is one of the k nearest neighbors of query_vector v2 and their distance is dist(v1, v2).
        """

        return self._nearest_neighbors_join(query_df=query_df, distCol=distCol)


class ApproximateNearestNeighborsClass(_CumlClass):

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "k": "n_neighbors",
            "algorithm": "algorithm",
            "metric": "metric",
            "algoParams": "algo_params",
        }

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "n_neighbors": 5,
            "verbose": False,
            "algorithm": "ivfflat",
            "metric": "euclidean",
            "algo_params": None,
        }

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None


class _ApproximateNearestNeighborsParams(_NearestNeighborsCumlParams):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault(algorithm="ivfflat")
        self._setDefault(algoParams=None)
        self._setDefault(metric="euclidean")

    algorithm = Param(
        Params._dummy(),
        "algorithm",
        "The algorithm to use for approximate nearest neighbors search.",
        typeConverter=TypeConverters.toString,
    )

    algoParams = Param(
        Params._dummy(),
        "algoParams",
        "The parameters to use to set up a neighbor algorithm.",
        typeConverter=DictTypeConverters._toDict,
    )

    metric = Param(
        Params._dummy(),
        "metric",
        "The distance metric to use.",
        typeConverter=TypeConverters.toString,
    )

    def setAlgorithm(self: P, value: str) -> P:
        """
        Sets the value of `algorithm`.
        """
        assert value in {
            "ivfflat",
            "ivfpq",
            "cagra",
        }, "Only ivfflat, ivfpq, and cagra are currently supported"
        self._set_params(algorithm=value)
        return self

    def getAlgorithm(self: P) -> str:
        """
        Gets the value of `algorithm`.
        """
        return self.getOrDefault("algorithm")

    def setAlgoParams(self: P, value: Dict[str, Any]) -> P:
        """
        Sets the value of `algoParams`.
        """
        self._set_params(algoParams=value)
        return self

    def getAlgoParams(self: P) -> Dict[str, Any]:
        """
        Gets the value of `algoParams`.
        """
        return self.getOrDefault("algoParams")

    def setMetric(self: P, value: str) -> P:
        """
        Sets the value of `metric`.
        """
        self._set_params(metric=value)
        return self

    def getMetric(self: P) -> str:
        """
        Gets the value of `metric`.
        """
        return self.getOrDefault("metric")


class ApproximateNearestNeighbors(
    ApproximateNearestNeighborsClass, _CumlEstimator, _ApproximateNearestNeighborsParams
):
    """
    ApproximateNearestNeighbors retrieves k approximate nearest neighbors (ANNs) in item vectors for each query.
    The key APIs are similar to the NearestNeighbor class which returns the exact k nearest neighbors.
    The ApproximateNearestNeighbors is currently implemented using cuvs. It supports the IVFFLAT, IVFPQ, and
    CAGRA (graph-based) algorithms and follows the API conventions of cuML.

    The current implementation build index independently on each data partition of item_df. Queries will be broadcast to all GPUs,
    then every query probes closest centers on individual index. Local topk results will be aggregated to obtain global topk ANNs.

    CAGRA is a graph-based algorithm designed to construct a nearest neighbors graph index using either "ivf_pq" or "nn_descent" method.
    This index is then utilized to efficiently answer approximate nearest neighbor (ANN) queries. Graph-based algorithms have consistently
    demonstrated superior performance in ANN search, offering the fastest search speeds with minimal loss in search quality. Due to the high
    computational complexity involved in graph construction, these algorithms are particularly well-suited for GPU acceleration.

    IVFFLAT algorithm trains a set of kmeans centers, then partition every item vector to the closest center. In the query processing
    phase, a query will be partitioned into a number of closest centers, and probe all the items associated with those centers. In
    the end the top k closest items will be returned as the approximate nearest neighbors.

    The IVFPQ algorithm employs product quantization to compress high-dimensional vectors into compact bit representations,
    enabling rapid distance computation between vectors. While IVFPQ typically delivers faster search speeds compared to IVFFLAT,
    it does so with a tradeoff in search quality, such as reduced recall. It is important to note that the distances returned by IVFPQ
    are approximate and do not represent the exact distances in the original high-dimensional space.

    Parameters
    ----------
    k: int (default = 5)
        the default number of approximate nearest neighbors to retrieve for each query.

        If fewer than k neighbors are found for a query (for example, due to a small nprobe value):
        (1)In ivfflat and ivfpq:
            (a) If no item vector is probed, the indices are filled with long_max (9,223,372,036,854,775,807) and distances are set to infinity.
            (b) If at least one item vector is probed, the indices are filled with the top-1 neighbor's ID, and distances are filled with infinity.
        (2) cagra does not have this problem, as at least itopk_size (where itopk_size ≥ k) items are always probed.

    algorithm: str (default = 'ivfflat')
        the algorithm parameter to be passed into cuML. It currently must be 'ivfflat', 'ivfpq' or 'cagra'. Other algorithms are expected to be supported later.

    algoParams: Optional[Dict[str, Any]] (default = None)
        if set, algoParam is used to configure the algorithm, on each data partition (or maxRecordsPerBatch if Arrow is enabled) of the item_df.
        Note this class constructs the kmeans index independently on individual data partition (or maxRecordPerBatch if Arrow is enabled).
        When algorithm is 'cagra', parameters for index construction:

            - build_algo: (str, default = 'ivf_pq') algorithm to build graph index, can be either 'ivf_pq' or 'nn_descent'. nn_descent is expected to be generally faster than ivf_pq.
            - intermediate_graph_degree: (int, default = 128) an intermediate variable used during graph index construction.
            - graph_degree: (int, default = 64) the degree of each node in the final graph index.

        When algorithm is 'cagra', parameters for search (full list in `cuvs python API documentation <https://docs.rapids.ai/api/cuvs/stable/python_api/neighbors_cagra/#cuvs.neighbors.cagra.SearchParams>`_):

            - itopk_size: (int, default = 64) number of intermediate search results retained during the search. Larger value improves the search accuracy but increases the search time. cuVS internally increases the value to be multiple of 32 and expects the internal value to be larger than or equal to k.
            - max_iterations (int, default = 0) maximum number of search iterations. 0 means auto select.
            - min_iterations (int, default = 0) minimum number of search iterations. 0 means auto select.
            - search_width: (int, default = 1) number of graph nodes as the initial set of search points in each iteration.
            - num_random_samplings: (int, default = 1) number of iterations for selecting initial random seed nodes.

        When algorithm is 'ivfflat':

            - nlist: (int) number of kmeans clusters to partition the dataframe into.
            - nprobe: (int) number of closest clusters to probe for topk ANNs.

        When algorithm is 'ivfpq':

            - nlist: (int) number of kmeans clusters to partition the dataframe into.
            - nprobe: (int) number of closest clusters to probe for topk ANNs.
            - M: (int) number of subquantizers
            - n_bits: (int) number of bits allocated per subquantizer

            Note cuml requires M * n_bits to be multiple of 8 for the best efficiency.

    metric: str (default = "euclidean")
        the distance metric to use with the default set to "euclidean" (following cuml conventions, though cuvs defaults to "sqeuclidean").
        The 'ivfflat' and 'ivfpq' algorithms support ['euclidean', 'sqeuclidean', 'l2', 'inner_product', 'cosine'].
        The 'cagra' algorithm supports ['sqeuclidean'], and when using 'cagra' as an algorithm,
        the metric must be explicitly set to 'sqeuclidean'.

    inputCol: str or List[str]
        The feature column names, spark-rapids-ml supports vector, array and columnar as the input.\n
            * When the value is a string, the feature columns must be assembled into 1 column with vector or array type.
            * When the value is a list of strings, the feature columns must be numeric types.

    idCol: str (default = None)
        the name of the column in a dataframe that uniquely identifies each vector. idCol should be set
        if such a column exists in the dataframe. If idCol is not set, a column with the name `unique_id`
        will be automatically added to the dataframe and used as unique identifier for each vector.

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
    >>> from spark_rapids_ml.knn import ApproximateNearestNeighbors
    >>> data = [(0, [0.0, 0.0]),
    ...         (1, [1.0, 1.0]),
    ...         (2, [2.0, 2.0]),
    ...         (3, [30.0, 30.0]),
    ...         (4, [40.0, 40.0]),
    ...         (5, [50.0, 50.0]),]
    >>> data_df = spark.createDataFrame(data, schema="id int, features array<float>")
    >>> data_df = data_df.repartition(2) # ensure each partition having more data vectors than the 'nlist' of 'ivfflat'
    >>> query = [(10, [0.0, 0.0]),
    ...          (11, [50.0, 50.0]),]
    >>> query_df = spark.createDataFrame(query, schema="id int, features array<float>")
    >>> topk = 2
    >>> gpu_knn = ApproximateNearestNeighbors().setAlgorithm('ivfflat').setAlgoParams({"nlist" : 2, "nprobe": 1})
    >>> gpu_knn = gpu_knn.setInputCol("features").setIdCol("id").setK(topk)
    >>> gpu_model = gpu_knn.fit(data_df)
    >>> (data_df, query_df, knn_df) = gpu_model.kneighbors(query_df)
    >>> knn_df.show()
    +--------+-------+----------------+
    |query_id|indices|       distances|
    +--------+-------+----------------+
    |      10| [0, 1]|[0.0, 1.4142134]|
    |      11| [5, 4]|[0.0, 14.142137]|
    +--------+-------+----------------+
    >>> data_df.show()
    +---+------------+
    | id|    features|
    +---+------------+
    |  0|  [0.0, 0.0]|
    |  1|  [1.0, 1.0]|
    |  4|[40.0, 40.0]|
    |  2|  [2.0, 2.0]|
    |  3|[30.0, 30.0]|
    |  5|[50.0, 50.0]|
    +---+------------+

    >>> query_df.show()
    +---+------------+
    | id|    features|
    +---+------------+
    | 10|  [0.0, 0.0]|
    | 11|[50.0, 50.0]|
    +---+------------+

    >>> knnjoin_df = gpu_model.approxSimilarityJoin(query_df, distCol="EuclideanDistance")
    +-----------------+------------------+-----------------+
    |          item_df|          query_df|EuclideanDistance|
    +-----------------+------------------+-----------------+
    |  {0, [0.0, 0.0]}|  {10, [0.0, 0.0]}|              0.0|
    |  {1, [1.0, 1.0]}|  {10, [0.0, 0.0]}|        1.4142134|
    |{5, [50.0, 50.0]}|{11, [50.0, 50.0]}|              0.0|
    |{4, [40.0, 40.0]}|{11, [50.0, 50.0]}|        14.142137|
    +-----------------+------------------+-----------------+


    >>> # vector column input
    >>> from spark_rapids_ml.knn import ApproximateNearestNeighbors
    >>> from pyspark.ml.linalg import Vectors
    >>> data = [(0, Vectors.dense([0.0, 0.0])),
    ...         (1, Vectors.dense([1.0, 1.0])),
    ...         (2, Vectors.dense([2.0, 2.0])),
    ...         (3, Vectors.dense([30.0, 30.0])),
    ...         (4, Vectors.dense([40.0, 40.0])),
    ...         (5, Vectors.dense([50.0, 50.0])),]
    >>> data_df = spark.createDataFrame(data, ["id", "features"]).repartition(2)
    >>> query = [(10, Vectors.dense([0.0, 0.0])),
    ...          (11, Vectors.dense([50.0, 50.0])),]
    >>> query_df = spark.createDataFrame(query, ["id", "features"])
    >>> topk = 2
    >>> gpu_knn = ApproximateNearestNeighbors().setAlgorithm('ivfflat').setAlgoParams({"nlist" : 2, "nprobe": 1})
    >>> gpu_knn = gpu_knn.setInputCol("features").setIdCol("id").setK(topk)
    >>> gpu_model = gpu_knn.fit(data_df)
    >>> (data_df, query_df, knn_df) = gpu_model.kneighbors(query_df)
    >>> knn_df.show()


    >>> # multi-column input
    >>> from spark_rapids_ml.knn import ApproximateNearestNeighbors
    >>> data = [(0, 0.0, 0.0),
    ...         (1, 1.0, 1.0),
    ...         (2, 2.0, 2.0),
    ...         (3, 30.0, 30.0),
    ...         (4, 40.0, 40.0),
    ...         (5, 50.0, 50.0),]
    >>> data_df = spark.createDataFrame(data, schema="id int, f1 float, f2 float").repartition(2)
    >>> query = [(10, 0.0, 0.0),
    ...          (11, 50.0, 50.0),]
    >>> query_df = spark.createDataFrame(query, schema="id int, f1 float, f2 float")
    >>> topk = 2
    >>> gpu_knn = ApproximateNearestNeighbors().setAlgorithm('ivfflat').setAlgoParams({"nlist" : 2, "nprobe": 1})
    >>> gpu_knn = gpu_knn.setInputCols(["f1", "f2"]).setIdCol("id").setK(topk)
    >>> gpu_model = gpu_knn.fit(data_df)
    >>> (data_df, query_df, knn_df) = gpu_model.kneighbors(query_df)
    >>> knn_df.show()
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: Optional[int] = None,
        algorithm: str = "ivfflat",
        metric: str = "euclidean",
        algoParams: Optional[Dict[str, Any]] = None,
        inputCol: Optional[Union[str, List[str]]] = None,
        idCol: Optional[str] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert algorithm in {
            "ivfflat",
            "ivfpq",
            "cagra",
        }, "currently only ivfflat, ivfpq, and cagra are supported"
        if not self._input_kwargs.get("float32_inputs", True):
            get_logger(self.__class__).warning(
                "This estimator supports only float32 inputs on GPU and will convert all other data types to float32. Setting float32_inputs to False will be ignored."
            )
            self._input_kwargs.pop("float32_inputs")

        self._set_params(**self._input_kwargs)

    def _fit(self, item_df: DataFrame) -> "ApproximateNearestNeighborsModel":  # type: ignore
        self._item_df_withid = self._ensureIdCol(item_df).coalesce(self.num_workers)

        model = self._create_pyspark_model(
            Row(
                item_df_withid=self._item_df_withid,
            )
        )
        model._float32_inputs = self._float32_inputs
        self._copyValues(model)
        self._copy_cuml_params(model)  # type: ignore
        return model

    def _create_pyspark_model(self, result: Row) -> "ApproximateNearestNeighborsModel":  # type: ignore
        return ApproximateNearestNeighborsModel._from_row(result)

    def _out_schema(self) -> Union[StructType, str]:  # type: ignore
        """
        This class overrides _fit and will not call _out_schema.
        """
        pass

    def _get_cuml_fit_func(self, dataset: DataFrame) -> Callable[  # type: ignore
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        """
        This class overrides _fit and will not call _get_cuml_fit_func.
        """
        pass

    # for the following 4 methods leave doc string as below so that they are filtered out from api docs
    def write(self) -> MLWriter:
        """Unsupported."""
        raise NotImplementedError(
            "ApproximateNearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def read(cls) -> MLReader:
        """Unsupported."""
        raise NotImplementedError(
            "ApproximateNearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def load(cls, path: str) -> MLReader:
        """Unsupported."""
        raise NotImplementedError(
            "ApproximateNearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    def save(self, path: str) -> None:
        """Unsupported."""
        raise NotImplementedError(
            "ApproximateNearestNeighbors does not support saving/loading, just re-create the estimator."
        )


class ApproximateNearestNeighborsModel(
    ApproximateNearestNeighborsClass, _NNModelBase, _ApproximateNearestNeighborsParams
):
    def __init__(
        self,
        item_df_withid: DataFrame,
    ):
        super().__init__()

        self._item_df_withid = item_df_withid

        self.bcast_qids: Optional[Broadcast] = None
        self.bcast_qfeatures: Optional[Broadcast] = None

    def _out_schema(self) -> Union[StructType, str]:  # type: ignore
        return f"query_{self._getIdColOrDefault()} long, indices array<long>, distances array<float>"

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[DataFrame, List[str], bool, List[str]]:

        dataset, select_cols, input_is_multi_cols, tmp_cols = super()._pre_process_data(
            dataset
        )

        id_col_name = self._getIdColOrDefault()
        dataset = dataset.withColumnRenamed(id_col_name, alias.row_number)

        select_cols.append(alias.row_number)

        return dataset, select_cols, input_is_multi_cols, tmp_cols

    # TODO: should we support dtype?
    def _broadcast_as_nparray(
        self,
        query_df_withid: DataFrame,
        dtype: Union[str, np.dtype] = "float32",
        BROADCAST_LIMIT: int = 8 << 30,
    ) -> Tuple[Broadcast, Broadcast]:
        """
        broadcast idCol and inputCol/inputCols of a query_df
        the broadcast splits an array by the BROADCAST_LIMIT bytes
        """

        query_df_withid, select_cols, input_is_multi_cols, tmp_cols = (
            self._pre_process_data(query_df_withid)
        )
        query_id_pd = query_df_withid.select(*select_cols).toPandas()

        id_col = alias.row_number
        query_ids = query_id_pd[id_col].to_numpy()  # type: ignore
        query_pd = query_id_pd.drop(id_col, axis=1)  # type: ignore

        if input_is_multi_cols:
            assert len(query_pd.columns) == len(self.getInputCols())
            query_features = query_pd.to_numpy()
        else:
            assert len(query_pd.columns) == 1
            query_features = np.array(list(query_pd[query_pd.columns[0]]), dtype=dtype)

        bcast_qids = _get_spark_session().sparkContext.broadcast(query_ids)
        bcast_qfeatures = _get_spark_session().sparkContext.broadcast(query_features)

        return (bcast_qids, bcast_qfeatures)

    @classmethod
    def _agg_topk(
        cls: Type["ApproximateNearestNeighborsModel"],
        knn_df: DataFrame,
        id_col_name: str,
        indices_col_name: str,
        distances_col_name: str,
        k: int,
        ascending: bool = True,
    ) -> DataFrame:
        if knn_df.rdd.getNumPartitions() == 1:
            return knn_df

        from pyspark.sql.functions import (
            arrays_zip,
            col,
            collect_list,
            desc,
            explode,
            row_number,
            slice,
            sort_array,
            struct,
        )

        zip_df = knn_df.select(
            id_col_name,
            explode(arrays_zip(distances_col_name, indices_col_name)).alias("zipped"),
        )
        topk_df = zip_df.groupBy(id_col_name).agg(
            slice(sort_array(collect_list("zipped"), asc=ascending), 1, k).alias(
                "zipped"
            )
        )
        global_knn_df = topk_df.select(
            id_col_name,
            col(f"zipped.{indices_col_name}").alias(indices_col_name),
            col(f"zipped.{distances_col_name}").alias(distances_col_name),
        )

        return global_knn_df

    @classmethod
    def _cal_cagra_params_and_check(
        cls, algoParams: Optional[Dict[str, Any]], metric: str, topk: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert (
            metric == "sqeuclidean"
        ), "when using 'cagra' algorithm, the metric must be explicitly set to 'sqeuclidean'."

        cagra_index_params: Dict[str, Any] = {"metric": metric}
        cagra_search_params: Dict[str, Any] = {}

        if algoParams is None:
            return (cagra_index_params, cagra_search_params)

        for p in algoParams:
            if p in {
                "intermediate_graph_degree",
                "graph_degree",
                "build_algo",
                "compression",
            }:
                cagra_index_params[p] = algoParams[p]
            else:
                cagra_search_params[p] = algoParams[p]

        # check cagra params
        itopk_size = (
            64
            if "itopk_size" not in cagra_search_params
            else cagra_search_params["itopk_size"]
        )
        internal_topk_size = math.ceil(itopk_size / 32) * 32
        if internal_topk_size < topk:
            raise ValueError(
                f"cagra increases itopk_size to be closest multiple of 32 and expects the value, i.e. {internal_topk_size}, to be larger than or equal to k, i.e. {topk})."
            )

        return (cagra_index_params, cagra_search_params)

    @classmethod
    def _cal_cuvs_ivf_flat_params_and_check(
        cls, algoParams: Optional[Dict[str, Any]], metric: str, topk: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ivfflat_index_params: Dict[str, Any] = {"metric": metric}
        ivfflat_search_params: Dict[str, Any] = {}

        # support both cuml names (nlist, nprobe) and cuvs names (n_lists, n_probes)
        if algoParams is not None:
            for p in algoParams:
                if p in {"n_probes", "nprobe"}:
                    ivfflat_search_params["n_probes"] = algoParams[p]
                elif p in {"n_lists", "nlist"}:
                    ivfflat_index_params["n_lists"] = algoParams[p]
                else:
                    ivfflat_index_params[p] = algoParams[p]

        return (ivfflat_index_params, ivfflat_search_params)

    @classmethod
    def _cal_cuvs_ivf_pq_params_and_check(
        cls, algoParams: Optional[Dict[str, Any]], metric: str, topk: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pq_index_params: Dict[str, Any] = {"metric": metric}
        pq_search_params: Dict[str, Any] = {}

        if algoParams is not None:
            for p in algoParams:
                if p in {"n_probes", "nprobe"}:
                    pq_search_params["n_probes"] = algoParams[p]
                elif p in {"lut_dtype", "internal_distance_dtype"}:
                    pq_search_params[p] = algoParams[p]
                elif p in {"n_lists", "nlist"}:
                    pq_index_params["n_lists"] = algoParams[p]
                elif p in {"M", "pq_dim"}:
                    pq_index_params["pq_dim"] = algoParams[p]
                elif p in {"n_bits", "pq_bits"}:
                    pq_index_params["pq_bits"] = algoParams[p]
                else:
                    pq_index_params[p] = algoParams[p]

        return (pq_index_params, pq_search_params)

    def kneighbors(
        self, query_df: DataFrame, sort_knn_df_by_query_id: bool = True
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Return the approximate nearest neighbors for each query in query_df. The data
        vectors (or equivalently item vectors) should be provided through the fit
        function (see Examples in the spark_rapids_ml.knn.ApproximateNearestNeighbors). The
        distance measure here is euclidean distance and the number of target approximate
        nearest neighbors can be set through setK(). The function currently only
        supports float32 type and will convert other data types into float32.

        Parameters
        ----------
        query_df: pyspark.sql.DataFrame
            query vectors where each row corresponds to one query. The query_df can be in the
            format of a single array column, a single vector column, or multiple float columns.

        sort_knn_df_by_query_id: bool (default=True)
            whether to sort the returned dataframe knn_df by query_id

        Returns
        -------
        query_df: pyspark.sql.DataFrame
            the query_df itself if it has an id column set through setIdCol(). If not,
            a monotonically increasing id column will be added.

        item_df: pyspark.sql.DataFrame
            the item_df (or equivalently data_df) itself if it has an id column set
            through setIdCol(). If not, a monotonically increasing id column will be added.

        knn_df: pyspark.sql.DataFrame
            the result k approximate nearest neighbors (ANNs) dataframe that has three
            columns (id, indices, distances). Each row of knn_df corresponds to the k-ANNs
            result of a query vector, identified by the id column. The indices/distances
            column stores the ids/distances of knn item_df vectors.
        """

        query_df_withid = self._ensureIdCol(query_df)
        self.bcast_qids, self.bcast_qfeatures = self._broadcast_as_nparray(
            query_df_withid
        )

        item_npartitions_before = self._item_df_withid.rdd.getNumPartitions()
        knn_df = self._transform_evaluate_internal(
            self._item_df_withid, schema=self._out_schema()
        )
        k = self.getK()

        query_id_col_name = f"query_{self._getIdColOrDefault()}"

        ascending = False if self.getMetric() == "inner_product" else True

        knn_df_agg = self.__class__._agg_topk(
            knn_df,
            query_id_col_name,
            "indices",
            "distances",
            k,
            ascending,
        )

        knn_df_returned = (
            knn_df_agg
            if sort_knn_df_by_query_id is False
            else knn_df_agg.sort(query_id_col_name)
        )
        return (self._item_df_withid, query_df_withid, knn_df_returned)

    def _concate_pdf_batches(self) -> bool:
        return True

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:

        cuml_alg_params = self.cuml_params.copy()
        assert cuml_alg_params["metric"] in {
            "euclidean",
            "sqeuclidean",
            "inner_product",
            "l2",
            "cosine",
        }

        if (
            cuml_alg_params["algorithm"] != "brute"
        ):  # brute links to CPUNearestNeighborsModel of benchmark.bench_nearest_neighbors
            if cuml_alg_params["algorithm"] == "cagra":
                check_fn = self._cal_cagra_params_and_check
            elif cuml_alg_params["algorithm"] in {"ivf_flat", "ivfflat"}:
                check_fn = self._cal_cuvs_ivf_flat_params_and_check
            else:
                assert cuml_alg_params["algorithm"] in {"ivf_pq", "ivfpq"}
                check_fn = self._cal_cuvs_ivf_pq_params_and_check

            index_params, search_params = check_fn(
                algoParams=self.cuml_params["algo_params"],
                metric=self.cuml_params["metric"],
                topk=cuml_alg_params["n_neighbors"],
            )

        def _construct_sgnn() -> CumlT:

            if cuml_alg_params["algorithm"] in {"ivf_pq", "ivfpq"}:
                from cuvs.neighbors import ivf_pq

                return ivf_pq
            elif cuml_alg_params["algorithm"] in {"ivfflat" or "ivf_flat"}:
                from cuvs.neighbors import ivf_flat

                return ivf_flat
            else:
                assert cuml_alg_params["algorithm"] == "cagra"
                from cuvs.neighbors import cagra

                return cagra

        row_number_col = alias.row_number
        input_col, input_cols = self._get_input_columns()
        assert input_col is not None or input_cols is not None
        id_col_name = self._getIdColOrDefault()

        bcast_qids = self.bcast_qids
        bcast_qfeatures = self.bcast_qfeatures

        assert bcast_qids is not None and bcast_qfeatures is not None

        logging_class_name = _get_class_or_callable_name(self.__class__)

        def _transform_internal(
            nn_object: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:

            item_row_number = df[row_number_col].to_numpy(dtype=np.int64)
            item = df.drop(row_number_col, axis=1)  # type: ignore
            if input_col is not None:
                assert len(item.columns) == 1
                item = np.array(list(item[item.columns[0]]), order="C")

            if len(item) == 0 or len(bcast_qfeatures.value) == 0:
                res = pd.DataFrame(
                    {
                        f"query_{id_col_name}": pd.Series(dtype="int64"),
                        "indices": pd.Series(dtype="object"),
                        "distances": pd.Series(dtype="object"),
                    }
                )
                return res

            import cupy as cp
            from pyspark import TaskContext

            ctx = TaskContext.get()
            pid = ctx.partitionId() if ctx is not None else -1

            logger = get_logger(logging_class_name)
            logger.info(f"partition {pid} starts with {len(item)} item vectors")
            import time

            start_time = time.time()

            if not inspect.ismodule(
                nn_object
            ):  # derived class (e.g. benchmark.bench_nearest_neighbors.CPUNearestNeighborsModel)
                nn_object.fit(item)
            else:  # cuvs ivf_flat or cagra
                build_params = nn_object.IndexParams(**index_params)

                # cuvs does not take pd.DataFrame as input
                if isinstance(item, pd.DataFrame):
                    item = cp.array(item.to_numpy(), order="C", dtype="float32")
                if isinstance(item, np.ndarray):
                    item = cp.array(item, dtype="float32")

                try:
                    index_obj = nn_object.build(build_params, item)
                except Exception as e:
                    if "k must be less than topk::kMaxCapacity (256)" in str(e):
                        from cuvs.neighbors import cagra

                        assert nn_object == cagra
                        assert (
                            "build_algo" not in index_params
                            or index_params["build_algo"] == "ivf_pq"
                        )

                        intermediate_graph_degree = (
                            build_params.intermediate_graph_degree
                        )
                        assert intermediate_graph_degree >= 256

                        error_msg = f"cagra with ivf_pq build_algo expects intermediate_graph_degree ({intermediate_graph_degree}) to be smaller than 256"
                        raise ValueError(error_msg)
                    else:
                        raise e

            logger.info(
                f"partition {pid} indexing finished in {time.time() - start_time} seconds."
            )

            start_time = time.time()

            if not inspect.ismodule(
                nn_object
            ):  # derived class (e.g. benchmark.bench_nearest_neighbors.CPUNearestNeighborsModel)
                distances, indices = nn_object.kneighbors(bcast_qfeatures.value)
            else:  # cuvs ivf_flat cagra ivf_pq
                gpu_qfeatures = cp.array(
                    bcast_qfeatures.value, order="C", dtype="float32"
                )

                assert cuml_alg_params["n_neighbors"] <= len(
                    item
                ), "k is larger than the number of item vectors on a GPU. Please increase the dataset size or use less GPUs"

                distances, indices = nn_object.search(
                    nn_object.SearchParams(**search_params),
                    index_obj,
                    gpu_qfeatures,
                    cuml_alg_params["n_neighbors"],
                )

                if cuml_alg_params["algorithm"] in {"ivf_pq", "ivfpq"}:
                    from cuvs.neighbors import refine

                    distances, indices = refine(
                        dataset=item,
                        queries=gpu_qfeatures,
                        candidates=indices,
                        k=cuml_alg_params["n_neighbors"],
                        metric=cuml_alg_params["metric"],
                    )

                distances = cp.asarray(distances)
                indices = cp.asarray(indices)

                # in case refine API reset inf distances to 0.
                if cuml_alg_params["algorithm"] in {"ivf_pq", "ivfpq"}:
                    distances[indices >= len(item)] = float("inf")

                    # for the case top-1 nn got filled into indices
                    top1_ind = indices[:, 0]
                    rest_indices = indices[:, 1:]
                    rest_distances = distances[:, 1:]
                    rest_distances[rest_indices == top1_ind[:, cp.newaxis]] = float(
                        "inf"
                    )

            if isinstance(distances, cp.ndarray):
                distances = distances.get()

            # in case a query did not probe any items, indices are filled with int64 max and distances are filled with inf
            item_row_number = np.append(item_row_number, np.iinfo("int64").max)
            if isinstance(indices, cp.ndarray):
                indices[indices >= len(item)] = len(item)
                indices = indices.get()

            indices_global = item_row_number[indices]

            logger.info(
                f"partition {pid} search finished in {time.time() - start_time} seconds."
            )

            res = pd.DataFrame(
                {
                    f"query_{id_col_name}": bcast_qids.value,
                    "indices": list(indices_global),
                    "distances": list(distances),
                }
            )
            return res

        return _construct_sgnn, _transform_internal, None

    def approxSimilarityJoin(
        self,
        query_df: DataFrame,
        distCol: str = "distCol",
    ) -> DataFrame:
        """
        This function returns the k approximate nearest neighbors (k-ANNs) in item_df of each query vector in query_df.
        item_df is the dataframe passed to the fit function of the ApproximateNearestNeighbors estimator.
        Note that the knn relationship is asymmetric with respect to the input datasets (e.g., if x is a ann of y
        , y is not necessarily a ann of x).

        Parameters
        ----------
        query_df: pyspark.sql.DataFrame
            the query_df dataframe. Each row represents a query vector.

        distCol: str
            the name of the output distance column

        Returns
        -------
        knnjoin_df: pyspark.sql.DataFrame
            the result dataframe that has three columns (item_df, query_df, distCol).
            item_df column is of struct type that includes as fields all the columns of input item dataframe.
            Similarly, query_df column is of struct type that includes as fields all the columns of input query dataframe.
            distCol is the distance column. A row in knnjoin_df is in the format (v1, v2, dist(v1, v2)),
            where item_vector v1 is one of the k nearest neighbors of query_vector v2 and their distance is dist(v1, v2).
        """

        return self._nearest_neighbors_join(query_df, distCol)

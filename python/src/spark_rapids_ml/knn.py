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
from abc import ABCMeta
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
from .params import HasIDCol, P, _CumlClass, _CumlParams
from .utils import _concat_and_free, _get_spark_session, get_logger


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
        self._setDefault(idCol=alias.row_number)

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
        if not self.isSet("idCol") and self.getIdCol() in df.columns:
            raise ValueError(
                f"Cannot create a default id column since a column with the default name '{self.getIdCol()}' already exists."
                + "Please specify an id column"
            )

        id_col_name = self.getIdCol()
        df_withid = (
            df
            if self.isSet("idCol")
            else df.select(monotonically_increasing_id().alias(id_col_name), "*")
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

    idCol: str
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
        raise NotImplementedError(
            "NearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def read(cls) -> MLReader:
        raise NotImplementedError(
            "NearestNeighbors does not support saving/loading, just re-create the estimator."
        )


class NearestNeighborsModel(
    _CumlCaller, _CumlModel, NearestNeighborsClass, _NearestNeighborsCumlParams
):
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
                    f"query_{self.getIdCol()}", ArrayType(LongType(), False), False
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

        if self.hasParam("idCol") and self.isDefined("idCol"):
            id_col_name = self.getOrDefault("idCol")
            select_cols.append(col(id_col_name).alias(alias.row_number))
        else:
            select_cols.append(col(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type

    def kneighbors(self, query_df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
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

        union_df = self._processed_item_df.union(processed_query_df)

        pipelinedrdd = self._call_cuml_fit_func(union_df, partially_collect=False)
        pipelinedrdd = pipelinedrdd.repartition(query_default_num_partitions)  # type: ignore

        query_id_col_name = f"query_{self.getIdCol()}"
        id_col_type = dict(union_df.dtypes)[self.getIdCol()]
        knn_rdd = pipelinedrdd.flatMap(
            lambda row: list(
                zip(row[query_id_col_name], row["indices"], row["distances"])
            )
        )
        knn_df = knn_rdd.toDF(
            schema=f"{query_id_col_name} {id_col_type}, indices array<{id_col_type}>, distances array<float>"
        ).sort(query_id_col_name)

        return (self._item_df_withid, query_df_withid, knn_df)

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
        id_col_name = self.getIdCol()

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

    def _transform(self, dataset: DataFrame) -> DataFrame:
        raise NotImplementedError(
            "NearestNeighborsModel does not provide a transform function. Use 'kneighbors' instead."
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

        id_col_name = self.getIdCol()

        # call kneighbors then prepare return results
        (item_df_withid, query_df_withid, knn_df) = self.kneighbors(query_df)

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
        raise NotImplementedError(
            "NearestNeighborsModel does not support saving/loading, just re-fit the estimator to re-create a model."
        )

    @classmethod
    def read(cls) -> MLReader:
        raise NotImplementedError(
            "NearestNeighborsModel does not support loading/loading, just re-fit the estimator to re-create a model."
        )


class ApproximateNearestNeighborsClass(_CumlClass):

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {
            "k": "n_neighbors",
            "algorithm": "algorithm",
            "algo_params": "algo_params",
        }

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {"n_neighbors": 5, "verbose": False, "algorithm": "ivfflat"}

    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None


class DictTypeConverters(TypeConverters):
    @staticmethod
    def _toDict(value: Any) -> Dict[str, Any]:
        """
        Convert a value to a Dict type for Param typeConverter, if possible.
        """
        if isinstance(value, Dict):
            return {TypeConverters.toString(k): v for k, v in value.items()}
        raise TypeError("Could not convert %s to Dict[str, Any]" % value)


class _ApproximateNearestNeighborsParams(_NearestNeighborsCumlParams):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault(algorithm="ivfflat")
        self._setDefault(algo_params=None)

    algorithm = Param(
        Params._dummy(),
        "algorithm",
        "The algorithm to use for approximate nearest neighbors search.",
        typeConverter=TypeConverters.toString,
    )

    algo_params = Param(
        Params._dummy(),
        "algo_params",
        "The parameters to use to set up a neighbor algorithm.",
        typeConverter=DictTypeConverters._toDict,
    )


class ApproximateNearestNeighbors(
    ApproximateNearestNeighborsClass, _CumlEstimator, _ApproximateNearestNeighborsParams
):
    """
    IVF_FLAT retrieves the k approximate nearest neighbors in item vectors for each query
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: Optional[int] = None,
        algorithm: str = "ivfflat",
        algo_params: Optional[Dict[str, Any]] = None,
        inputCol: Optional[Union[str, List[str]]] = None,
        idCol: Optional[str] = None,
        num_workers: Optional[int] = None,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert algorithm in {"brute", "ivfflat"}
        self._set_params(**self._input_kwargs)

    def _fit(self, item_df: DataFrame) -> "ApproximateNearestNeighborsModel":  # type: ignore
        self._item_df_withid = self._ensureIdCol(item_df)

        # TODO: should test this at scale to see if/when we hit limits
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

    def write(self) -> MLWriter:
        raise NotImplementedError(
            "ApproximateNearestNeighbors does not support saving/loading, just re-create the estimator."
        )

    @classmethod
    def read(cls) -> MLReader:
        raise NotImplementedError(
            "ApproximateNearestNeighbors does not support saving/loading, just re-create the estimator."
        )


class ApproximateNearestNeighborsModel(
    ApproximateNearestNeighborsClass, _CumlModel, _ApproximateNearestNeighborsParams
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
        return (
            f"query_{self.getIdCol()} long, indices array<long>, distances array<float>"
        )

    def write(self) -> MLWriter:
        raise NotImplementedError(
            "ApproximateNearestNeighborsModel does not support saving/loading, just re-fit the estimator to re-create a model."
        )

    @classmethod
    def read(cls) -> MLReader:
        raise NotImplementedError(
            "ApproximateNearestNeighborsModel does not support loading/loading, just re-fit the estimator to re-create a model."
        )

    def _pre_process_data(
        self, dataset: DataFrame
    ) -> Tuple[DataFrame, List[str], bool, List[str]]:

        dataset, select_cols, input_is_multi_cols, tmp_cols = super()._pre_process_data(
            dataset
        )

        if self.hasParam("idCol") and self.isDefined("idCol"):
            id_col_name = self.getOrDefault("idCol")
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

        # def _split_by_bytes(arr: np.ndarray) -> List[np.ndarray]:
        #     if arr.nbytes <= BROADCAST_LIMIT:
        #         return [arr]
        #     rows_per_chunk = BROADCAST_LIMIT // arr.itemsize
        #     num_chunks = (arr.shape[0] + rows_per_chunk - 1) // rows_per_chunk
        #     return np.array_split(arr, num_chunks)

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
    ) -> DataFrame:
        from pyspark.sql.functions import pandas_udf

        @pandas_udf("array<long>")  # type: ignore
        def func_agg_indices(indices: pd.Series, distances: pd.Series) -> list[int]:
            flat_indices = indices.explode().reset_index(drop=True)
            flat_distances = (
                distances.explode().reset_index(drop=True).astype("float32")
            )
            assert len(flat_indices) == len(flat_distances)
            topk_index = flat_distances.nsmallest(k).index
            res = flat_indices[topk_index].to_numpy()
            return res

        @pandas_udf("array<float>")  # type: ignore
        def func_agg_distances(distances: pd.Series) -> list[float]:
            flat_distances = (
                distances.explode().reset_index(drop=True).astype("float32")
            )
            res = flat_distances.nsmallest(k).to_numpy()
            return res

        res_df = knn_df.groupBy(id_col_name).agg(
            func_agg_indices(
                knn_df[indices_col_name], knn_df[distances_col_name]
            ).alias(indices_col_name),
            func_agg_distances(knn_df[distances_col_name]).alias(distances_col_name),
        )

        return res_df

    def kneighbors(self, query_df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Return the approximate nearest neighbors for each query in query_df."""

        query_df_withid = self._ensureIdCol(query_df)
        self.bcast_qids, self.bcast_qfeatures = self._broadcast_as_nparray(
            query_df_withid
        )

        knn_df = self._transform_evaluate_internal(
            self._item_df_withid, schema=self._out_schema()
        )
        k = self.getK()

        query_id_col_name = f"query_{self.getIdCol()}"
        knn_df_agg = self.__class__._agg_topk(
            knn_df, query_id_col_name, "indices", "distances", k
        )

        return (self._item_df_withid, query_df_withid, knn_df_agg)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        raise NotImplementedError(
            "ApproximateNearestNeighborsModel does not provide a transform function. Use 'kneighbors' instead."
        )

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:

        cuml_alg_params = self.cuml_params.copy()

        def _construct_sgnn() -> CumlT:

            from cuml.neighbors import NearestNeighbors as SGNN

            nn_object = SGNN(output_type="cupy", **cuml_alg_params)

            return nn_object

        row_number_col = alias.row_number
        input_col, input_cols = self._get_input_columns()
        assert input_col is not None or input_cols is not None
        id_col_name = self.getIdCol()

        bcast_qids = self.bcast_qids
        bcast_qfeatures = self.bcast_qfeatures

        assert bcast_qids is not None and bcast_qfeatures is not None

        def _transform_internal(
            nn_object: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:

            item_row_number = df[row_number_col].to_numpy()
            item = df.drop(row_number_col, axis=1)  # type: ignore
            if input_col is not None:
                assert len(item.columns) == 1
                item = np.array(list(item[item.columns[0]]), order="C")

            if len(item) == 0:
                return pd.DataFrame(
                    {
                        "query_{id_col_name}": [],
                        "indices": [],
                        "distances": [],
                    }
                )

            nn_object.fit(item)
            import cupy as cp

            distances, indices = nn_object.kneighbors(bcast_qfeatures.value)

            if cuml_alg_params["algorithm"] == "ivfflat":
                distances = distances * distances

            indices = indices.get()
            indices_global = item_row_number[indices]

            res = pd.DataFrame(
                {
                    f"query_{id_col_name}": bcast_qids.value,
                    "indices": list(indices_global),
                    "distances": list(distances.get()),
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
        TODO: merge code with NearestNeighborsModel class
        """

        id_col_name = self.getIdCol()

        # call kneighbors then prepare return results
        (item_df_withid, query_df_withid, knn_df) = self.kneighbors(query_df)

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

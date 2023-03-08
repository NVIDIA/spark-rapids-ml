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

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import cudf
import numpy as np
import pandas as pd
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
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    Row,
    StructField,
    StructType,
)

from spark_rapids_ml.core import (
    INIT_PARAMETERS_NAME,
    CumlInputType,
    CumlT,
    _CumlEstimatorSupervised,
    _CumlModel,
    alias,
)
from spark_rapids_ml.params import _CumlClass, _CumlParams


class NearestNeighborsClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml import NearestNeighbors as cumlNearestNeighbors
        from cuml.neighbors.nearest_neighbors_mg import (
            NearestNeighborsMG,  # to include the batch_size parameter that exists in the MG class
        )

        return [cumlNearestNeighbors, NearestNeighborsMG]

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {"k": "n_neighbors"}

    @classmethod
    def _param_excludes(cls) -> List[str]:
        return [
            "handle",
            "algorithm",
            "metric",
            "p",
            "algo_params",
            "metric_expanded",
            "metric_params",
            "output_type",
        ]


class _NearestNeighborsCumlParams(_CumlParams, HasInputCol, HasLabelCol, HasInputCols):
    """
    Shared Spark Params for NearestNeighbor and NearestNeighborModel.
    """

    k = Param(
        Params._dummy(),
        "k",
        "The number nearest neighbors to retrieve. Must be >= 1.",
        typeConverter=TypeConverters.toInt,
    )

    id_col = Param(
        Params._dummy(),
        "id_col",
        "id column name.",
        typeConverter=TypeConverters.toString,
    )

    def setInputCol(
        self, value: Union[str, List[str]]
    ) -> "_NearestNeighborsCumlParams":
        """
        Sets the value of :py:attr:`inputCol` or :py:attr:`inputCols`. Used when input vectors are stored in a single column.
        """
        if isinstance(value, str):
            self.set_params(inputCol=value)
        else:
            self.set_params(inputCols=value)
        return self

    def setInputCols(self, value: List[str]) -> "_NearestNeighborsCumlParams":
        """
        Sets the value of :py:attr:`inputCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self.set_params(inputCols=value)

    def setIdCol(self, value: str) -> "_NearestNeighborsCumlParams":
        """
        Sets the value of `id_col`. If not set, an id column will be added with column name `id`. The id column is used to specify nearest neighbor vectors by associated id value.
        """
        self.set_params(id_col=value)
        return self

    def getIdCol(self) -> str:
        """
        Gets the value of `id_col`.
        """
        col_name = (
            self.getOrDefault("id_col")
            if self.isDefined("id_col")
            else alias.row_number
        )
        return col_name


class _CumlEstimatorNearestNeighbors(
    _CumlEstimatorSupervised, _NearestNeighborsCumlParams
):
    """
    Base class for Cuml Nearest Neighbor.
    """

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
        ) = super()._pre_process_data(dataset)

        # if input format is vectorUDT, convert data type from float64 to float32
        input_col, _ = self._get_input_columns()
        if input_col is not None and isinstance(
            dataset.schema[input_col].dataType, VectorUDT
        ):
            select_cols[0] = vector_to_array(col(input_col), dtype="float32").alias(
                alias.data
            )

        if self.hasParam("id_col") and self.isDefined("id_col"):
            id_col_name = self.getOrDefault("id_col")
            select_cols.append(col(id_col_name).alias(alias.row_number))
        else:
            select_cols.append(col(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type


class NearestNeighbors(
    NearestNeighborsClass, _CumlEstimatorNearestNeighbors, _NearestNeighborsCumlParams
):
    """
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
    >>> gpu_knn.fit(data_df)
    NearestNeighbors_084799c72508
    >>> (query_df, data_df, knn_df) = gpu_knn.kneighbors(query_df)
    >>> knn_df.show()
    +--------+-------+----------------+
    |query_id|indices|       distances|
    +--------+-------+----------------+
    |       3| [0, 1]|[0.0, 1.4142135]|
    |       4| [2, 1]|[0.0, 1.4142135]|
    +--------+-------+----------------+
    >>> query_df.show()
    +---+----------+
    | id|  features|
    +---+----------+
    |  3|[1.0, 1.0]|
    |  4|[3.0, 3.0]|
    +---+----------+
    >>> data_df.show()
    +---+----------+
    | id|  features|
    +---+----------+
    |  0|[1.0, 1.0]|
    |  1|[2.0, 2.0]|
    |  2|[3.0, 3.0]|
    +---+----------+
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)
        self.label_isdata = 0
        self.label_isquery = 1
        self.row_number_col = alias.row_number
        self.set_params(labelCol=alias.label)

    def setK(self, value: int) -> "NearestNeighbors":
        """
        Sets the value of `k`.
        """
        self.set_params(k=value)
        return self

    def _create_pyspark_model(self, result: Row) -> "NearestNeighborsModel":
        return NearestNeighborsModel.from_row(result)

    def fit(self, dataset: DataFrame, params: Optional[Dict[Param[Any], Any]] = None) -> "NearestNeighbors":  # type: ignore
        self.item_df = dataset
        if not self.isDefined("id_col"):
            self.item_df = self._df_zip_with_index(self.item_df)

        self.processed_item_df = self.item_df.withColumn(
            alias.label, lit(self.label_isdata)
        )
        return self

    def kneighbors(self, query_df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        query_default_num_partitions = query_df.rdd.getNumPartitions()
        if not self.isDefined("id_col"):
            query_df = self._df_zip_with_index(query_df)
        processed_query_df = query_df.withColumn(alias.label, lit(self.label_isquery))

        union_df = self.processed_item_df.union(processed_query_df)

        pipelinedrdd = self._fit(union_df)
        pipelinedrdd = pipelinedrdd.repartition(query_default_num_partitions)  # type: ignore
        knn_rdd = pipelinedrdd.flatMap(
            lambda row: list(zip(row["query_id"], row["indices"], row["distances"]))
        )
        knn_df = knn_rdd.toDF(
            schema=f"query_{self.getIdCol()} int, indices array<int>, distances array<float>"
        ).sort(f"query_{self.getIdCol()}")

        return (query_df, self.item_df, knn_df)

    def _df_zip_with_index(self, df: DataFrame) -> DataFrame:
        """
        Add an row number column (or equivalently id column) to df using zipWithIndex. Used when id_col is not set.
        TODO: May replace zipWithIndex with monotonically_increasing_id if row number does not have to consecutive.
        """
        out_schema = StructType(
            [StructField(self.row_number_col, IntegerType(), False)] + df.schema.fields
        )
        zipped_rdd = df.rdd.zipWithIndex()
        new_rdd = zipped_rdd.map(lambda row: [row[1]] + list(row[0]))
        return new_rdd.toDF(schema=out_schema)

    def _require_ucx(self) -> bool:
        return True

    def _return_model(self) -> bool:
        return False

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("query_id", ArrayType(IntegerType(), False), False),
                StructField(
                    "distances", ArrayType(ArrayType(DoubleType(), False), False), False
                ),
                StructField(
                    "indices", ArrayType(ArrayType(IntegerType(), False), False), False
                ),
            ]
        )

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[CumlInputType, Dict[str, Any]], Dict[str, Any],]:

        label_isdata = self.label_isdata

        def _cuml_fit(
            dfs: CumlInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:

            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            rank = context.partitionId()

            from cuml.neighbors.nearest_neighbors_mg import NearestNeighborsMG as cumlNN

            nn_object = cumlNN(
                handle=params["handle"],
                n_neighbors=params[INIT_PARAMETERS_NAME]["n_neighbors"],
                output_type="numpy",
                verbose=params[INIT_PARAMETERS_NAME]["verbose"],
            )

            item_list = []
            query_list = []
            item_row_number = []
            query_row_number = []

            for x_array, label_array, row_number_array in dfs:
                item_filter = [
                    True if label_array[i] == label_isdata else False  # type: ignore
                    for i in range(len(x_array))
                ]
                query_filter = [
                    False if label_array[i] == label_isdata else True  # type: ignore
                    for i in range(len(x_array))
                ]

                item_list.append(x_array[item_filter])
                query_list.append(x_array[query_filter])

                item_row_number += row_number_array[item_filter].tolist()  # type: ignore
                query_row_number += row_number_array[query_filter].tolist()  # type: ignore

            if isinstance(item_list[0], pd.DataFrame):
                item = [pd.concat(item_list)]
                query = [pd.concat(query_list)]
            else:
                item = [np.concatenate(item_list)]
                query = [np.concatenate(query_list)]

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

            messages = params["loop"].run_until_complete(
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
                ncols=params["n"],
                rank=rank,
                n_neighbors=params[INIT_PARAMETERS_NAME]["n_neighbors"],
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
                "query_id": query_row_number,
                "distances": distances,
                "indices": transformed_indices,
            }

        return _cuml_fit


class NearestNeighborsModel(
    NearestNeighborsClass, _CumlModel, _NearestNeighborsCumlParams
):
    def __init__(
        self,
        query_index: List[int],
        distances: List[List[float]],
        indices: List[List[int]],
    ):
        super().__init__(
            query_index=query_index,
            distances=distances,
            indices=indices,
        )

        cumlParams = NearestNeighbors._get_cuml_params_default()
        self.set_params(**cumlParams)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:  # type: ignore
        pass

    def _get_cuml_transform_func(  # type: ignore
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        pass

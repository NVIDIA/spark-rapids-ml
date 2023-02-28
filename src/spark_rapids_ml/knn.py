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

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import cudf
import numpy as np
import pandas as pd
from pyspark.ml.param.shared import (
    HasInputCol,
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
    _set_pyspark_cuml_cls_param_attrs,
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


class _NearestNeighborsCumlParams(_CumlParams, HasInputCol, HasLabelCol):
    """
    Shared Spark Params for NearestNeighbor and NearestNeighborModel.
    """

    k = Param(
        Params._dummy(),
        "k",
        "The number nearest neighbors to retrieve. Must be >= 1.",
        typeConverter=TypeConverters.toInt,
    )

    def setInputCol(self, value: str) -> "_NearestNeighborsCumlParams":
        """
        Sets the value of `inputCol`.
        """
        self.set_params(inputCol=value)
        return self


class _CumlEstimatorNearestNeighbors(_CumlEstimatorSupervised):
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

        select_cols.append(col(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type


class NearestNeighbors(
    NearestNeighborsClass, _CumlEstimatorNearestNeighbors, _NearestNeighborsCumlParams
):
    """
    Examples
    --------
    >>> from spark_rapids_ml.knn import NearestNeighbors
    >>> data = [([1.0, 1.0],),
    ...         ([2.0, 2.0],),
    ...         ([3.0, 3.0],),]
    >>> topk = 2
    >>> gpu_knn = NearestNeighbors().setInputCol("features").setK(topk)
    >>> data_df = spark.createDataFrame(data, ["features"])
    >>> gpu_knn.fit(data_df)
    >>> query = [[1.0, 1.0], [3.0, 3.0]]
    >>> query_df = spark.createDataFrame(query, ["features"])
    >>> distances_df, indices_df = gpu_knn.kneighbors(query_df)
    >>> distances_df.show()
    [0, 1.414]
    [0, 1.414]
    >>> indices_df.show()
    [0, 1]
    [2, 1]
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
        self.data_df = dataset.withColumn(alias.label, lit(self.label_isdata))
        self.data_df = self.df_zip_with_index(self.data_df)
        return self

    def kneighbors(self, query_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        query_default_num_partitions = query_df.rdd.getNumPartitions()
        query_df = query_df.withColumn(alias.label, lit(self.label_isquery))
        query_df = self.df_zip_with_index(query_df)
        union_df = self.data_df.union(query_df)

        pipelinedrdd = self._fit(union_df)
        pipelinedrdd = pipelinedrdd.repartition(query_default_num_partitions)  # type: ignore
        res_rdd = pipelinedrdd.flatMap(
            lambda row: list(zip(row["query_index"], row["indices"], row["distances"]))
        )
        res_df = res_rdd.toDF(
            schema="query_index int, indices array<int>, distances array<float>"
        )
        res_df = res_df.sort("query_index")
        distances_df = res_df.select("distances")
        indices_df = res_df.select("indices")
        return (distances_df, indices_df)

    def df_zip_with_index(self, df: DataFrame) -> DataFrame:
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
                StructField("query_index", ArrayType(IntegerType(), False), False),
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
                verbose=params[INIT_PARAMETERS_NAME]["verbose"],
            )

            item = np.empty(
                (0, params["n"]), dtype=np.float32
            )  # Cuml NN only supports np.float32
            query = np.empty((0, params["n"]), dtype=np.float32)
            item_row_number = []
            query_row_number = []
            for x_array, label_array, row_number_array in dfs:
                item_filter = [
                    True if label_array[i] == label_isdata else False
                    for i in range(len(x_array))
                ]
                query_filter = [
                    False if label_array[i] == label_isdata else True
                    for i in range(len(x_array))
                ]

                item = np.concatenate((item, x_array[item_filter]), axis=0)
                query = np.concatenate((query, x_array[query_filter]), axis=0)

                item_row_number += row_number_array[item_filter].tolist()
                query_row_number += row_number_array[query_filter].tolist()

            item = [item]
            query = [query]
            item_row_number = [item_row_number]
            query_row_number = [query_row_number]

            item_query_sizes = [len(chunk) for chunk in item] + [
                len(chunk) for chunk in query
            ]
            import json

            messages = context.allGather(
                message=json.dumps((rank, item_query_sizes, item_row_number))
            )
            triplets = [json.loads(msg) for msg in messages]

            item_parts_to_ranks = []
            query_parts_to_ranks = []
            for r, sizes, _ in triplets:
                half_len = len(sizes) // 2
                item_parts_to_ranks += [(r, size) for size in sizes[:half_len]]
                query_parts_to_ranks += [(r, size) for size in sizes[half_len:]]
            item_nrows = sum(pair[1] for pair in item_parts_to_ranks)
            query_nrows = sum(pair[1] for pair in query_parts_to_ranks)

            res_tuple: Tuple[List[np.array], List[np.array]] = nn_object.kneighbors(
                index=[item],
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

            distances: List[np.array] = res_tuple[0]
            indices: List[np.array] = res_tuple[1]

            distances = [ary.tolist() for ary in distances]
            indices = [ary.tolist() for ary in indices]
            query = [ary.tolist() for ary in query]
            item = [ary.tolist() for ary in item]

            # id mapping
            id2row = {}
            count = 0
            for r, _, item_rn in triplets:
                for chunk in item_rn:
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
                "query_index": query_row_number,
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

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        pass

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Tuple[
        Callable[..., CumlT],
        Callable[[CumlT, Union[cudf.DataFrame, np.ndarray]], pd.DataFrame],
    ]:
        pass

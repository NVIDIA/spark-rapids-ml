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

from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
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
from pyspark.ml.param.shared import HasInputCol, HasLabelCol 
import cudf

class NearestNeighborsClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml import NearestNeighbors as cumlNearestNeighbors
        return [cumlNearestNeighbors]

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

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
            "output_type"
        ]

class _NearestNeighborsCumlParams(_CumlParams, HasInputCol, HasLabelCol):
    """
    Shared Spark Params for NearestNeighbor and NearestNeighborModel.
    """

    def setInputCol(self, value: str) -> "NearestNeighbors":
        """
        Sets the value of `inputCol`.
        """
        self.set_params(inputCol=value)
        return self

    def setOutputCol(self, value: str) -> "NearestNeighbors":
        """
        Sets the value of `outputCol`.
        """
        self.set_params(outputCol=value)
        return self

class NearestNeighbors(NearestNeighborsClass, _CumlEstimatorSupervised, _NearestNeighborsCumlParams):
    """
    Examples
    --------
    >>> from sparkcuml.neighbors import NearestNeighbors
    >>> data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    >>> topk = 2
    >>> gpu_knn = NearestNeighbors().setInputCol("features").setK(topk)
    >>> data_df = spark.SparkContext.parallelize(data).map(lambda row: (row,)).toDF(["features"])
    >>> gpu_knn.fit(data_df)
    >>> query = [[1.0, 1.0], [3.0, 3.0]]
    >>> query_df = spark.SparkContext.parallelize(query).map(lambda row: (row,)).toDF(["features"])
    >>> distances_df, indices_df = gpu_knn.kneighbors(query_df, return_distance=True)
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
        self.set_params(labelCol=alias.label)

    def setK(self, value: int) -> "NearestNeighbors":
        """
        Sets the value of `k`.
        """
        self.set_params(n_neighbors=value)
        return self

    def _create_pyspark_model(self, result: Row) -> "NearestNeighborsModel":
        return NearestNeighborsModel.from_row(result)

    def fit(self, dataset: DataFrame) -> "NearestNeighbors":
        self.data_df = dataset.withColumn(alias.label, lit(self.label_isdata))
        return self

    def kneighbors(self, query_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        query_df = query_df.withColumn(alias.label, lit(self.label_isquery))
        union_df = self.data_df.union(query_df)
        pipelinedrdd = self._fit(union_df, return_model=False)
        pipelinedrdd = pipelinedrdd.repartition(self.num_workers)
        query_res_rdd = pipelinedrdd.flatMap(lambda row : list(zip(row["query_index"], row["query"], row["indices"], row["distances"])))
        data_rdd = pipelinedrdd.flatMap(lambda row : list(zip(row["item_index"], row["item"])))

        knn_df = query_res_rdd.toDF(schema="query_index int, query_features array<float>, nn_indices array<int>, nn_distances array<float>")
        data_df = data_rdd.toDF(schema="item_index int, item_features array<float>")
        return (knn_df, data_df)

    def _require_ucx(self) -> bool:
        return True

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("distances", ArrayType(ArrayType(DoubleType(), False), False), False),
                StructField("indices", ArrayType(ArrayType(IntegerType(), False), False), False),
                StructField("query", ArrayType(ArrayType(DoubleType(), False), False), False),
                StructField("query_index", ArrayType(IntegerType(), False), False),
                StructField("item", ArrayType(ArrayType(DoubleType(), False), False), False),
                StructField("item_index", ArrayType(IntegerType(), False), False),
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
                verbose = params[INIT_PARAMETERS_NAME]["verbose"],
            )

            item = []
            query = []
            for x_array, label_array in dfs:
                item_filter = [True if label_array[i] == label_isdata else False for i in range(len(x_array))]
                query_filter = [False if label_array[i] == label_isdata else True for i in range(len(x_array))]
                item.append(x_array[item_filter])
                query.append(x_array[query_filter])

            assert(len(item) == len(query))
            item_query_sizes = [len(chunk) for chunk in item] + [len(chunk) for chunk in query]

            import json
            messages = context.allGather(message=json.dumps((rank, item_query_sizes)))
            rank_sizes = [json.loads(msg) for msg in messages]
            
            item_parts_to_ranks = [] 
            query_parts_to_ranks = [] 

            for r, sizes in rank_sizes:
                half_len = len(sizes) // 2
                item_parts_to_ranks += [(r, size) for size in sizes[:half_len]]
                query_parts_to_ranks += [(r, size) for size in sizes[half_len:]]
                
            item_nrows = sum(pair[1] for pair in item_parts_to_ranks)
            query_nrows = sum(pair[1] for pair in query_parts_to_ranks)

            res_tuple: Tuple[List[np.array], List[np.array]] = nn_object.kneighbors(
                index = item, 
                index_parts_to_ranks = item_parts_to_ranks, 
                index_nrows = item_nrows,
                query = query,
                query_parts_to_ranks = query_parts_to_ranks,
                query_nrows = query_nrows,
                ncols = params['n'],
                rank = rank,
                n_neighbors = params[INIT_PARAMETERS_NAME]["n_neighbors"],
                convert_dtype = False, # only np.float32 is supported in cuml. Should set to True for all other types
            )

            distances: List[np.array] = res_tuple[0]
            indices: List[np.array] = res_tuple[1]

            distances = [ary.tolist() for ary in distances]
            indices = [ary.tolist() for ary in indices]
            query = [ary.tolist() for ary in query]
            item = [ary.tolist() for ary in item]

            query_id = []
            start = 0
            for r, s in query_parts_to_ranks:
                if r == rank:
                    query_id.append(list(range(start, start + s)))
                start += s

            item_id = []
            start = 0
            for r, s in item_parts_to_ranks:
                if r == rank:
                    item_id.append(list(range(start, start + s)))
                start += s
                    
            return {
                "distances": distances,
                "indices": indices,
                "query_index": query_id,
                "query": query,
                "item_index": item_id,
                "item": item,
            }
        return _cuml_fit

class NearestNeighborsModel(NearestNeighborsClass, _CumlModel, _NearestNeighborsCumlParams):
    def __init__(
            self,
            distances: List[List[float]],
            indices: List[List[int]],
            query_index: List[int],
            query: List[List[float]],
            item_index: List[int],
            item: List[List[float]],
        ):
            super().__init__(
                distances=distances,
                indices=indices,
                query_index = query_index,
                query=query,
                item_index=item_index,
                item=item,
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

_set_pyspark_cuml_cls_param_attrs(NearestNeighbors, NearestNeighborsModel)
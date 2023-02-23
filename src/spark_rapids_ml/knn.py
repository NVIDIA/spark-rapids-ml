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
from pyspark.sql.functions import spark_partition_id, lit
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    Row,
    StringType,
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
from spark_rapids_ml.utils import PartitionDescriptor
from spark_rapids_ml.params import _CumlClass, _CumlParams
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasLabelCol 
import cudf

class NearestNeighborsClass(_CumlClass):
    @classmethod
    def _cuml_cls(cls) -> List[type]:
        from cuml import NearestNeighbors as cumlNearestNeighbors
        return [cumlNearestNeighbors]

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}
        #return {"k": "n_neighbors"}

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

    #k: Param[int] = Param(
    #    Params._dummy(),
    #    "k",
    #    "The number of nearest neighbors to retrieve. Must be >= 1.",
    #    typeConverter=TypeConverters.toInt,
    #)

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

    def kneighbors(self, query_df: DataFrame) -> DataFrame:
        query_df = query_df.withColumn(alias.label, lit(self.label_isquery))
        union_df = self.data_df.union(query_df)
        rdd = self._fit(union_df, return_model=False)

        #print(type(rdd))
        print(rdd.collect())
        #df = rdd.toDF()
        #df.show()
        return pd.DataFrame()

    def _require_ucx(self) -> bool:
        return True

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("distances", ArrayType(ArrayType(DoubleType(), False), False), False),
                StructField("indices", ArrayType(ArrayType(IntegerType(), False), False), False),
                StructField("query", ArrayType(ArrayType(DoubleType(), False), False), False),
                StructField("index", ArrayType(ArrayType(DoubleType(), False), False), False),
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

            index = []
            query = []
            for x_array, label_array in dfs:
                index_filter = [True if label_array[i] == label_isdata else False for i in range(len(x_array))]
                query_filter = [False if label_array[i] == label_isdata else True for i in range(len(x_array))]
                index.append(x_array[index_filter])
                query.append(x_array[query_filter])

            assert(len(index) == len(query))
            index_query_sizes = [len(chunk) for chunk in index] + [len(chunk) for chunk in query]

            import json
            messages = context.allGather(message=json.dumps((rank, index_query_sizes)))
            rank_sizes = [json.loads(msg) for msg in messages]
            
            index_parts_to_ranks = [] 
            query_parts_to_ranks = [] 

            for r, sizes in rank_sizes:
                half_len = len(sizes) // 2
                index_parts_to_ranks += [(r, size) for size in sizes[:half_len]]
                query_parts_to_ranks += [(r, size) for size in sizes[half_len:]]
                
            index_nrows = sum(pair[1] for pair in index_parts_to_ranks)
            query_nrows = sum(pair[1] for pair in query_parts_to_ranks)

            print(f"rank {rank} gets index: {index}")
            print(f"rank {rank} gets index_parts_to_ranks: {index_parts_to_ranks}")
            print(f"rank {rank} gets index_nrows: {index_nrows}")
            print(f"rank {rank} gets query: {query}")
            print(f"rank {rank} gets query_parts_to_ranks: {query_parts_to_ranks}")
            print(f"rank {rank} gets query_nrows: {query_nrows}")

            res_tuple: Tuple[List[np.array], List[np.array]] = nn_object.kneighbors(
                index = index, 
                index_parts_to_ranks = index_parts_to_ranks, 
                index_nrows = index_nrows,
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
            index = [ary.tolist() for ary in index]

            return {
                "distances": distances,
                "indices": indices,
                "query": query,
                "index": index,
            }
        return _cuml_fit

class NearestNeighborsModel(NearestNeighborsClass, _CumlModel, _NearestNeighborsCumlParams):
    def __init__(
            self,
            distances: List[List[float]],
            indices: List[List[int]],
            query: List[List[float]],
            index: List[List[float]],
        ):
            super().__init__(
                distances=distances,
                indices=indices,
                query=query,
                index=index,
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
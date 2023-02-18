from typing import Any, Dict, Tuple

import numpy as np
import pytest
from pyspark import BarrierTaskContext
from sklearn.datasets import make_blobs

from spark_rapids_ml.common.cuml_context import CumlContext
from spark_rapids_ml.core import _CumlCommon
from spark_rapids_ml.knn import NearestNeighbors
from spark_rapids_ml.utils import _get_spark_session, _is_local 

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    assert_params,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    feature_types,
    idfn,
    pyspark_supported_feature_types,
)

def test_toy(gpu_number: int) -> None:
    data = [(0, [1.0, 1.0],),
            (1, [2.0, 2.0],),
            (2, [3.0, 3.0],),
            (3, [4.0, 4.0],)]

    query = [(4, [0.0, 0.0],),
             (5, [4.0, 4.0],)]


    topk = 1

    #from cuml import NearestNeighbors as cuNN
    #cu_nn = cuNN(n_neighbors=topk, output_type="numpy", verbose = 7)
    #cu_nn = cu_nn.fit(data)
    #cu_result = cu_nn.kneighbors(query) 
    #print(cu_result[0])
    #print(cu_result[1])

    data_type = np.float32
    with CleanSparkSession() as spark:
        data_type = "float" if data_type == np.float32 else "double"
        schema = f"id int, features array<{data_type}>"
        data_df = spark.createDataFrame(data, schema)
        query_df = spark.createDataFrame(query, schema)

        gpu_knn = NearestNeighbors(num_workers=gpu_number)
        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setK(topk)

        gpu_knn = gpu_knn.fit(data_df)
        res = gpu_knn.kneighbors(query_df)

    

#@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
#@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
#@pytest.mark.parametrize("data_type", cuml_supported_data_types)
#@pytest.mark.parametrize("max_record_batch", [100, 10000])
#def test_pca(
#    gpu_number: int,
#    feature_type: str,
#    data_shape: Tuple[int, int],
#    data_type: np.dtype,
#    max_record_batch: int,
#) -> None:

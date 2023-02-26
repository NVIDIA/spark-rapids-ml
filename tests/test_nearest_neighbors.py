from typing import Any, Dict, Tuple
import numpy as np
import pytest
from pyspark import BarrierTaskContext
from pyspark.sql.functions import monotonically_increasing_id 
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

def test_example(gpu_number: int) -> None:
    data = [([1.0, 1.0],), 
            ([2.0, 2.0],), 
            ([3.0, 3.0],),
            ([4.0, 4.0],),
            ([5.0, 5.0],),
            ([6.0, 6.0],),
            ([7.0, 7.0],),
            ([8.0, 8.0],),]

    query = [([0.0, 0.0],), 
             ([1.0, 1.0],),
             ([4.1, 4.1],),
             ([8.0, 8.0],),
             ([9.0, 9.0],),]

    topk = 2

    with CleanSparkSession() as spark:
        schema = f"features array<float>"
        data_df = spark.createDataFrame(data, schema)
        query_df = spark.createDataFrame(query, schema)

        gpu_knn = NearestNeighbors(num_workers=gpu_number)
        gpu_knn = gpu_knn.setInputCol("features")
        gpu_knn = gpu_knn.setK(topk)

        gpu_knn = gpu_knn.fit(data_df)
        (knn_df, data_df) = gpu_knn.kneighbors(query_df)
        distances_df = knn_df.select("distances")
        indices_df = knn_df.select("indices")

        
        import math
        distances = distances_df.toPandas().to_numpy()
        distances.show()
        print(distances.toPandas())
        print(distances.toPandas().to_numpy())
        indices = indices_df.toPandas().to_numpy()

        assert array_equal(distances[0], [math.sqrt(2.), math.sqrt(8.)])
        assert indices[0] == [0, 1]

        assert array_equal(distances[1], [0., math.sqrt(2.)])
        assert indices[1] == [0, 1]

        assert array_equal(distances[2], [math.sqrt(0.01 + 0.01), math.sqrt(0.81 + 0.81)])
        assert indices[2] == [3, 4]

        assert array_equal(distances[3], [0., math.sqrt(2.)])
        assert indices[3] == [7, 6]

        assert array_equal(distances[4], [math.sqrt(2.), math.sqrt(8.)])
        assert indices[4] == [7, 6]


    

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

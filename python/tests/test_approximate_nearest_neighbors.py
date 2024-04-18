from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from pyspark.sql import DataFrame
from sklearn.datasets import make_blobs

from spark_rapids_ml.core import alias
from spark_rapids_ml.knn import (
    ApproximateNearestNeighbors,
    ApproximateNearestNeighborsModel,
)

from .sparksession import CleanSparkSession
from .utils import (
    array_equal,
    create_pyspark_dataframe,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)


def test_example(gpu_number: int, tmp_path: str) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)

    data = [
        ([1.0, 1.0], "a"),
        ([2.0, 2.0], "b"),
        ([3.0, 3.0], "c"),
        ([4.0, 4.0], "d"),
        ([5.0, 5.0], "e"),
        ([6.0, 6.0], "f"),
        ([7.0, 7.0], "g"),
        ([8.0, 8.0], "h"),
    ]

    query = [
        ([0.0, 0.0], "qa"),
        ([1.0, 1.0], "qb"),
        ([4.1, 4.1], "qc"),
        ([8.0, 8.0], "qd"),
        ([9.0, 9.0], "qe"),
    ]

    topk = 2

    # conf: Dict[str, Any] = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(20)}
    conf: Dict[str, Any] = {}
    with CleanSparkSession(conf) as spark:
        schema = f"features array<float>, metadata string"
        data_df = spark.createDataFrame(data, schema)
        query_df = spark.createDataFrame(query, schema)

        ivfflat = ApproximateNearestNeighbors(num_workers=gpu_number)
        ivfflat = ivfflat.setInputCol("features")
        ivfflat = ivfflat.setK(topk)

        with pytest.raises(NotImplementedError):
            ivfflat.save(tmp_path + "/knn_esimator")

        ivfflat_model = ivfflat.fit(data_df)

        with pytest.raises(NotImplementedError):
            ivfflat_model.save(tmp_path + "/knn_model")

        (item_df_withid, query_df_withid, knn_df) = ivfflat_model.kneighbors(query_df)
        item_df_withid.show()
        query_df_withid.show()
        knn_df.show()

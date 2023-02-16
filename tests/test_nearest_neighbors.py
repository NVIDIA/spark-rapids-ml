from typing import Any, Dict, Tuple

import numpy as np
import pytest
from pyspark import BarrierTaskContext
from sklearn.datasets import make_blobs

from spark_rapids_ml.feature import PCA, PCAModel
from spark_rapids_ml.common.cuml_context import CumlContext

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

@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
def test_ucx(gpu_number: int, data_shape: Tuple[int, int]) -> None:
    X, _ = make_blobs(n_samples=data_shape[0], n_features=data_shape[1], random_state=0)

    with CleanSparkSession() as spark:
        train_df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type = feature_types.array, dtype= np.float32, data = X, label = None
        )

        dataset = train_df.repartition(gpu_number)

        def _train_udf(pdf_iter):
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            rank = context.partitionId()
            with CumlContext(rank = rank, nranks = gpu_number, context = context, enable=True, require_ucx=True) as cc:
                assert cc._ucx != None
                assert cc._ucx_port != None
                assert len(cc._ucx._server_endpoints) == gpu_number
                assert len(cc._ucx._endpoints) == gpu_number
                for pdf in pdf_iter:
                    yield pdf

        rdd = (
            dataset.mapInPandas(_train_udf, schema=dataset.schema)  # type: ignore
            .rdd.barrier()
            .mapPartitions(lambda x: x)
        )

        rdd.count()
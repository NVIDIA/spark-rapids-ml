from typing import Any, Dict, Tuple

import numpy as np
import pytest
from pyspark import BarrierTaskContext
from sklearn.datasets import make_blobs

from spark_rapids_ml.common.cuml_context import CumlContext
from spark_rapids_ml.core import _CumlCommon
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

def test_ucx_over_nccl(gpu_number: int, data_shape: Tuple[int, int] = (1000, 20)) -> None:
    """
        If fails, try:
        (1) Run "export UCXPY_LOG_LEVEL=DEBUG" in termninal to enable UCX logging
        (2) Set additional_timeout to a larger value in CumlContext._ucp_create_endpoints(ucx_worker, target_ip_ports, additional_timeout)
    """
    X, _ = make_blobs(n_samples=data_shape[0], n_features=data_shape[1], random_state=0)

    with CleanSparkSession() as spark:
        train_df, features_col, _ = create_pyspark_dataframe(
            spark, feature_type = feature_types.array, dtype= np.float32, data = X, label = None
        )

        dataset = train_df.repartition(gpu_number)

        is_local = _is_local(_get_spark_session().sparkContext)
        def _train_udf(pdf_iter):
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            rank = context.partitionId()

            # ucx requires nccl, and nccl initialization requires gpu assignment  
            _CumlCommon.set_gpu_device(context, is_local)
            with CumlContext(rank = rank, nranks = gpu_number, context = context, enable=True, require_ucx=True) as cc:
                assert cc._ucx != None
                assert cc._ucx_port != None
                assert cc._ucx_eps != None
                assert len(cc._ucx_eps) == gpu_number
                assert len(cc._ucx._server_endpoints) == gpu_number
                for pdf in pdf_iter:
                    yield pdf

        rdd = (
            dataset.mapInPandas(_train_udf, schema=dataset.schema)  # type: ignore
            .rdd.barrier()
            .mapPartitions(lambda x: x)
        )

        rdd.count()
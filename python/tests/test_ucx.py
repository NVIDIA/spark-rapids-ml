#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
import json
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from spark_rapids_ml.common.cuml_context import CumlContext
from spark_rapids_ml.core import _CumlCommon
from spark_rapids_ml.utils import _get_spark_session, _is_local

from .conftest import _gpu_number
from .sparksession import CleanSparkSession
from .utils import create_pyspark_dataframe, feature_types


@pytest.mark.parametrize("gpu_number_used", range(1, _gpu_number + 1))
def test_ucx_over_nccl(
    gpu_number_used: int, data_shape: Tuple[int, int] = (1000, 20)
) -> None:
    """
    If fails, try:
    Run "export UCXPY_LOG_LEVEL=DEBUG" in termninal to enable UCX logging
    """
    gpu_number = gpu_number_used
    X, _ = make_blobs(n_samples=data_shape[0], n_features=data_shape[1], random_state=0)

    with CleanSparkSession() as spark:
        train_df, features_col, _ = create_pyspark_dataframe(
            spark,
            feature_type=feature_types.array,
            dtype=np.float32,  # type: ignore
            data=X,
            label=None,
        )

        dataset = train_df.repartition(gpu_number)

        is_local = _is_local(_get_spark_session().sparkContext)

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            rank = context.partitionId()

            # ucx requires nccl, and nccl initialization requires gpu assignment
            _CumlCommon._set_gpu_device(context, is_local)
            with CumlContext(
                rank=rank,
                nranks=gpu_number,
                context=context,
                enable=True,
                require_ucx=True,
            ) as cc:
                # pyspark uses sighup to kill python workers gracefully, and for some reason
                # the signal handler for sighup needs to be explicitly reset at this point
                # to avoid having SIGHUP be swallowed during a usleep call in the nccl library.
                # this helps avoid zombie surviving python workers when some workers fail.
                import signal

                signal.signal(signal.SIGHUP, signal.SIG_DFL)

                async def do_allGather() -> List[str]:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, context.allGather, json.dumps("hello")
                    )
                    return result

                assert cc._ucx is not None
                assert cc._ucx_port is not None
                assert cc._ucx_eps is not None
                assert cc._loop is not None
                assert len(cc._ucx_eps) == gpu_number
                assert len(cc._ucx._server_endpoints) == gpu_number

                cc._loop.run_until_complete(asyncio.ensure_future(do_allGather()))
                for pdf in pdf_iter:
                    yield pdf

        rdd = (
            dataset.mapInPandas(_train_udf, schema=dataset.schema)  # type: ignore
            .rdd.barrier()
            .mapPartitions(lambda x: x)
        )

        rdd.count()

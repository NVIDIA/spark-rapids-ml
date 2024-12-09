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

import math
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from gen_data_distributed import SparseRegressionDataGen
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.sql import functions as SparkF

from spark_rapids_ml.classification import LogisticRegression
from tests.test_logistic_regression import compare_model

from .conftest import _spark


def test_sparse_int64(multi_gpus: bool = False) -> None:
    """
    This test requires minimum 128G CPU memory, 32 GB GPU memory
    TODO: move generated dataset to a unified place
    """
    gpu_number = 2 if multi_gpus else 1
    output_num_files = 100  # large value smaller CPU memory for each spark task
    data_shape = (int(1e7), 2200)

    fraction_sampled_for_test = (
        1.0 if data_shape[0] <= 100000 else 100000 / data_shape[0]
    )
    n_classes = 8
    tolerance = 0.001
    est_params: Dict[str, Any] = {
        "regParam": 0.02,
        "maxIter": 10,
        "standardization": False,  # reduce GPU memory since standardization copies the value array
    }
    density = 0.1

    data_gen_args = [
        "--n_informative",
        f"{math.ceil(data_shape[1] / 3)}",
        "--num_rows",
        str(data_shape[0]),
        "--num_cols",
        str(data_shape[1]),
        "--output_num_files",
        str(output_num_files),
        "--dtype",
        "float32",
        "--feature_type",
        "vector",
        "--output_dir",
        "./temp",
        "--n_classes",
        str(n_classes),
        "--random_state",
        "0",
        "--logistic_regression",
        "True",
        "--density",
        str(density),
        "--use_gpu",
        "True",
    ]

    data_gen = SparseRegressionDataGen(data_gen_args)
    df, _, _ = data_gen.gen_dataframe_and_meta(_spark)

    df = df.cache()
    df_gpu = df

    if gpu_number > 1:
        main_pid = 0
        pid_col = "pid"
        delta_ratio = 0.1

        delta_df = df.sample(fraction=delta_ratio, seed=0)

        df = df.withColumn(pid_col, SparkF.lit(main_pid))
        delta_df = delta_df.withColumn(
            pid_col, SparkF.monotonically_increasing_id() % (gpu_number * 4)
        )

        df = df.union(delta_df)
        df_gpu = df.repartition(gpu_number, pid_col)

    def get_nnz_func(pdf_iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
        for pdf in pdf_iter:
            pd_res = pdf["features"].apply(lambda sparse_vec: len(sparse_vec["values"]))
            yield pd_res.rename("nnz").to_frame()

    nnz_df = df.mapInPandas(get_nnz_func, schema="nnz long")

    total_nnz = nnz_df.select(SparkF.sum("nnz").alias("res")).first()["res"]  # type: ignore
    assert total_nnz > np.iinfo(np.int32).max

    # compare gpu with spark cpu
    gpu_est = LogisticRegression(num_workers=gpu_number, verbose=True, **est_params)
    gpu_model = gpu_est.fit(df_gpu)
    cpu_est = SparkLogisticRegression(**est_params)
    cpu_model = cpu_est.fit(df)
    cpu_objective = cpu_model.summary.objectiveHistory[-1]
    assert (
        gpu_model.objective < cpu_objective
        or abs(gpu_model.objective - cpu_objective) < tolerance
    )

    df_test = df.sample(fraction=fraction_sampled_for_test, seed=0)
    compare_model(
        gpu_model,
        cpu_model,
        df_test,
        unit_tol=tolerance,
        total_tol=tolerance,
        accuracy_and_probability_only=True,
    )

    if gpu_number == 1:
        assert gpu_est._index_dtype == "int64"


def test_sparse_int64_mg() -> None:
    test_sparse_int64(multi_gpus=True)

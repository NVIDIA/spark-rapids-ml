# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import pytest
from gen_data_distributed import SparseRegressionDataGen
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as SparkF
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)

from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel
from tests.test_logistic_regression import compare_model

from .conftest import _spark


def _compare_with_cpu_estimator(
    gpu_model: LogisticRegressionModel,
    est_params: Dict[str, Any],
    df: DataFrame,
    fraction_sampled_for_test: float,
    tolerance: float,
) -> None:
    cpu_est = SparkLogisticRegression(**est_params)
    cpu_model = cpu_est.fit(df)
    cpu_objective = cpu_model.summary.objectiveHistory[-1]

    for i, loss in enumerate(cpu_model.summary.objectiveHistory):
        print(f"Iteration {i}: loss = {loss:.6f}")

    df_test = df.sample(fraction=fraction_sampled_for_test, seed=0)

    assert (
        gpu_model.objective < cpu_objective
        or abs(gpu_model.objective - cpu_objective) < tolerance
    )

    compare_model(
        gpu_model,
        cpu_model,
        df_test,
        unit_tol=tolerance,
        total_tol=tolerance,
        accuracy_and_probability_only=True,
        y_true_col=cpu_est.getLabelCol(),
    )


def test_sparse_large(
    multi_gpus: bool = False,
    standardization: bool = False,
    float32_inputs: bool = True,
    n_rows: int = int(1e7),
    n_cols: int = 2200,
    density: float = 0.1,
    tolerance: float = 0.001,
) -> None:
    """
    This test requires minimum 128G CPU memory, 32 GB GPU memory
    TODO: move generated dataset to a unified place

    if standardization or float32_inputs is True, the test case reduces more GPU memory since standardization copies the value array
    """
    gpu_number = 2 if multi_gpus else 1
    output_num_files = 100  # large value smaller CPU memory for each spark task
    data_shape = (n_rows, n_cols)

    fraction_sampled_for_test = (
        1.0 if data_shape[0] <= 100000 else 100000 / data_shape[0]
    )
    n_classes = 8
    est_params: Dict[str, Any] = {
        "regParam": 0.02,
        "maxIter": 10,
        "standardization": standardization,
    }

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

    # compare gpu with spark cpu
    gpu_est = LogisticRegression(
        num_workers=gpu_number,
        verbose=True,
        float32_inputs=float32_inputs,
        **est_params,
    )
    gpu_model = gpu_est.fit(df_gpu)

    _compare_with_cpu_estimator(
        gpu_model, est_params, df, fraction_sampled_for_test, tolerance
    )

    if (total_nnz / gpu_number) > LogisticRegression._nnz_limit_for_int32():
        assert gpu_est._index_dtype == "int64"
    else:
        assert gpu_est._index_dtype == "int32"


def test_sparse_int64_mg() -> None:
    test_sparse_large(multi_gpus=True, tolerance=0.005)


@pytest.mark.parametrize("float32_inputs", [True, False])
def test_sparse_int64_standardization(float32_inputs: bool) -> None:
    test_sparse_large(
        multi_gpus=False, standardization=True, float32_inputs=float32_inputs
    )


@pytest.mark.parametrize("float32_inputs", [True, False])
@pytest.mark.parametrize("beyond_limit", [True, False])
def test_sparse_large_int32(float32_inputs: bool, beyond_limit: bool) -> None:
    """
    test large nnz representable by int32
    """

    if beyond_limit:
        n_rows = int(1e7)
        n_cols = 1800
        density = 0.1

        expected_nnz = n_rows * n_cols * density
        assert (
            expected_nnz > LogisticRegression._nnz_limit_for_int32()
            and expected_nnz < np.iinfo("int32").max
        )
    else:
        n_rows = int(1e7)
        n_cols = 900
        density = 0.1

        expected_nnz = n_rows * n_cols * density
        assert expected_nnz <= LogisticRegression._nnz_limit_for_int32()

    test_sparse_large(
        multi_gpus=False,
        standardization=True,
        float32_inputs=float32_inputs,
        n_rows=n_rows,
        n_cols=n_cols,
        density=density,
        tolerance=0.005,
    )

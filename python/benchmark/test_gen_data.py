# Copyright (c) 2007-2023 The scikit-learn developers. All rights reserved.
# Modifications copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import pytest
from gen_data_distributed import BlobsDataGen, LowRankMatrixDataGen, RegressionDataGen
from pandas import DataFrame
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from benchmark.utils import WithSparkSession


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_make_blobs(dtype: str) -> None:
    input_args = [
        "--num_rows",
        "50",
        "--num_cols",
        "2",
        "--dtype",
        dtype,
        "--output_dir",
        "temp",
        "--output_num_files",
        "3",
        "--cluster_std",
        "0.7",
        "--random_state",
        "0",
    ]
    data_gen = BlobsDataGen(input_args)
    args = data_gen.args
    assert args is not None
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, _, centers = data_gen.gen_dataframe_and_meta(spark)
        assert df.rdd.getNumPartitions() == 3, "Unexpected number of partitions"
        pdf: DataFrame = df.toPandas()

        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.dtype == np.dtype(dtype), "Unexpected dtype"
        assert X.shape == (50, 2), "X shape mismatch"
        assert y.shape == (50,), "y shape mismatch"
        assert centers.shape == (3, 2), "Centers shape mismatch"
        assert np.unique(y).shape == (3,), "Unexpected number of blobs"

        cluster_stds = [0.7] * 3
        for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
            assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_make_low_rank_matrix(dtype: str) -> None:
    input_args = [
        "--num_rows",
        "50",
        "--num_cols",
        "20",
        "--dtype",
        dtype,
        "--output_dir",
        "temp",
        "--output_num_files",
        "2",
        "--effective_rank",
        "5",
        "--tail_strength",
        "0.01",
        "--random_state",
        "0",
    ]
    data_gen = LowRankMatrixDataGen(input_args)
    args = data_gen.args
    assert args is not None
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, _ = data_gen.gen_dataframe(spark)
        assert df.rdd.getNumPartitions() == 2, "Unexpected number of partitions"
        pdf: DataFrame = df.toPandas()
        X = pdf.to_numpy()

        assert X.dtype == np.dtype(dtype), "Unexpected dtype"
        assert X.shape == (50, 20), "X shape mismatch"
        from numpy.linalg import svd

        _, s, _ = svd(X)
        assert sum(s) - 5 < 0.1, "X rank is not approximately 5"


@pytest.mark.parametrize("dtype, low_rank", [("float32", True), ("float32", False), ("float64", True), ("float64", False)])
def test_make_regression(dtype: str, low_rank: bool) -> None:
    input_args = [
        "--num_rows",
        "100",
        "--num_cols",
        "10",
        "--dtype",
        dtype,
        "--output_dir",
        "temp",
        "--output_num_files",
        "3",
        "--n_informative",
        "3",
        "--bias",
        "0.0",
        "--noise",
        "1.0",
        "--random_state",
        "0",
    ]
    if low_rank:
        input_args.extend(("--effective_rank", "5"))
    data_gen = RegressionDataGen(input_args)
    args = data_gen.args
    assert args is not None
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, _, c = data_gen.gen_dataframe_and_meta(spark)
        assert df.rdd.getNumPartitions() == 3, "Unexpected number of partitions"
        pdf: DataFrame = df.toPandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.dtype == np.dtype(dtype), "Unexpected dtype"
        assert X.shape == (100, 10), "X shape mismatch"
        assert y.shape == (100,), "y shape mismatch"
        assert c.shape == (10,), "coef shape mismatch"
        assert sum(c != 0.0) == 3, "Unexpected number of informative features"

        # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
        assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)

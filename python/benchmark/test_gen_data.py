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
from gen_data_distributed import BlobsDataGen
from pandas import DataFrame
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from benchmark.utils import WithSparkSession


def test_make_blobs() -> None:
    input_args = [
        "--num_rows",
        "50",
        "--num_cols",
        "2",
        "--dtype",
        "float64",
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
        assert df.rdd.getNumPartitions() == 3
        pdf: DataFrame = df.toPandas()

        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.shape == (50, 2), "X shape mismatch"
        assert y.shape == (50,), "y shape mismatch"
        assert centers.shape == (3, 2), "Centers shape mismatch"
        assert np.unique(y).shape == (3,), "Unexpected number of blobs"

        cluster_stds = [0.7] * 3
        for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
            assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")

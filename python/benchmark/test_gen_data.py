# Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
# Modifications copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from typing import List, Union

import numpy as np
import pytest
from gen_data_distributed import (
    BlobsDataGen,
    ClassificationDataGen,
    LowRankMatrixDataGen,
    RegressionDataGen,
    SparseRegressionDataGen,
)
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
        "--include_labels",
        "true",
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
@pytest.mark.parametrize("use_gpu", ["True", "False"])
def test_make_low_rank_matrix(dtype: str, use_gpu: str) -> None:
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
        "--use_gpu",
        use_gpu,
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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("low_rank", [True, False])
@pytest.mark.parametrize("use_gpu", ["True", "False"])
@pytest.mark.parametrize(
    "logistic_regression, n_classes, bias",
    [
        ("True", "2", "1.0"),
        ("True", "5", ["0.5", "1.5", "2.5", "3.5", "4.5"]),
        ("True", "15", "1.5"),
        ("False", "0", "1.0"),
    ],
)
def test_make_regression(
    dtype: str,
    low_rank: bool,
    use_gpu: str,
    logistic_regression: str,
    n_classes: str,
    bias: Union[str, List[str]],
) -> None:
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
        "--noise",
        "1.0",
        "--random_state",
        "0",
        "--use_gpu",
        use_gpu,
        "--logistic_regression",
        logistic_regression,
        "--n_classes",
        n_classes,
    ]
    if low_rank:
        input_args.extend(("--effective_rank", "5"))

    input_args.append("--bias")
    if isinstance(bias, List):
        input_args.extend(bias)
    else:
        input_args.append(bias)

    data_gen = RegressionDataGen(input_args)
    args = data_gen.args
    assert args is not None
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, _, c = data_gen.gen_dataframe_and_meta(spark)
        assert df.rdd.getNumPartitions() == 3, "Unexpected number of partitions"
        pdf: DataFrame = df.toPandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        col_num = 10

        assert X.dtype == np.dtype(dtype), "Unexpected dtype"
        assert X.shape == (100, 10), "X shape mismatch"
        assert y.shape == (100,), "y shape mismatch"

        n_classes_num = int(n_classes)

        if logistic_regression == "False" or n_classes_num == 2:
            assert c.shape == (col_num,), "coef shape mismatch"
            assert np.count_nonzero(c) == 3, "Unexpected number of informative features"
        else:
            assert c.shape == (
                col_num,
                n_classes_num,
            ), "coef shape mismatch"
            assert (
                np.count_nonzero(c) == 3 * n_classes_num
            ), "Unexpected number of informative features"

        if logistic_regression == "True":
            # Test that y consists of only discrete label
            possible_labels = range(n_classes_num)
            for n in y:
                found = False
                for l in possible_labels:
                    if n == l:
                        found = True
                        break
                assert found, "Invalid label"
        else:
            # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
            assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)


@pytest.mark.parametrize("dtype", ["float64"])
@pytest.mark.parametrize("use_gpu", ["True", "False"])
@pytest.mark.parametrize("redundant_cols", ["0", "2"])
@pytest.mark.parametrize(
    "logistic_regression, n_classes, bias",
    [
        ("True", "2", "1.0"),
        ("True", "5", ["0.5", "1.5", "2.5", "3.5", "4.5"]),
        ("True", "15", "1.5"),
        ("False", "0", "1.0"),
    ],
)
@pytest.mark.parametrize(
    "density",
    ["0.25", ["0.05", "0.1", "0.2"], pytest.param("0.2", marks=pytest.mark.slow)],
)
@pytest.mark.parametrize(
    "rows, cols", [("100", "20"), pytest.param("1000", "100", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize(
    "density_curve, shuffle",
    [
        ("None", "True"),
        ("Linear", "True"),
        ("Linear", "False"),
        ("Exponential", "False"),
        pytest.param("Exponential", "True", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "n_chunks", ["10", pytest.param("100", marks=pytest.mark.slow)]
)
def test_make_sparse_regression(
    dtype: str,
    use_gpu: str,
    redundant_cols: str,
    logistic_regression: str,
    n_classes: str,
    bias: Union[str, List[str]],
    density: Union[str, List[str]],
    rows: str,
    cols: str,
    density_curve: str,
    shuffle: str,
    n_chunks: str,
) -> None:

    input_args = [
        "--num_rows",
        rows,
        "--num_cols",
        cols,
        "--dtype",
        dtype,
        "--output_dir",
        "temp",
        "--output_num_files",
        "3",
        "--n_informative",
        "3",
        "--n_classes",
        n_classes,
        "--noise",
        "1.0",
        "--random_state",
        "0",
        "--use_gpu",
        use_gpu,
        "--redundant_cols",
        redundant_cols,
        "--logistic_regression",
        logistic_regression,
        "--density_curve",
        density_curve,
        "--shuffle",
        shuffle,
    ]

    input_args.append("--bias")
    if isinstance(bias, List):
        input_args.extend(bias)
    else:
        input_args.append(bias)

    input_args.append("--density")
    if isinstance(density, List):
        input_args.extend(density)
    else:
        input_args.append(density)

    row_num = int(rows)
    col_num = int(cols)
    n_classes_num = int(n_classes)

    data_gen = SparseRegressionDataGen(input_args)
    args = data_gen.args
    assert args is not None
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, _, c = data_gen.gen_dataframe_and_meta(spark)
        assert df.rdd.getNumPartitions() == 3, "Unexpected number of partitions"

        pdf: DataFrame = df.toPandas()
        X = pdf.iloc[:, 0].to_numpy()
        y = pdf.iloc[:, 1].to_numpy()

        assert len(X) == row_num, "X row number mismatch"
        for sparseVec in X:
            # assert sparseVec.toArray().dtype == np.dtype(dtype), "Unexpected dtype"
            assert sparseVec.size == col_num, "X col number mismatch"
        assert y.shape == (row_num,), "y shape mismatch"

        if logistic_regression == "False" or n_classes_num == 2:
            assert c.shape == (col_num,), "coef shape mismatch"
            assert np.count_nonzero(c) == 3, "Unexpected number of informative features"
        else:
            assert c.shape == (
                col_num,
                n_classes_num,
            ), "coef shape mismatch"
            assert (
                np.count_nonzero(c) == 3 * n_classes_num
            ), "Unexpected number of informative features"

        X_np = np.array([r.toArray() for r in X])

        if logistic_regression == "True":
            # Test that X consists of only discrete label
            possible_labels = range(n_classes_num)
            for n in y:
                found = False
                for l in possible_labels:
                    if n == l:
                        found = True
                        break
                assert found, "Invalid label"
        else:
            # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
            assert_almost_equal(np.std(y - np.dot(X_np, c)), 1.0, decimal=1)

        # Check density match
        count = np.count_nonzero(X_np)

        total = row_num * col_num

        # If there is no random shuffled redundant cols, we can check the total density
        if redundant_cols == "0" and density_curve == "None":
            if isinstance(density, List):
                density_num = sum([float(d) for d in density]) / len(density)
            else:
                density_num = float(density)

            assert (
                count > total * density_num * 0.95
                and count < total * density_num * 1.05
            )

        # If no shuffle with a density curve, test to see if the column density is increasing as the desired curve
        n_chunks = int(n_chunks)
        if density_curve != "None" and shuffle == "False":
            orig_cols = col_num - int(redundant_cols)
            num_partitions = 3

            if isinstance(density, List):
                density_num = float(density[0])
            else:
                density_num = float(density)

            if density_curve == "Linear":
                density_values = np.linspace(
                    num_partitions / row_num, density_num, n_chunks
                )
                density_values *= n_chunks * density_num / sum(density_values)
            else:
                density_values = np.logspace(
                    np.log10(num_partitions / row_num), np.log10(density_num), n_chunks
                )
                density_values *= n_chunks * density_num / sum(density_values)

            col_per_chunk = np.full(n_chunks, orig_cols // n_chunks)
            col_per_chunk[: (orig_cols % n_chunks)] += 1
            col_per_chunk = np.cumsum(col_per_chunk)

            density_idx = 0
            dense_count = 0
            for idx, col in enumerate(X_np):
                # Ignore the redundant columns
                if idx >= orig_cols:
                    break

                if density_values[density_idx] == idx:
                    col_density = density_values[density_idx]
                    chunk_size = col_per_chunk[density_idx]

                    assert dense_count >= chunk_size * num_partitions * int(
                        (row_num // num_partitions) * col_density - 1
                    ) and dense_count <= chunk_size * num_partitions * int(
                        (row_num // num_partitions) * col_density + 1
                    )

                    density_idx += 1
                    dense_count = 0

                dense_count += np.count_nonzero(X_np[:, idx])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("num_rows", [2000, 2001])  # test uneven samples per cluster
@pytest.mark.parametrize(
    "n_informative, n_repeated, n_redundant", [(31, 0, 0), (28, 3, 0), (23, 3, 4)]
)
def test_make_classification(
    dtype: str, num_rows: int, n_informative: int, n_repeated: int, n_redundant: int
) -> None:
    input_args = [
        "--num_rows",
        str(num_rows),
        "--num_cols",
        "31",
        "--dtype",
        dtype,
        "--output_dir",
        "temp",
        "--output_num_files",
        "3",
        "--n_informative",
        str(n_informative),
        "--n_redundant",
        str(n_redundant),
        "--n_repeated",
        str(n_repeated),
        "--hypercube",
        "True",
        "--scale",
        "0.5",
        "--flip_y",
        "0",
        "--random_state",
        "0",
    ]
    data_gen = ClassificationDataGen(input_args)
    args = data_gen.args
    assert args is not None
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        df, _ = data_gen.gen_dataframe(spark)
        assert df.rdd.getNumPartitions() == 3, "Unexpected number of partitions"
        pdf: DataFrame = df.toPandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.dtype == np.dtype(dtype), "Unexpected dtype"
        assert X.shape == (num_rows, 31), "X shape mismatch"
        assert y.shape == (num_rows,), "y shape mismatch"
        assert np.unique(y).shape == (2,), "Unexpected number of classes"
        if num_rows == 2000:
            assert sum(y == 0) == 1000, "Unexpected number of samples in class 0"
            assert sum(y == 1) == 1000, "Unexpected number of samples in class 1"
        else:
            assert (
                abs(sum(y == 0) - sum(y == 1)) == 1
            ), "Unexpected number of samples per class"
        assert (
            np.unique(X, axis=0).shape[0] == num_rows
        ), "Unexpected number of unique rows"
        assert (
            np.unique(X, axis=1).shape[1] == 31 - n_repeated
        ), "Unexpected number of unique columns"
        assert (
            np.linalg.matrix_rank(X) == 31 - n_repeated - n_redundant
        ), "Unexpected matrix rank"

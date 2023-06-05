import numpy as np
from gen_data_distributed import (
    BlobsDataGen,
    ClassificationDataGen,
    LowRankMatrixDataGen,
    RegressionDataGen,
)
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_low_rank_matrix,
    make_regression,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from benchmark.utils import WithSparkSession, inspect_default_params_from_func, to_bool


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
        pdf = df.toPandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.shape == (50, 2), "X shape mismatch"
        assert y.shape == (50,), "y shape mismatch"
        assert centers.shape == (3, 2), "Centers shape mismatch"
        assert np.unique(y).shape == (3,), "Unexpected number of blobs"

        cluster_stds = [0.7 for _ in range(3)]
        for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
            assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")

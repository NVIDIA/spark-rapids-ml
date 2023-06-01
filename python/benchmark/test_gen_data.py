import numpy as np
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_allclose

from benchmark.gen_data import DataGen, DataGenBase, BlobsDataGen, LowRankMatrixDataGen, RegressionDataGen, ClassificationDataGen
from benchmark.benchmark.utils import WithSparkSession, inspect_default_params_from_func, to_bool

def test_make_blobs():
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    
    args = ['--num_rows', '50', '--num_cols', '2', '--dtype', 'float64', '--output_dir', 'temp', 
            '--output_num_files', '3', '--centers', '[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]', 
            '--cluster_std', '[0.05, 0.2, 0.4]', '--random_state', '0']
    data_gen = BlobsDataGen(args)
    args = data_gen.args
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        pdf, _ = data_gen.gen_dataframe(spark).to_Pandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.shape == (50, 2), "X shape mismatch"
        assert y.shape == (50,), "y shape mismatch"
        assert np.unique(y).shape == (3,), "Unexpected number of blobs"
        for i, (ctr, std) in enumerate(zip(cluster_centers, cluster_stds)):
            assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")

def test_make_low_rank_matrix():
    args = ['--num_rows', '50', '--num_cols', '25', '--dtype', 'float64', '--output_dir', 'temp', 
            '--output_num_files', '3', '--effective_rank', '5', '--tail_strength', '0.01', '--random_state', '0']
    data_gen = LowRankMatrixDataGen(args)
    args = data_gen.args
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        pdf, _ = data_gen.gen_dataframe(spark).toPandas()
        X = pdf.to_numpy()

        assert X.shape == (50, 25), "X shape mismatch"
        from numpy.linalg import svd
        u, s, v = svd(X)
        assert sum(s) - 5 < 0.1, "X rank is not approximately 5"

'''
def test_make_regression():
    args = ['--num_rows', '100', '--num_cols', '10', '--dtype', 'float64', '--output_dir', 'temp', 
            '--output_num_files', '3', '--n_informative', '3', '--effective_rank', '5', '--coef', 'True', 
            '--bias', '0.0', '--noise', '1.0', '--random_state', '0']
    data_gen = RegressionDataGen(args)
    args = data_gen.args
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        pdf, _ = data_gen.gen_dataframe(spark).to_Pandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()
        #TODO: determine whether c will be stored in df or returned separately
        c = None

        assert X.shape == (100, 10), "X shape mismatch"
        assert y.shape == (100,), "y shape mismatch"
        assert c.shape == (10,), "coef shape mismatch"
        assert sum(c != 0.0) == 3, "Unexpected number of informative features"

        # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
        assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)
'''

def test_make_classification():
    args = ['--num_rows', '2000', '--num_cols', '31', '--dtype', 'float64', '--output_dir', 'temp', 
            '--output_num_files', '3', '--n_informative', '31', '--n_redundant', '0', '--n_repeated', '0',
            '--hypercube', 'True', '--scale', '0.5', '--random_state', '0']
    data_gen = ClassificationDataGen(args)
    args = data_gen.args
    with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
        pdf, _ = data_gen.gen_dataframe(spark).toPandas()
        X = pdf.iloc[:, :-1].to_numpy()
        y = pdf.iloc[:, -1].to_numpy()

        assert X.shape == (2000, 31), "X shape mismatch"
        assert y.shape == (2000,), "y shape mismatch"
        assert (
            np.unique(X.view([("", X.dtype)] * X.shape[1]))
            .view(X.dtype)
            .reshape(-1, X.shape[1])
            .shape[0]
            == 2000
        ), "Unexpected number of unique rows"
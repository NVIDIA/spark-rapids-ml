import numpy as np
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_allclose

from ..benchmark.gen_data import DataGen, DataGenBase, BlobsDataGen, LowRankMatrixDataGen, RegressionDataGen, ClassificationDataGen
from ..benchmark.benchmark.utils import WithSparkSession, inspect_default_params_from_func, to_bool

# sys.argv = ['--num_rows', '40', '--num_cols', '2', '--dtype', 'float64', '--output_dir', './40_blobs.parquet', '--cluster_std', '0.9']

def test_make_blobs():
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    
    args = ['--num_rows', '50', '--num_cols', '2', '--dtype', 'float64', '--output_dir', './40_blobs.parquet', 
            '--cluster_std', '[0.05, 0.2, 0.4]', '--random_state', '0', 
            '--centers', '[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]']
    data_gen = BlobsDataGen(args)
    args = data_gen.args

    '''
    X, y = make_blobs(
        random_state=0,
        n_samples=50,
        n_features=2,
        centers=cluster_centers,
        cluster_std=cluster_stds,
    )
    '''
    assert X.shape == (50, 2), "X shape mismatch"
    assert y.shape == (50,), "y shape mismatch"
    assert np.unique(y).shape == (3,), "Unexpected number of blobs"
    for i, (ctr, std) in enumerate(zip(cluster_centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")

def test_make_blobs_n_samples_centers_none(n_samples):
    centers = None
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0)

    assert X.shape == (sum(n_samples), 2), "X shape mismatch"
    assert all(
        np.bincount(y, minlength=len(n_samples)) == n_samples
    ), "Incorrect number of samples per blob"

def test_make_low_rank_matrix():
    X = make_low_rank_matrix(
        n_samples=50,
        n_features=25,
        effective_rank=5,
        tail_strength=0.01,
        random_state=0,
    )

    assert X.shape == (50, 25), "X shape mismatch"

    from numpy.linalg import svd

    u, s, v = svd(X)
    assert sum(s) - 5 < 0.1, "X rank is not approximately 5"

def test_make_classification():
    weights = [0.1, 0.25]
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=1,
        n_repeated=1,
        n_classes=3,
        n_clusters_per_class=1,
        hypercube=False,
        shift=None,
        scale=None,
        weights=weights,
        random_state=0,
    )

    assert weights == [0.1, 0.25]
    assert X.shape == (100, 20), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    assert np.unique(y).shape == (3,), "Unexpected number of classes"
    assert sum(y == 0) == 10, "Unexpected number of samples in class #0"
    assert sum(y == 1) == 25, "Unexpected number of samples in class #1"
    assert sum(y == 2) == 65, "Unexpected number of samples in class #2"

    # Test for n_features > 30
    X, y = make_classification(
        n_samples=2000,
        n_features=31,
        n_informative=31,
        n_redundant=0,
        n_repeated=0,
        hypercube=True,
        scale=0.5,
        random_state=0,
    )

    assert X.shape == (2000, 31), "X shape mismatch"
    assert y.shape == (2000,), "y shape mismatch"
    assert (
        np.unique(X.view([("", X.dtype)] * X.shape[1]))
        .view(X.dtype)
        .reshape(-1, X.shape[1])
        .shape[0]
        == 2000
    ), "Unexpected number of unique rows"

def test_make_regression():
    X, y, c = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        effective_rank=5,
        coef=True,
        bias=0.0,
        noise=1.0,
        random_state=0,
    )

    assert X.shape == (100, 10), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    assert c.shape == (10,), "coef shape mismatch"
    assert sum(c != 0.0) == 3, "Unexpected number of informative features"

    # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)

    # Test with small number of features.
    X, y = make_regression(n_samples=100, n_features=1)  # n_informative=3
    assert X.shape == (100, 1)

if __name__ == "__main__":
    print("hi :)")
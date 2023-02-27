import numpy as np
from clustpy.partition import PGMeans
from clustpy.partition.pgmeans import _initial_gmm_clusters
from clustpy.data import load_wine


def test_initial_centers():
    X = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4],
                  [5, 5, 5]])
    random_state = np.random.RandomState(1)
    n_clusters, gmm = _initial_gmm_clusters(X, n_clusters_init=1, gmm_repetitions=1, random_state=random_state)
    assert n_clusters == 1
    assert gmm.means_.shape == (1, 3)
    n_clusters, gmm = _initial_gmm_clusters(X, n_clusters_init=4, gmm_repetitions=1, random_state=random_state)
    assert n_clusters == 4
    assert gmm.means_.shape == (4, 3)
    n_clusters_init = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
    n_clusters, gmm = _initial_gmm_clusters(X, n_clusters_init=n_clusters_init, gmm_repetitions=1,
                                            random_state=random_state)
    assert n_clusters == 3
    assert gmm.means_.shape == (3, 3)


"""
Tests regarding the PGMeans object
"""


def test_simple_PGMeans_with_wine():
    X, labels = load_wine()
    pgmeans = PGMeans()
    assert not hasattr(pgmeans, "labels_")
    pgmeans.fit(X)
    assert pgmeans.labels_.dtype == np.int32
    assert pgmeans.labels_.shape == labels.shape
    assert pgmeans.cluster_centers_.shape == (pgmeans.n_clusters_, X.shape[1])
    assert len(np.unique(pgmeans.labels_)) == pgmeans.n_clusters_
    assert np.array_equal(np.unique(pgmeans.labels_), np.arange(pgmeans.n_clusters_))
    # Test with parameters
    pgmeans = PGMeans(significance=0.05, n_projections=5, n_samples=50, n_new_centers=6, amount_random_centers=0.2,
                      n_clusters_init=3, max_n_clusters=5, random_state=1)
    pgmeans.fit(X)
    assert pgmeans.labels_.dtype == np.int32
    assert pgmeans.labels_.shape == labels.shape
    assert pgmeans.cluster_centers_.shape == (pgmeans.n_clusters_, X.shape[1])
    assert len(np.unique(pgmeans.labels_)) == pgmeans.n_clusters_
    assert np.array_equal(np.unique(pgmeans.labels_), np.arange(pgmeans.n_clusters_))

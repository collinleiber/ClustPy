import numpy as np
from clustpy.partition import PGMeans
from clustpy.partition.pgmeans import _initial_gmm_clusters, _update_gmm_with_new_center, _project_model
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from sklearn.utils.estimator_checks import check_estimator


def test_pgmeans_estimator():
    check_estimator(PGMeans(), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_project_model():
    random_state = np.random.RandomState(1)
    X = np.array([[1, 2, 3],
                  [3, 1, 2],
                  [2, 3, 1],
                  [1, 1, 1],
                  [3, 3, 3],
                  [11, 12, 13],
                  [13, 11, 12],
                  [12, 13, 11],
                  [11, 11, 11],
                  [13, 13, 13],
                  [12, 12, 12]
                  ])
    gmm = GMM(n_components=2, n_init=1, random_state=random_state)
    gmm.fit(X)
    proj_gmm = _project_model(gmm, np.array([0, 1, 0]), 2, random_state)
    assert proj_gmm.means_.shape == (2, 1)
    assert proj_gmm.covariances_.shape == (2, 1, 1)
    assert np.allclose(proj_gmm.means_, np.array([[2], [12]])) or np.allclose(proj_gmm.means_, np.array([[12], [2]]))


def test_update_gmm_with_new_center():
    random_state = np.random.RandomState(1)
    X = np.array([[1, 2, 3],
                  [3, 1, 2],
                  [2, 3, 1],
                  [1, 1, 1],
                  [3, 3, 3],
                  [11, 12, 13],
                  [13, 11, 12],
                  [12, 13, 11],
                  [11, 11, 11],
                  [13, 13, 13],
                  [12, 12, 12],
                  [41, 42, 43],
                  [43, 41, 42],
                  [42, 43, 41],
                  [41, 41, 41],
                  [43, 43, 43],
                  [43, 42, 43],
                  [41, 42, 41]])
    gmm = GMM(n_components=2, n_init=1, random_state=random_state)
    gmm.fit(X)
    assert gmm.means_.shape == (2, 3)
    updated_gmm = _update_gmm_with_new_center(X, 3, gmm, 3, 3, random_state)
    assert updated_gmm.means_.shape == (3, 3)
    assert np.allclose(updated_gmm.means_[0], [2, 2, 2]) or np.allclose(updated_gmm.means_[1],
                                                                              [2, 2, 2]) or np.allclose(
        updated_gmm.means_[2], [2, 2, 2])
    assert np.allclose(updated_gmm.means_[0], [12, 12, 12]) or np.allclose(updated_gmm.means_[1],
                                                                                 [12, 12, 12]) or np.allclose(
        updated_gmm.means_[2], [12, 12, 12])
    assert np.allclose(updated_gmm.means_[0], [42, 42, 42]) or np.allclose(updated_gmm.means_[1],
                                                                                 [42, 42, 42]) or np.allclose(
        updated_gmm.means_[2], [42, 42, 42])


def test_initial_gmm_clusters():
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


def test_simple_PGMeans():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    pgmeans = PGMeans(random_state=1)
    assert not hasattr(pgmeans, "labels_")
    pgmeans.fit(X)
    assert pgmeans.labels_.dtype == np.int32
    assert pgmeans.labels_.shape == labels.shape
    assert pgmeans.cluster_centers_.shape == (pgmeans.n_clusters_, X.shape[1])
    assert len(np.unique(pgmeans.labels_)) == pgmeans.n_clusters_
    assert np.array_equal(np.unique(pgmeans.labels_), np.arange(pgmeans.n_clusters_))
    # Test if random state is working
    pgmeans2 = PGMeans(random_state=1)
    pgmeans2.fit(X)
    assert np.array_equal(pgmeans.n_clusters_, pgmeans2.n_clusters_)
    assert np.array_equal(pgmeans.labels_, pgmeans2.labels_)
    assert np.array_equal(pgmeans.cluster_centers_, pgmeans2.cluster_centers_)
    # Test with parameters
    pgmeans = PGMeans(significance=0.05, n_projections=5, n_samples=50, n_new_centers=6, amount_random_centers=0.2,
                      n_clusters_init=3, max_n_clusters=5, random_state=1)
    pgmeans.fit(X)
    assert pgmeans.labels_.dtype == np.int32
    assert pgmeans.labels_.shape == labels.shape
    assert pgmeans.cluster_centers_.shape == (pgmeans.n_clusters_, X.shape[1])
    assert len(np.unique(pgmeans.labels_)) == pgmeans.n_clusters_
    assert np.array_equal(np.unique(pgmeans.labels_), np.arange(pgmeans.n_clusters_))

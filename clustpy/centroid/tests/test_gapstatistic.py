import numpy as np
from clustpy.centroid import GapStatistic
from clustpy.centroid.gapstatistic import _execute_clusterer, _generate_random_data, _get_within_cluster_dispersion
from sklearn.datasets import make_blobs
from unittest.mock import patch
from clustpy.utils.checks import check_clustpy_estimator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


def test_gapstatistic_estimator():
    check_clustpy_estimator(GapStatistic(), ("check_complex_data"))


def test_execute_kmeans():
    X = np.array(
        [[0, 0], [1, -1], [1, 0], [1, 1], [2, 0],
         [4, 0], [5, -1], [5, 1], [5, 1], [6, 0], [5, -1]])
    random_state = np.random.RandomState(1)
    # With number of clusters == 1
    n_clusters = 1
    labels, centers, inertia = _execute_clusterer(X, n_clusters, KMeans, None, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 11))
    assert centers is None
    assert inertia is None
    # With number of clusters > 1
    n_clusters = 2
    labels, centers, inertia = _execute_clusterer(X, n_clusters, KMeans, None, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 5 + [1] * 6)) or np.array_equal(labels, np.array([1] * 5 + [0] * 6))
    assert np.array_equal(centers, [[1, 0], [5, 0]]) or np.array_equal(centers, [[5, 0], [1, 0]]) 
    assert abs(inertia - (4 + 6)) < 1e-9
    # With number of clusters > 1 and GMM
    n_clusters = 2
    labels, centers, inertia = _execute_clusterer(X, n_clusters, GaussianMixture, None, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 5 + [1] * 6)) or np.array_equal(labels, np.array([1] * 5 + [0] * 6))
    assert np.allclose(centers, [[1, 0], [5, 0]], atol=1e-4) or np.allclose(centers, [[5, 0], [1, 0]], atol=1e-4) 
    assert inertia is None
    # With DBSCAN
    labels, centers, inertia = _execute_clusterer(X, n_clusters, DBSCAN, {"eps":10, "min_samples": 5}, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 11))
    assert centers is None
    assert inertia is None


def test_get_within_cluster_dispersion():
    X = np.array(
        [[0, 0], [1, -1], [1, 0], [1, 1], [2, 0],
         [4, 0], [5, -1], [5, 1], [5, 1], [6, 0], [5, -1]])
    labels = np.array([0] * 5 + [1] * 6)
    centers = np.array([[1, 0], [5, 0]])
    inertia = 10
    W_k = _get_within_cluster_dispersion(X, labels, centers, inertia, False, False)
    assert W_k == 10
    W_k = _get_within_cluster_dispersion(X, labels, centers, None, True, False)
    assert W_k == np.log(10)
    W_k = _get_within_cluster_dispersion(X, labels, centers, inertia, True, True)
    assert W_k == np.log(4 / 4 + 6 / 5)
    W_k = _get_within_cluster_dispersion(X, labels, None, inertia, False, True)
    assert W_k == (4 / 4 + 6 / 5)


def test_generate_random_data():
    shape = (100, 3)
    mins = np.array([0, 1, 2])
    maxs = np.array([2, 4, 8])
    data = _generate_random_data(shape, mins, maxs, None, np.random.RandomState(1))
    assert data.shape == shape
    for i in range(shape[1]):
        assert np.min(data[:, i]) >= mins[i]
        assert np.max(data[:, i]) <= maxs[i]


"""
Tests regarding the Gap Statistic object
"""


def test_simple_GapStatistic():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    gapstat = GapStatistic(random_state=1)
    assert not hasattr(gapstat, "labels_")
    gapstat.fit(X)
    assert gapstat.labels_.dtype == np.int32
    assert gapstat.labels_.shape == labels.shape
    assert gapstat.cluster_centers_.shape == (gapstat.n_clusters_, X.shape[1])
    assert len(np.unique(gapstat.labels_)) == gapstat.n_clusters_
    assert np.array_equal(np.unique(gapstat.labels_), np.arange(gapstat.n_clusters_))
    # Test if random state is working
    gapstat2 = GapStatistic(random_state=1)
    gapstat2.fit(X)
    assert np.array_equal(gapstat.n_clusters_, gapstat2.n_clusters_)
    assert np.array_equal(gapstat.labels_, gapstat2.labels_)
    assert np.array_equal(gapstat.cluster_centers_, gapstat2.cluster_centers_)
    # Test with parameters
    gapstat = GapStatistic(min_n_clusters=2, max_n_clusters=10, n_boots=3, use_principal_components=False,
                           use_log=False, random_state=1, stopping_criterion="max")
    gapstat.fit(X)
    assert gapstat.labels_.dtype == np.int32
    assert gapstat.labels_.shape == labels.shape
    assert gapstat.cluster_centers_.shape == (gapstat.n_clusters_, X.shape[1])
    assert len(np.unique(gapstat.labels_)) == gapstat.n_clusters_
    assert np.array_equal(np.unique(gapstat.labels_), np.arange(gapstat.n_clusters_))
    labels_predict = gapstat.predict(X)
    assert np.array_equal(gapstat.labels_, labels_predict)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_gapstatistic(mock_fig):
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    gapstat = GapStatistic(min_n_clusters=1, max_n_clusters=2, random_state=1, stopping_criterion="ddgap")
    gapstat.fit(X)
    assert None == gapstat.plot(add_ddgap=True)
    gapstat = GapStatistic(min_n_clusters=1, max_n_clusters=2, random_state=1, stopping_criterion="original")
    gapstat.fit(X)
    assert None == gapstat.plot(add_ddgap=True)
import numpy as np
from clustpy.partition import GapStatistic
from clustpy.partition.gapstatistic import _execute_kmeans, _generate_random_data
from sklearn.datasets import make_blobs
from unittest.mock import patch
from sklearn.utils.estimator_checks import check_estimator


def test_gapstatistic_estimator():
    check_estimator(GapStatistic(), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_execute_kmeans():
    X = np.array(
        [[0, 0], [1, -1], [1, 0], [1, 1], [2, 0],
         [4, 0], [5, -1], [5, 0], [5, 1], [6, 0], [7, 0]])
    random_state = np.random.RandomState(1)
    # With number of clusters == 1
    n_clusters = 1
    labels, dispersion = _execute_kmeans(X, n_clusters, use_log=False, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 11))
    assert abs(dispersion - np.sum([np.sum((X - x) ** 2) for x in X]) / (11 * 2)) < 1e-9
    # With number of clusters > 1
    n_clusters = 2
    labels, dispersion = _execute_kmeans(X, n_clusters, use_log=True, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 5 + [1] * 6)) or np.array_equal(labels, np.array([1] * 5 + [0] * 6))
    assert abs(dispersion -
               (np.log(np.sum([np.sum((X[labels == 0] - x) ** 2) for x in X[labels == 0]]) / (5 * 2) +
                       np.sum([np.sum((X[labels == 1] - x) ** 2) for x in X[labels == 1]]) / (6 * 2)))) < 1e-9 or \
           abs(dispersion -
               (np.log(np.sum([np.sum((X[labels == 1] - x) ** 2) for x in X[labels == 1]]) / (5 * 2) +
                       np.sum([np.sum((X[labels == 0] - x) ** 2) for x in X[labels == 0]]) / (6 * 2)))) < 1e-9


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
                           use_log=False, random_state=1)
    gapstat.fit(X)
    assert gapstat.labels_.dtype == np.int32
    assert gapstat.labels_.shape == labels.shape
    assert gapstat.cluster_centers_.shape == (gapstat.n_clusters_, X.shape[1])
    assert len(np.unique(gapstat.labels_)) == gapstat.n_clusters_
    assert np.array_equal(np.unique(gapstat.labels_), np.arange(gapstat.n_clusters_))


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_gapstatistic(mock_fig):
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    gapstat = GapStatistic(min_n_clusters=1, max_n_clusters=2, random_state=1)
    gapstat.fit(X)
    assert None == gapstat.plot()

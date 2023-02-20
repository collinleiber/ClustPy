import numpy as np
from cluspy.partition import GapStatistic
from cluspy.partition.gapstatistic import _execute_kmeans, _generate_random_data
from cluspy.data import load_wine
from unittest.mock import patch


def test_execute_kmeans():
    X = np.array(
        [[0, 0], [1, -1], [1, 0], [1, 1], [2, 0],
         [4, 0], [5, -1], [5, 0], [5, 1], [6, 0], [7, 0]])
    random_state = np.random.RandomState(1)
    # With number of clusters == 1
    n_clusters = 1
    labels, dispersion = _execute_kmeans(X, n_clusters, use_log=False, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 11))
    assert dispersion == np.sum([np.sum((X - x) ** 2) for x in X]) / (11 * 2)
    # With number of clusters > 1
    n_clusters = 2
    labels, dispersion = _execute_kmeans(X, n_clusters, use_log=True, random_state=random_state)
    assert np.array_equal(labels, np.array([0] * 5 + [1] * 6)) or np.array_equal(labels, np.array([1] * 5 + [0] * 6))
    assert dispersion == np.log(np.sum([np.sum((X - x) ** 2) for x in X[labels == 0]]) / (5 * 2) + np.sum(
        [np.sum((X - x) ** 2) for x in X[labels == 1]]) / (6 * 2)) or np.log(dispersion == np.sum(
        [np.sum((X - x) ** 2) for x in X[labels == 1]]) / (5 * 2) + np.sum(
        [np.sum((X - x) ** 2) for x in X[labels == 0]]) / (6 * 2))


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


def test_simple_GapStatistic_with_wine():
    X, labels = load_wine()
    gapstat = GapStatistic()
    assert not hasattr(gapstat, "labels_")
    gapstat.fit(X)
    assert gapstat.labels_.dtype == np.int32
    assert gapstat.labels_.shape == labels.shape
    assert gapstat.cluster_centers_.shape == (gapstat.n_clusters_, X.shape[1])
    assert len(np.unique(gapstat.labels_)) == gapstat.n_clusters_
    assert np.array_equal(np.unique(gapstat.labels_), np.arange(gapstat.n_clusters_))
    # Test with parameters
    gapstat = GapStatistic(min_n_clusters=2, max_n_clusters=10, n_samplings=3, use_principal_components=False,
                           use_log=False, random_state=1)
    gapstat.fit(X)
    assert gapstat.labels_.dtype == np.int32
    assert gapstat.labels_.shape == labels.shape
    assert gapstat.cluster_centers_.shape == (gapstat.n_clusters_, X.shape[1])
    assert len(np.unique(gapstat.labels_)) == gapstat.n_clusters_
    assert np.array_equal(np.unique(gapstat.labels_), np.arange(gapstat.n_clusters_))

    @patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
    def test_plot_subkmeans_result_with_wine(mock_fig):
        X, labels = load_wine()
        gapstat = GapStatistic(min_n_clusters=1, max_n_clusters=2)
        gapstat.fit(X)
        assert None == gapstat.plot()

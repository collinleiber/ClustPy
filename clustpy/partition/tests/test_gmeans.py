import numpy as np
from clustpy.partition import GMeans
from clustpy.partition.gmeans import _anderson_darling_statistic_to_prob
from clustpy.data import load_wine
from scipy.stats import anderson


def test_anderson_darling_statistic_to_prob():
    n_points = 20
    statistic = 0.435
    assert np.isclose(_anderson_darling_statistic_to_prob(statistic, n_points), 0.270, atol=0.001)
    # Example from https://www.spcforexcel.com/knowledge/basic-statistics/anderson-darling-test-for-normality
    data = np.array([3334, 3554, 3625, 3837, 3838])
    ad_result = anderson(data, "norm")
    statistic = ad_result.statistic
    assert np.isclose(statistic, 0.288, atol=0.001)
    assert np.isclose(_anderson_darling_statistic_to_prob(statistic, data.shape[0]), 0.456, atol=0.001)


"""
Tests regarding the GMeans object
"""


def test_simple_GMeans_with_wine():
    X, labels = load_wine()
    gmeans = GMeans()
    assert not hasattr(gmeans, "labels_")
    gmeans.fit(X)
    assert gmeans.labels_.dtype == np.int32
    assert gmeans.labels_.shape == labels.shape
    assert gmeans.cluster_centers_.shape == (gmeans.n_clusters_, X.shape[1])
    assert len(np.unique(gmeans.labels_)) == gmeans.n_clusters_
    assert np.array_equal(np.unique(gmeans.labels_), np.arange(gmeans.n_clusters_))
    # Test with parameters
    gmeans = GMeans(significance=0.1, n_clusters_init=3, max_n_clusters=5, n_split_trials=5, random_state=1)
    gmeans.fit(X)
    assert gmeans.labels_.dtype == np.int32
    assert gmeans.labels_.shape == labels.shape
    assert gmeans.cluster_centers_.shape == (gmeans.n_clusters_, X.shape[1])
    assert len(np.unique(gmeans.labels_)) == gmeans.n_clusters_
    assert np.array_equal(np.unique(gmeans.labels_), np.arange(gmeans.n_clusters_))

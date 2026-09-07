import numpy as np
from clustpy.centroid import ThreeCPO
from clustpy.centroid.threecpo import poisson_dist, poisson_seeding, _poisson_labels_through_centroids, initial_poisson_clustering_labels, get_log_probs_poisson
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator
from unittest.mock import patch


def test_poissonl_estimator():
    check_clustpy_estimator(ThreeCPO(3), ("check_complex_data"))


"""
Tests regarding the PoissonL / PoissonC object
"""


def test_simple_ThreeCPO():
    X, labels = make_blobs(200, 4, centers=3, random_state=1, center_box=(10, 20))
    threecpo = ThreeCPO(3, random_state=1)
    assert not hasattr(threecpo, "labels_")
    threecpo.fit(X)
    assert threecpo.labels_.dtype == np.int32
    assert threecpo.labels_.shape == labels.shape
    assert threecpo.column_lambdas_.shape == (threecpo.n_clusters_, X.shape[1])
    assert len(np.unique(threecpo.labels_)) == threecpo.n_clusters_
    assert np.array_equal(np.unique(threecpo.labels_), np.arange(threecpo.n_clusters_))
    # Test if random state is working
    threecpo2 = ThreeCPO(3, random_state=1)
    threecpo2.fit(X)
    assert np.array_equal(threecpo.n_clusters_, threecpo2.n_clusters_)
    assert np.array_equal(threecpo.labels_, threecpo2.labels_)
    assert np.array_equal(threecpo.column_lambdas_, threecpo2.column_lambdas_)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_subkmeans_result(mock_fig):
    X, labels = make_blobs(200, 4, centers=3, random_state=1, center_box=(10, 20))
    tcpo = ThreeCPO(3, max_iter=3, track_reward=True, n_init=1)
    tcpo.fit(X)
    assert None == tcpo.plot_reward()
    assert None == tcpo.plot_reward(True, False, False)

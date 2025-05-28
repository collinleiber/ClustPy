import numpy as np
from clustpy.partition import XMeans
from clustpy.partition.xmeans import _execute_two_means, _merge_clusters, _initial_kmeans_clusters
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.utils.checks import check_clustpy_estimator


def test_xmeans_estimator():
    check_clustpy_estimator(XMeans(), ("check_complex_data"))


def test_initial_kmeans_clusters():
    X = np.array(
        [[0, 0], [1, -1], [1, 0], [1, 1], [2, 0],
         [4, 0], [5, -1], [5, 0], [5, 1], [6, 0]])
    random_state = np.random.RandomState(1)
    # With number of clusters == 1
    n_clusters_init = 1
    n_clusters, labels, centers, kmeans_error = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    assert n_clusters == 1
    assert np.array_equal(labels, np.array([0] * 10))
    assert np.array_equal(centers, np.array([[3, 0]]))
    # With number of clusters > 1
    n_clusters_init = 2
    n_clusters, labels, centers, kmeans_error = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    assert n_clusters == 2
    assert np.array_equal(labels, np.array([0] * 5 + [1] * 5)) or np.array_equal(labels, np.array([1] * 5 + [0] * 5))
    assert np.array_equal(centers, np.array([[1, 0], [5, 0]])) or np.array_equal(centers, np.array([[5, 0], [1, 0]]))
    # With starting centers
    n_clusters_init = np.array([[0, 0], [6, 0]])
    n_clusters, labels, centers, kmeans_error = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    assert n_clusters == 2
    assert np.array_equal(labels, np.array([0] * 5 + [1] * 5)) or np.array_equal(labels, np.array([1] * 5 + [0] * 5))
    assert np.array_equal(centers, np.array([[1, 0], [5, 0]])) or np.array_equal(centers, np.array([[5, 0], [1, 0]]))


def test_execute_two_means():
    X = np.array([[50, 50],
                  [1, 2],
                  [2, 3],
                  [3, 1],
                  [51, 51],
                  [11, 12],
                  [12, 13],
                  [52, 52],
                  [13, 14],
                  [14, 11],
                  [53, 53]])
    ids_in_each_cluster = [np.array([0, 4, 7, 10]), np.array([1, 2, 3, 5, 6, 8, 9])]
    cluster_id_to_split = 1
    centers = np.array([[51.5, 51.5], [8, 8]])
    random_state = np.random.RandomState(1)
    labels, centers, _ = _execute_two_means(X, ids_in_each_cluster, cluster_id_to_split, centers,
                                            10, random_state)
    assert nmi(labels, np.array([0, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0]))
    assert np.array_equal(centers, np.array([[51.5, 51.5], [2, 2], [12.5, 12.5]])) or np.array_equal(centers, np.array(
        [[51.5, 51.5], [12.5, 12.5], [2, 2]]))


def test_merge_clusters():
    X = np.array(
        [[0, 0], [1, -1], [1, 0], [1, 1], [2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
         [3, -2], [3, -1], [3, 0], [3, 1], [3, 2], [4, -1], [4, 0], [4, 1], [5, 0]])
    n_clusters = 2
    labels = np.array([0] * 9 + [1] * 9)
    centers = np.array([[1., 0.], [4., 0.]])
    ids_in_each_cluster = [np.array([i for i in range(9)]), np.array([i for i in range(9, 18)])]
    cluster_sizes = np.array([9, 9])
    cluster_variances = np.array([np.sum((X[ids_in_each_cluster[c]] - centers[c]) ** 2) / (
            cluster_sizes[c] - 1) for c in range(n_clusters)])
    n_clusters, labels, centers = _merge_clusters(X, n_clusters, labels, centers, ids_in_each_cluster, cluster_sizes,
                                                  cluster_variances)
    assert n_clusters == 1
    assert np.array_equal(labels, np.array([0] * 18))
    assert np.array_equal(centers, np.array([[2.5, 0]]))


"""
Tests regarding the XMeans object
"""


def test_simple_XMeans():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    xmeans = XMeans(random_state=1)
    assert not hasattr(xmeans, "labels_")
    xmeans.fit(X)
    assert xmeans.labels_.dtype == np.int32
    assert xmeans.labels_.shape == labels.shape
    assert xmeans.cluster_centers_.shape == (xmeans.n_clusters_, X.shape[1])
    assert len(np.unique(xmeans.labels_)) == xmeans.n_clusters_
    assert np.array_equal(np.unique(xmeans.labels_), np.arange(xmeans.n_clusters_))
    # Test if random state is working
    xmeans2 = XMeans(random_state=1)
    xmeans2.fit(X)
    assert np.array_equal(xmeans.n_clusters_, xmeans2.n_clusters_)
    assert np.array_equal(xmeans.labels_, xmeans2.labels_)
    assert np.array_equal(xmeans.cluster_centers_, xmeans2.cluster_centers_)
    # Test with parameters
    xmeans = XMeans(n_clusters_init=3, max_n_clusters=5, check_global_score=False, allow_merging=True, n_split_trials=5,
                    random_state=1)
    xmeans.fit(X)
    assert xmeans.labels_.dtype == np.int32
    assert xmeans.labels_.shape == labels.shape
    assert xmeans.cluster_centers_.shape == (xmeans.n_clusters_, X.shape[1])
    assert len(np.unique(xmeans.labels_)) == xmeans.n_clusters_
    assert np.array_equal(np.unique(xmeans.labels_), np.arange(xmeans.n_clusters_))
    labels_predict = xmeans.predict(X)
    assert np.array_equal(xmeans.labels_, labels_predict)

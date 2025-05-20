import numpy as np
from clustpy.partition import ProjectedDipMeans
from clustpy.partition.projected_dipmeans import _get_projected_data
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator


def test_projected_dipmeans_estimator():
    check_clustpy_estimator(ProjectedDipMeans(), ("check_complex_data"))


def test_get_projected_data():
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],
                  [8, 9, 7],
                  [5, 6, 4],
                  [2, 3, 1]])
    random_state = np.random.RandomState(1)
    projected_data = _get_projected_data(X, 2, random_state)
    assert projected_data.shape == (7, 8)


"""
Tests regarding the ProjectedDipMeans object
"""


def test_simple_ProjectedDipMeans():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    pdipmeans = ProjectedDipMeans(random_state=1)
    assert not hasattr(pdipmeans, "labels_")
    pdipmeans.fit(X)
    assert pdipmeans.labels_.dtype == np.int32
    assert pdipmeans.labels_.shape == labels.shape
    assert pdipmeans.cluster_centers_.shape == (pdipmeans.n_clusters_, X.shape[1])
    assert len(np.unique(pdipmeans.labels_)) == pdipmeans.n_clusters_
    assert np.array_equal(np.unique(pdipmeans.labels_), np.arange(pdipmeans.n_clusters_))
    # Test if random state is working
    pdipmeans2 = ProjectedDipMeans(random_state=1)
    pdipmeans2.fit(X)
    assert np.array_equal(pdipmeans.n_clusters_, pdipmeans2.n_clusters_)
    assert np.array_equal(pdipmeans.labels_, pdipmeans2.labels_)
    assert np.array_equal(pdipmeans.cluster_centers_, pdipmeans2.cluster_centers_)
    # Test with parameters
    pdipmeans = ProjectedDipMeans(significance=0.1, n_random_projections=3, pval_strategy="bootstrap", n_boots=10,
                                  n_split_trials=2, n_clusters_init=3, max_n_clusters=5, random_state=1)
    pdipmeans.fit(X)
    assert pdipmeans.labels_.dtype == np.int32
    assert pdipmeans.labels_.shape == labels.shape
    assert pdipmeans.cluster_centers_.shape == (pdipmeans.n_clusters_, X.shape[1])
    assert len(np.unique(pdipmeans.labels_)) == pdipmeans.n_clusters_
    assert np.array_equal(np.unique(pdipmeans.labels_), np.arange(pdipmeans.n_clusters_))
    labels_predict = pdipmeans.predict(X)
    assert np.array_equal(pdipmeans.labels_, labels_predict)

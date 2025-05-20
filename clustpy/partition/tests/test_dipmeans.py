import numpy as np
from clustpy.partition import DipMeans
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator


def test_dipmeans_estimator():
    check_clustpy_estimator(DipMeans(), ("check_complex_data"))

"""
Tests regarding the DipMeans object
"""


def test_simple_DipMeans():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    dipmeans = DipMeans(random_state=1)
    assert not hasattr(dipmeans, "labels_")
    dipmeans.fit(X)
    assert dipmeans.labels_.dtype == np.int32
    assert dipmeans.labels_.shape == labels.shape
    assert dipmeans.cluster_centers_.shape == (dipmeans.n_clusters_, X.shape[1])
    assert len(np.unique(dipmeans.labels_)) == dipmeans.n_clusters_
    assert np.array_equal(np.unique(dipmeans.labels_), np.arange(dipmeans.n_clusters_))
    # Test if random state is working
    dipmeans2 = DipMeans(random_state=1)
    dipmeans2.fit(X)
    assert np.array_equal(dipmeans.n_clusters_, dipmeans2.n_clusters_)
    assert np.array_equal(dipmeans.labels_, dipmeans2.labels_)
    assert np.array_equal(dipmeans.cluster_centers_, dipmeans2.cluster_centers_)
    # Test with parameters
    dipmeans = DipMeans(significance=0.1, split_viewers_threshold=0.001, pval_strategy="bootstrap", n_boots=10,
                        n_split_trials=2, n_clusters_init=3, max_n_clusters=5, random_state=1)
    dipmeans.fit(X)
    assert dipmeans.labels_.dtype == np.int32
    assert dipmeans.labels_.shape == labels.shape
    assert dipmeans.cluster_centers_.shape == (dipmeans.n_clusters_, X.shape[1])
    assert len(np.unique(dipmeans.labels_)) == dipmeans.n_clusters_
    assert np.array_equal(np.unique(dipmeans.labels_), np.arange(dipmeans.n_clusters_))
    labels_predict = dipmeans.predict(X)
    assert np.array_equal(dipmeans.labels_, labels_predict)

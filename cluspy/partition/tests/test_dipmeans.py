import numpy as np
from cluspy.partition import DipMeans
from cluspy.data import load_wine

"""
Tests regarding the DipMeans object
"""


def test_simple_DipMeans_with_wine():
    X, labels = load_wine()
    dipmeans = DipMeans()
    assert not hasattr(dipmeans, "labels_")
    dipmeans.fit(X)
    assert dipmeans.labels_.dtype == np.int32
    assert dipmeans.labels_.shape == labels.shape
    assert dipmeans.cluster_centers_.shape == (dipmeans.n_clusters_, X.shape[1])
    assert len(np.unique(dipmeans.labels_)) == dipmeans.n_clusters_
    assert np.array_equal(np.unique(dipmeans.labels_), np.arange(dipmeans.n_clusters_))
    # Test with parameters
    dipmeans = DipMeans(significance=0.1, split_viewers_threshold=0.001, pval_strategy="bootstrap", n_boots=10,
                        n_split_trials=2, n_clusters_init=3, max_n_clusters=5, random_state=1)
    dipmeans.fit(X)
    assert dipmeans.labels_.dtype == np.int32
    assert dipmeans.labels_.shape == labels.shape
    assert dipmeans.cluster_centers_.shape == (dipmeans.n_clusters_, X.shape[1])
    assert len(np.unique(dipmeans.labels_)) == dipmeans.n_clusters_
    assert np.array_equal(np.unique(dipmeans.labels_), np.arange(dipmeans.n_clusters_))

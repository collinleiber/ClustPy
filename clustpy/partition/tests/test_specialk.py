import numpy as np
from clustpy.partition import SpecialK
from sklearn.datasets import make_blobs

"""
Tests regarding the SpecialK object
"""


def test_simple_SpecialK():
    X, labels = make_blobs(250, 4, centers=3, random_state=1)
    specialk = SpecialK(random_state=1)
    assert not hasattr(specialk, "labels_")
    specialk.fit(X)
    assert specialk.labels_.dtype == np.int32
    assert specialk.labels_.shape == labels.shape
    assert len(np.unique(specialk.labels_)) == specialk.n_clusters_
    assert np.array_equal(np.unique(specialk.labels_), np.arange(specialk.n_clusters_))
    # Test if random state is working
    dipmeans2 = SpecialK(random_state=1)
    dipmeans2.fit(X)
    assert np.array_equal(specialk.n_clusters_, dipmeans2.n_clusters_)
    assert np.array_equal(specialk.labels_, dipmeans2.labels_)
    # Test with parameters
    specialk = SpecialK(significance=0.1, n_dimensions=150, use_neighborhood_adj_mat=False, n_neighbors=5,
                        n_cluster_pairs_to_consider=None, max_n_clusters=5, random_state=1, debug=True)
    specialk.fit(X)
    assert specialk.labels_.dtype == np.int32
    assert specialk.labels_.shape == labels.shape
    assert len(np.unique(specialk.labels_)) == specialk.n_clusters_
    assert np.array_equal(np.unique(specialk.labels_), np.arange(specialk.n_clusters_))

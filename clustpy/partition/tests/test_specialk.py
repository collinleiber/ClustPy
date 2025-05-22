import numpy as np
from clustpy.partition import SpecialK
from clustpy.partition.specialk import _get_neighborhood_adjacency_matrix
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator
import pytest

@pytest.mark.skip(reason="There seems to be a non-trivial error. Needs to be fixed.")
def test_specialk_estimator():
    check_clustpy_estimator(SpecialK(), ("check_complex_data"))

"""
Tests regarding the SpecialK object
"""


def test_simple_SpecialK():
    X, labels = make_blobs(250, 4, centers=3, random_state=1)
    n_neighbors = 10
    specialk = SpecialK(random_state=1, n_neighbors=n_neighbors)
    assert not hasattr(specialk, "labels_")
    specialk.fit(X)
    assert specialk.labels_.dtype == np.int32
    assert specialk.labels_.shape == labels.shape
    assert len(np.unique(specialk.labels_)) == specialk.n_clusters_
    assert np.array_equal(np.unique(specialk.labels_), np.arange(specialk.n_clusters_))
    # Test if random state is working
    similarity_matrix = _get_neighborhood_adjacency_matrix(X, n_neighbors=n_neighbors)
    specialk2 = SpecialK(random_state=1, similarity_matrix=similarity_matrix, n_neighbors=n_neighbors)
    specialk2.fit(X)
    assert np.array_equal(specialk.n_clusters_, specialk2.n_clusters_)
    assert np.array_equal(specialk.labels_, specialk2.labels_)
    # Test with parameters
    specialk = SpecialK(significance=0.1, n_dimensions=150, similarity_matrix='SAM', n_neighbors=5,
                        n_cluster_pairs_to_consider=None, max_n_clusters=5, random_state=1, debug=True)
    specialk.fit(X)
    assert specialk.labels_.dtype == np.int32
    assert specialk.labels_.shape == labels.shape
    assert len(np.unique(specialk.labels_)) == specialk.n_clusters_
    assert np.array_equal(np.unique(specialk.labels_), np.arange(specialk.n_clusters_))

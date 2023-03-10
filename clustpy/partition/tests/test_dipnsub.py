import numpy as np
from clustpy.partition import DipNSub
from clustpy.data import create_subspace_data

"""
Tests regarding the DipNSub object
"""


def test_simple_DipNSub():
    X, labels = create_subspace_data(200, subspace_features=(3, 5), random_state=1)
    dipnsub = DipNSub(random_state=1)
    assert not hasattr(dipnsub, "labels_")
    dipnsub.fit(X)
    assert dipnsub.labels_.dtype == np.int32
    assert dipnsub.labels_.shape == labels.shape
    assert len(np.unique(dipnsub.labels_)) == dipnsub.n_clusters_
    # Test with parameters
    dipnsub = DipNSub(significance=0.5, threshold=0.2, step_size=0.2, momentum=0.8,
                      n_starting_vectors=5, add_tails=False, outliers=True,
                      consider_duplicates=True, random_state=1, debug=True)
    dipnsub.fit(X)
    assert dipnsub.labels_.dtype == np.int32
    assert dipnsub.labels_.shape == labels.shape
    assert len(np.unique(dipnsub.labels_)) == dipnsub.n_clusters_ + 1

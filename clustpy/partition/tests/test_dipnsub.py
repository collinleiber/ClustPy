import numpy as np
from clustpy.partition import DipNSub
from clustpy.data import load_wine

"""
Tests regarding the DipNSub object
"""


def test_simple_DipNSub_with_wine():
    X, labels = load_wine()
    dipnsub = DipNSub()
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

import numpy as np
from clustpy.partition import SkinnyDip, UniDip
from clustpy.data import load_wine

"""
Tests regarding the SkinnyDip object
"""


def test_simple_SkinnyDip_with_wine():
    X, labels = load_wine()
    skinny = SkinnyDip(random_state=1)
    assert not hasattr(skinny, "labels_")
    skinny.fit(X)
    assert skinny.labels_.dtype == np.int32
    assert skinny.labels_.shape == labels.shape
    assert len(np.unique(skinny.labels_)) == skinny.n_clusters_ + 1
    assert np.array_equal(np.unique(skinny.labels_), np.append([-1], np.arange(skinny.n_clusters_)))
    # Test with parameters
    skinny = SkinnyDip(significance=0.01, pval_strategy="bootstrap", n_boots=10, add_tails=True, outliers=False,
                       max_cluster_size_diff_factor=3, debug=True, random_state=1)
    assert not hasattr(skinny, "labels_")
    skinny.fit(X)
    assert skinny.labels_.dtype == np.int32
    assert skinny.labels_.shape == labels.shape
    assert len(np.unique(skinny.labels_)) == skinny.n_clusters_
    assert np.array_equal(np.unique(skinny.labels_), np.arange(skinny.n_clusters_))


"""
Tests regarding the UniDip object
"""


def test_simple_UniDip_with_1d_of_wine():
    X, labels = load_wine()
    X = X[:, -1]
    unidip = UniDip()
    assert not hasattr(unidip, "labels_")
    unidip.fit(X)
    assert unidip.labels_.dtype == np.int32
    assert unidip.labels_.shape == labels.shape
    assert len(np.unique(unidip.labels_)) == unidip.n_clusters_ + 1
    assert np.array_equal(np.unique(unidip.labels_), np.append([-1], np.arange(unidip.n_clusters_)))
    # Test with parameters
    unidip = UniDip(significance=0.01, pval_strategy="bootstrap", n_boots=10, add_tails=True, outliers=False,
                    max_cluster_size_diff_factor=3, random_state=1)
    assert not hasattr(unidip, "labels_")
    unidip.fit(X)
    assert unidip.labels_.dtype == np.int32
    assert unidip.labels_.shape == labels.shape
    assert len(np.unique(unidip.labels_)) == unidip.n_clusters_
    assert np.array_equal(np.unique(unidip.labels_), np.arange(unidip.n_clusters_))

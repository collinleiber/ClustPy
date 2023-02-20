import numpy as np
from cluspy.partition import LDAKmeans
from cluspy.data import load_wine

"""
Tests regarding the LDAKmeans object
"""


def test_simple_LDAKmeans_with_wine():
    X, labels = load_wine()
    ldakm = LDAKmeans(3)
    assert not hasattr(ldakm, "labels_")
    ldakm.fit(X)
    assert ldakm.labels_.dtype == np.int32
    assert ldakm.labels_.shape == labels.shape
    assert ldakm.cluster_centers_.shape == (ldakm.n_clusters, 2)
    # Test with parameters
    ldakm = LDAKmeans(3, n_dims=5, max_iter=10, kmeans_repetitions=5, random_state=1)
    ldakm.fit(X)
    assert ldakm.labels_.dtype == np.int32
    assert ldakm.labels_.shape == labels.shape
    assert ldakm.cluster_centers_.shape == (ldakm.n_clusters, 5)


def test_transform_subspace():
    X, labels = load_wine()
    X = X[:, :4]
    ldakm = LDAKmeans(3, n_dims=3, max_iter=10, kmeans_repetitions=1)
    ldakm.fit(X)
    # Overwrite rotation
    ldakm.rotation_ = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]])
    X_transformed = ldakm.transform_subspace(X)
    assert np.array_equal(X_transformed, np.c_[X[:, 1], X[:, 0], X[:, 3]])

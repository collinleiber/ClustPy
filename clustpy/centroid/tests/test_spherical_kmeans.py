import numpy as np
from clustpy.centroid import SphericalKMeans
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator


def test_spherical_kmeans_estimator():
    check_clustpy_estimator(SphericalKMeans(3), ("check_complex_data"))


"""
Tests regarding the SphericalKMeans object
"""


def test_simple_SphericalKMeans():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    skm = SphericalKMeans(3, random_state=1)
    assert not hasattr(skm, "labels_")
    skm.fit(X)
    assert skm.labels_.dtype == np.int32
    assert skm.labels_.shape == labels.shape
    assert skm.column_lambdas_.shape == (skm.n_clusters_, X.shape[1])
    assert len(np.unique(skm.labels_)) == skm.n_clusters_
    assert np.array_equal(np.unique(skm.labels_), np.arange(skm.n_clusters_))
    labels_predict = skm.predict(X)
    assert np.array_equal(skm.labels_, labels_predict)
    # Test if random state is working
    skm2 = SphericalKMeans(3, random_state=1)
    skm2.fit(X)
    assert np.array_equal(skm.n_clusters_, skm2.n_clusters_)
    assert np.array_equal(skm.labels_, skm2.labels_)
    assert np.array_equal(skm.column_lambdas_, skm2.column_lambdas_)

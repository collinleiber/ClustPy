import numpy as np
from clustpy.centroid import PoissonL, PoissonC
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator


def test_poissonl_estimator():
    check_clustpy_estimator(PoissonL(3), ("check_complex_data"))


def test_poissonc_estimator():
    check_clustpy_estimator(PoissonC(3), ("check_complex_data"))


"""
Tests regarding the PoissonL / PoissonC object
"""


def test_simple_PoissonL():
    X, labels = make_blobs(200, 4, centers=3, random_state=1, center_box=(10, 20))
    poissonl = PoissonL(3, random_state=1)
    assert not hasattr(poissonl, "labels_")
    poissonl.fit(X)
    assert poissonl.labels_.dtype == np.int32
    assert poissonl.labels_.shape == labels.shape
    assert poissonl.column_lambdas_.shape == (poissonl.n_clusters_, X.shape[1])
    assert len(np.unique(poissonl.labels_)) == poissonl.n_clusters_
    assert np.array_equal(np.unique(poissonl.labels_), np.arange(poissonl.n_clusters_))
    labels_predict = poissonl.predict(X)
    assert np.array_equal(poissonl.labels_, labels_predict)
    # Test if random state is working
    poissonl2 = PoissonL(3, random_state=1)
    poissonl2.fit(X)
    assert np.array_equal(poissonl.n_clusters_, poissonl2.n_clusters_)
    assert np.array_equal(poissonl.labels_, poissonl2.labels_)
    assert np.array_equal(poissonl.column_lambdas_, poissonl2.column_lambdas_)

def test_simple_PoissonC():
    X, labels = make_blobs(200, 4, centers=3, random_state=1, center_box=(10, 20))
    poissonc = PoissonC(3, random_state=1)
    assert not hasattr(poissonc, "labels_")
    poissonc.fit(X)
    assert poissonc.labels_.dtype == np.int32
    assert poissonc.labels_.shape == labels.shape
    assert poissonc.column_lambdas_.shape == (poissonc.n_clusters_, X.shape[1])
    assert len(np.unique(poissonc.labels_)) == poissonc.n_clusters_
    assert np.array_equal(np.unique(poissonc.labels_), np.arange(poissonc.n_clusters_))
    labels_predict = poissonc.predict(X)
    assert np.array_equal(poissonc.labels_, labels_predict)
    # Test if random state is working
    poissonc2 = PoissonC(3, random_state=1)
    poissonc2.fit(X)
    assert np.array_equal(poissonc.n_clusters_, poissonc2.n_clusters_)
    assert np.array_equal(poissonc.labels_, poissonc2.labels_)
    assert np.array_equal(poissonc.column_lambdas_, poissonc2.column_lambdas_)

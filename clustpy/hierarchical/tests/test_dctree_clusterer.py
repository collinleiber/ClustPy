from clustpy.hierarchical import DCTree_Clusterer
from clustpy.utils.checks import check_clustpy_estimator
from sklearn.datasets import make_blobs
import numpy as np


def test_dctree_clusterer_estimator():
    check_clustpy_estimator(DCTree_Clusterer(min_points=2), ("check_complex_data"))


def test_simple_dctree_clusterer():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    dctree_clusterer = DCTree_Clusterer()
    assert not hasattr(dctree_clusterer, "labels_")
    dctree_clusterer.fit(X)
    assert dctree_clusterer.labels_.dtype == np.int32
    assert dctree_clusterer.labels_.shape == labels.shape

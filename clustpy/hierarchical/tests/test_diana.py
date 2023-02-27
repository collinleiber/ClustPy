from clustpy.data import load_wine
from clustpy.hierarchical import Diana
from clustpy.hierarchical.diana import _split_cluster, _get_cluster_with_max_diameter
import numpy as np


def test_get_cluster_with_max_diameter():
    # Uses the example from the original paper
    global_distance_matrix = np.array([[0, 2, 6, 10, 9],
                                       [2, 0, 5, 9, 8],
                                       [6, 5, 0, 4, 5],
                                       [10, 9, 4, 0, 3],
                                       [9, 8, 5, 3, 0]])
    # First iteration
    labels = np.array([0, 0, 0, 0, 0])
    split_cluster_id, cluster_distance_matrix = _get_cluster_with_max_diameter(global_distance_matrix, labels, 1, 0)
    assert split_cluster_id == 0
    assert np.array_equal(cluster_distance_matrix, global_distance_matrix)
    # Second iteration
    labels = np.array([0, 0, 1, 1, 1])
    split_cluster_id, cluster_distance_matrix = _get_cluster_with_max_diameter(global_distance_matrix, labels, 2, 0)
    assert split_cluster_id == 1
    assert np.array_equal(cluster_distance_matrix, np.array([[0, 4, 5],
                                                             [4, 0, 3],
                                                             [5, 3, 0]]))


def test_split_cluster():
    # Uses the example from the original paper
    # First iteration
    cluster_distance_matrix = np.array([[0, 2, 6, 10, 9],
                                        [2, 0, 5, 9, 8],
                                        [6, 5, 0, 4, 5],
                                        [10, 9, 4, 0, 3],
                                        [9, 8, 5, 3, 0]])
    labels = _split_cluster(cluster_distance_matrix, 0, 1)
    assert np.array_equal(labels, np.array([1, 1, 0, 0, 0]))
    # Second iteration
    cluster_distance_matrix = np.array([[0, 4, 5],
                                        [4, 0, 3],
                                        [5, 3, 0]])
    labels = _split_cluster(cluster_distance_matrix, 0, 2)
    assert np.array_equal(labels, np.array([2, 0, 0]))


"""
Tests regarding the DIANA object
"""


def test_simple_Diana_with_wine():
    X, labels = load_wine()
    # Without any parameters
    diana = Diana()
    assert not hasattr(diana, "labels_")
    diana.fit(X)
    assert diana.labels_.dtype == np.int32
    assert diana.labels_.shape == labels.shape
    assert np.unique(diana.labels_).shape[0] == X.shape[0]
    assert len(diana.tree_) == X.shape[0] - 1
    # With n_clusters specified as 3
    diana = Diana(n_clusters=3)
    assert not hasattr(diana, "labels_")
    diana.fit(X)
    assert diana.labels_.dtype == np.int32
    assert diana.labels_.shape == labels.shape
    assert np.unique(diana.labels_).shape[0] == 3
    assert len(diana.tree_) == 2
    # With n_clusters specified as 3 and construct_full_tree = True
    diana = Diana(n_clusters=3, construct_full_tree=True)
    assert not hasattr(diana, "labels_")
    diana.fit(X)
    assert diana.labels_.dtype == np.int32
    assert diana.labels_.shape == labels.shape
    assert np.unique(diana.labels_).shape[0] == 3
    assert len(diana.tree_) == X.shape[0] - 1


def test_pruning_tree():
    X, labels = load_wine()
    diana = Diana()
    diana.fit(X)
    assert np.unique(diana.labels_).shape[0] == X.shape[0]
    labels_pruned = diana.prune_tree(4)
    assert np.unique(labels_pruned).shape[0] == 5

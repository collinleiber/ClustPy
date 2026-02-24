from clustpy.metrics import dendrogram_purity, leaf_purity, node_purity
from clustpy.metrics.hierarchical_metrics import _get_parent_matrix
from clustpy.hierarchical._cluster_tree import BinaryClusterTree
import numpy as np


def test_node_purity():
    bct = BinaryClusterTree()
    node_023, node_145 = bct.split_cluster(0)
    node_03, node_2 = bct.split_cluster(0)
    node_0, node_3 = bct.split_cluster(0)
    node_15, node_4 = bct.split_cluster(1)
    node_1, node_5 = bct.split_cluster(1)
    l1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    l2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert node_purity(bct.root_node_, l1, l2) == 1/5
    assert node_purity(node_023, l1, l2) == 1/3
    assert node_purity(node_145, l1, l2) == 1/2
    assert node_purity(node_03, l1, l2) == 1/2
    assert node_purity(node_2, l1, l2) == 1.
    assert node_purity(node_0, l1, l2) == 1.
    assert node_purity(node_3, l1, l2) == 1.
    assert node_purity(node_15, l1, l2) == 1.
    assert node_purity(node_4, l1, l2) == 1.
    assert node_purity(node_1, l1, l2) == 1.
    assert node_purity(node_5, l1, l2) == 0.


def test_leaf_purity():
    bct = BinaryClusterTree()
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(1)
    bct.split_cluster(1)
    l1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    l2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert leaf_purity(bct, l1, l2) == 1.0
    l2 = np.array([1, 5, 5, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert leaf_purity(bct, l1, l2) == 1.0
    l2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert np.isclose(leaf_purity(bct, l1, l2), 1 / 5)
    l2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3])
    assert np.isclose(leaf_purity(bct, l1, l2), (4 * 3) / 15)
    bct.split_cluster(5)
    bct.split_cluster(6)
    bct.split_cluster(7)
    bct.split_cluster(8)
    l2 = np.array([0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9])
    l2_pruned = bct.prune_to_n_leaf_nodes(5, l2)
    assert np.array_equal(l2_pruned, np.array([0, 0, 1, 2, 2, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1]))
    assert leaf_purity(bct, l1, l2) == leaf_purity(bct, l1, l2_pruned)
    assert np.isclose(leaf_purity(bct, l1, l2), 10 / 15)


def test_dendrogram_purity():
    bct = BinaryClusterTree()
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(1)
    bct.split_cluster(1)
    l1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    l2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert dendrogram_purity(bct, l1, l2) == 1.0
    l2 = np.array([1, 5, 5, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert dendrogram_purity(bct, l1, l2) == 1.0
    l2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert np.isclose(dendrogram_purity(bct, l1, l2), 1 / 5)
    l2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3])
    assert np.isclose(dendrogram_purity(bct, l1, l2), (3 * 3 * 1 + 2 * 3 * 0.5) / 15)


def test_get_parent_matrix():
    # First test
    dendrogram = np.array([
        [0, 1, 0, 2],
        [4, 2, 0, 3],
        [5, 3, 0, 4]
    ])
    parent_matrix = _get_parent_matrix(dendrogram)
    expected_parent_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ])
    assert np.array_equal(expected_parent_matrix, parent_matrix)
    # Second test
    dendrogram = np.array([
        [0, 3, 0, 2],
        [1, 5, 0, 2],
        [6, 2, 0, 3],
        [7, 4, 0, 3],
        [8, 9, 0, 6]
    ])
    parent_matrix = _get_parent_matrix(dendrogram)
    expected_parent_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ])
    assert np.array_equal(expected_parent_matrix, parent_matrix)

from clustpy.metrics import dendrogram_purity, leaf_purity
from clustpy.hierarchical._cluster_tree import BinaryClusterTree
import numpy as np


def test_leaf_purity():
    bct = BinaryClusterTree()
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(1)
    bct.split_cluster(1)
    l1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    l2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert leaf_purity(l1, l2, bct) == 1.0
    l2 = np.array([1, 5, 5, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert leaf_purity(l1, l2, bct) == 1.0
    l2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert np.isclose(leaf_purity(l1, l2, bct), 1 / 5)
    l2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3])
    assert np.isclose(leaf_purity(l1, l2, bct), (4 * 3) / 15)
    bct.split_cluster(5)
    bct.split_cluster(6)
    bct.split_cluster(7)
    bct.split_cluster(8)
    l2 = np.array([0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9])
    l2_pruned = bct.prune_to_n_leaf_nodes(5, l2)
    assert np.array_equal(l2_pruned, np.array([0, 0, 1, 2, 2, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1]))
    assert leaf_purity(l1, l2, bct) == leaf_purity(l1, l2_pruned, bct)
    assert np.isclose(leaf_purity(l1, l2, bct), 10 / 15)


def test_dendrogram_purity():
    bct = BinaryClusterTree()
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(1)
    bct.split_cluster(1)
    l1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    l2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert dendrogram_purity(l1, l2, bct) == 1.0
    l2 = np.array([1, 5, 5, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0])
    assert dendrogram_purity(l1, l2, bct) == 1.0
    l2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert np.isclose(dendrogram_purity(l1, l2, bct), 1 / 5)
    l2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3])
    assert np.isclose(dendrogram_purity(l1, l2, bct), (3 * 3 * 1 + 2 * 3 * 0.5) / 15)

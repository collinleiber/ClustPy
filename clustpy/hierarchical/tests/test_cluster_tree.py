import numpy as np

from clustpy.hierarchical._cluster_tree import BinaryClusterTree, _ClusterTreeNode


def _check_node(node: _ClusterTreeNode, is_leaf_node: bool, node_id: int, labels: list, tree: BinaryClusterTree):
    assert node.is_leaf_node() == is_leaf_node
    assert node.node_id_ == node_id
    assert node.labels == labels
    if node.is_leaf_node():
        assert node.left_node_ is None
        assert node.right_node_ is None
    else:
        assert node.left_node_ is not None
        assert node.right_node_ is not None


def test_BinaryClusterTree():
    bct = BinaryClusterTree()
    assert bct.n_leaf_nodes_ == 1
    assert bct.n_split_nodes_ == 0
    _check_node(bct.root_node_, True, 0, [0], bct)
    # == New split
    bct.split_cluster(0)
    assert bct.n_leaf_nodes_ == 2
    assert bct.n_split_nodes_ == 1
    _check_node(bct.root_node_, False, 0, [0, 1], bct)
    _check_node(bct.root_node_.left_node_, True, 1, [0], bct)
    _check_node(bct.root_node_.right_node_, True, 2, [1], bct)
    assert bct.root_node_.left_node_.get_sibling() == bct.root_node_.right_node_
    assert bct.root_node_.right_node_.get_sibling() == bct.root_node_.left_node_
    # == New split
    bct.split_cluster(0)
    assert bct.n_leaf_nodes_ == 3
    assert bct.n_split_nodes_ == 2
    _check_node(bct.root_node_, False, 0, [0, 1, 2], bct)
    _check_node(bct.root_node_.left_node_, False, 1, [0, 2], bct)
    _check_node(bct.root_node_.left_node_.left_node_, True, 3, [0], bct)
    _check_node(bct.root_node_.left_node_.right_node_, True, 4, [2], bct)
    _check_node(bct.root_node_.right_node_, True, 2, [1], bct)
    assert bct.root_node_.left_node_.left_node_.get_sibling() == bct.root_node_.left_node_.right_node_
    assert bct.root_node_.left_node_.right_node_.get_sibling() == bct.root_node_.left_node_.left_node_
    # == New split
    bct.split_cluster(0)
    assert bct.n_leaf_nodes_ == 4
    assert bct.n_split_nodes_ == 3
    _check_node(bct.root_node_, False, 0, [0, 1, 2, 3], bct)
    _check_node(bct.root_node_.left_node_, False, 1, [0, 2, 3], bct)
    _check_node(bct.root_node_.left_node_.left_node_, False, 3, [0, 3], bct)
    _check_node(bct.root_node_.left_node_.left_node_.left_node_, True, 5, [0], bct)
    _check_node(bct.root_node_.left_node_.left_node_.right_node_, True, 6, [3], bct)
    _check_node(bct.root_node_.left_node_.right_node_, True, 4, [2], bct)
    _check_node(bct.root_node_.right_node_, True, 2, [1], bct)
    assert bct.root_node_.left_node_.left_node_.left_node_.get_sibling() == bct.root_node_.left_node_.left_node_.right_node_
    assert bct.root_node_.left_node_.left_node_.right_node_.get_sibling() == bct.root_node_.left_node_.left_node_.left_node_
    # == New split
    bct.split_cluster(1)
    assert bct.n_leaf_nodes_ == 5
    assert bct.n_split_nodes_ == 4
    _check_node(bct.root_node_, False, 0, [0, 1, 2, 3, 4], bct)
    _check_node(bct.root_node_.left_node_, False, 1, [0, 2, 3], bct)
    _check_node(bct.root_node_.left_node_.left_node_, False, 3, [0, 3], bct)
    _check_node(bct.root_node_.left_node_.left_node_.left_node_, True, 5, [0], bct)
    _check_node(bct.root_node_.left_node_.left_node_.right_node_, True, 6, [3], bct)
    _check_node(bct.root_node_.left_node_.right_node_, True, 4, [2], bct)
    _check_node(bct.root_node_.right_node_, False, 2, [1, 4], bct)
    _check_node(bct.root_node_.right_node_.left_node_, True, 7, [1], bct)
    _check_node(bct.root_node_.right_node_.right_node_, True, 8, [4], bct)
    assert bct.root_node_.right_node_.left_node_.get_sibling() == bct.root_node_.right_node_.right_node_
    assert bct.root_node_.right_node_.right_node_.get_sibling() == bct.root_node_.right_node_.left_node_

    # == New split
    bct.split_cluster(3)
    _check_node(bct.root_node_, False, 0, [0, 1, 2, 3, 4, 5], bct)
    _check_node(bct.root_node_.left_node_, False, 1, [0, 2, 3, 5], bct)
    _check_node(bct.root_node_.left_node_.left_node_, False, 3, [0, 3, 5], bct)
    _check_node(bct.root_node_.left_node_.left_node_.left_node_, True, 5, [0], bct)
    _check_node(bct.root_node_.left_node_.left_node_.right_node_, False, 6, [3, 5], bct)
    _check_node(bct.root_node_.left_node_.right_node_, True, 4, [2], bct)
    _check_node(bct.root_node_.right_node_, False, 2, [1, 4], bct)
    _check_node(bct.root_node_.right_node_.left_node_, True, 7, [1], bct)
    _check_node(bct.root_node_.right_node_.right_node_, True, 8, [4], bct)
    _check_node(bct.root_node_.left_node_.left_node_.right_node_.left_node_, True, 9, [3], bct)
    _check_node(bct.root_node_.left_node_.left_node_.right_node_.right_node_, True, 10, [5], bct)
    assert bct.root_node_.left_node_.left_node_.right_node_.left_node_.get_sibling() == bct.root_node_.left_node_.left_node_.right_node_.right_node_
    assert bct.root_node_.left_node_.left_node_.right_node_.right_node_.get_sibling() == bct.root_node_.left_node_.left_node_.right_node_.left_node_
    # Check leafs
    leaf_nodes, split_nodes = bct.get_leaf_and_split_nodes()
    assert len(leaf_nodes) == 6
    assert len(split_nodes) == 5
    assert [l.labels for l in leaf_nodes] == [[2], [1], [4], [0], [3], [5]]
    # Test to str
    to_str = str(bct)
    assert to_str == "[[[0, [3, 5]], 2], [1, 4]]"
    # Check prune
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    new_labels = bct.prune_to_n_leaf_nodes(3, labels)
    assert np.array_equal(new_labels, np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0]))
    assert bct.n_leaf_nodes_ == 3
    assert bct.n_split_nodes_ == 2
    _check_node(bct.root_node_, False, 0, [0, 1, 2, 3, 4, 5], bct)
    _check_node(bct.root_node_.left_node_, False, 1, [0, 2, 3, 5], bct)
    _check_node(bct.root_node_.right_node_, True, 2, [1, 4], bct)
    assert bct.root_node_.left_node_.get_sibling() == bct.root_node_.right_node_
    assert bct.root_node_.right_node_.get_sibling() == bct.root_node_.left_node_
    _check_node(bct.root_node_.left_node_.left_node_, True, 3, [0, 3, 5], bct)
    _check_node(bct.root_node_.left_node_.right_node_, True, 4, [2], bct)
    assert bct.root_node_.left_node_.left_node_.get_sibling() == bct.root_node_.left_node_.right_node_
    assert bct.root_node_.left_node_.right_node_.get_sibling() == bct.root_node_.left_node_.left_node_
    # Check leafs
    leaf_nodes, split_nodes = bct.get_leaf_and_split_nodes()
    assert len(leaf_nodes) == 3
    assert len(split_nodes) == 2
    assert [l.labels for l in leaf_nodes] == [[1, 4], [0, 3, 5], [2]]
    # Test to str
    to_str = str(bct)
    assert to_str == "[[(0, 3, 5), 2], (1, 4)]"


def test_delete_node():
    bct = BinaryClusterTree()
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(1)
    bct.split_cluster(3)
    assert str(bct) == "[[[0, [3, 5]], 2], [1, 4]]"
    # Delete leaf node
    left_node = bct.root_node_.right_node_.left_node_
    right_node = bct.root_node_.right_node_.right_node_
    assert left_node.labels == [1]
    assert right_node.labels == [4]
    sibling = left_node.delete_node()
    assert sibling is right_node
    assert bct.root_node_.right_node_ is sibling
    assert bct.root_node_.right_node_.parent_node is bct.root_node_
    assert str(bct) == "[[[0, [3, 5]], 2], 4]"
    # Delete split node
    left_node = bct.root_node_.left_node_.left_node_
    right_node = bct.root_node_.left_node_.right_node_
    assert left_node.labels == [0, 3, 5]
    assert right_node.labels == [2]
    sibling = left_node.delete_node()
    assert sibling is right_node
    assert bct.root_node_.left_node_ is sibling
    assert bct.root_node_.left_node_.parent_node is bct.root_node_
    assert str(bct) == "[2, 4]"


def test_get_least_common_ancestor():
    bct = BinaryClusterTree()
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(0)
    bct.split_cluster(1)
    bct.split_cluster(3)
    ancestors = bct.get_least_common_ancestor(5, 2).labels
    assert np.array_equal(ancestors, [0, 2, 3, 5])
    ancestors = bct.get_least_common_ancestor(1, 4).labels
    assert np.array_equal(ancestors, [1, 4])
    ancestors = bct.get_least_common_ancestor(1, 1).labels
    assert np.array_equal(ancestors, [1])

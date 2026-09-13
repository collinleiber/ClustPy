from __future__ import annotations
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections.abc import Callable


class _ClusterTreeNode:
    """
    A node in a BinaryClusterTree.
    Each node contains a set of labels and can have a left and a right child node.
    If a node has children it is called split node, else it is referred to as leaf node.

    Parameters
    ----------
    labels : list
        the initial labels assigned to this node
    tree : BinaryClusterTree
        the tree that contains this node
    parent_node : _ClusterTreeNode | None
        the parent of this node

    Attributes
    ----------
    labels : list
        a list containing all cluster labels that are below this node
    tree : BinaryClusterTree
        the tree that contains this node
    parent_node : _ClusterTreeNode | None
        the parent node. Is None in the case of the root node
    left_node_ : _ClusterTreeNode | None
        the left child node. Can be None if node is a leaf node
    right_node_ : _ClusterTreeNode | None
        the right child node. Can be None if node is a leaf node
    node_id_ : int
        the ID of this node
    """

    def __init__(
        self, labels: list, tree: BinaryClusterTree, parent_node: _ClusterTreeNode | None
    ):
        self.labels = labels
        self.tree = tree
        self.parent_node = parent_node
        self.node_id_ = tree.get_current_node_id_counter()
        self.tree.n_leaf_nodes_ += 1
        self.left_node_ : _ClusterTreeNode | None = None
        self.right_node_ : _ClusterTreeNode | None = None

    def is_leaf_node(self) -> bool:
        """
        Check if node is a leaf node, i.e. it does not contain child nodes.

        Returns
        -------
        is_leaf_node : bool
            boolean indicating if node is a leaf node
        """
        assert (self.left_node_ is None and self.right_node_ is None) or (
            self.left_node_ is not None and self.right_node_ is not None
        )
        is_leaf_node = self.left_node_ is None
        return is_leaf_node

    def split_cluster(
        self,
        split_cluster_id: int,
        new_cluster_id: int,
        cluster_tree_node_class: type[_ClusterTreeNode],
    ) -> tuple[_ClusterTreeNode, _ClusterTreeNode]:
        """
        Split this node.
        Checks if the reference cluster is contained on the left or right side.
        If the specific side already references to another node, this node will be recursively called.
        Else, two new nodes will be added as children by splitting the reference cluster.

        Parameters
        ----------
        split_cluster_id : int
            the cluster ID of the cluster that should be split
        new_cluster_id : int
            the new cluster ID that should be added to the tree
        cluster_tree_node_class : type[_ClusterTreeNode]
            the class used to create the new cluster tree nodes (default: _ClusterTreeNode)

        Returns
        -------
        tuple : tuple[_ClusterTreeNode, _ClusterTreeNode]
            The two newly created nodes
        """
        assert (
            split_cluster_id in self.labels
        ), "split_cluster_id ({0}) is not contained in this node. Following labels are contained: {1}".format(
            split_cluster_id, self.labels
        )
        if self.is_leaf_node():
            self.left_node_ = cluster_tree_node_class(
                [split_cluster_id], self.tree, self
            )
            self.right_node_ = cluster_tree_node_class(
                [new_cluster_id], self.tree, self
            )
            self.tree.n_split_nodes_ += 1
            self.tree.n_leaf_nodes_ -= 1  # This node switches from split to leaf node
            to_return = (self.left_node_, self.right_node_)
        else:
            left_node = self.left_node_
            right_node = self.right_node_
            assert left_node is not None and right_node is not None
            if split_cluster_id in left_node.labels:
                to_return = left_node.split_cluster(
                    split_cluster_id, new_cluster_id, cluster_tree_node_class
                )
            else:
                to_return = right_node.split_cluster(
                    split_cluster_id, new_cluster_id, cluster_tree_node_class
                )
        self.labels.append(new_cluster_id)
        return to_return

    def delete_node(self) -> _ClusterTreeNode | None:
        """
        Delete this node from the cluster tree. Also deletes all children of this node and the parent.
        The sibling will get the position of the former parent.
        This method also adjusts the labels of all nodes upward in the tree.

        Returns
        -------
        sibling : _ClusterTreeNode | None
            The sibling node that substitutes the former parent node. Can be None if node has already been deleted
        """
        # Check if node was already (recursively) removed from tree or is root
        if self.parent_node is None:
            return None
        # Delete all child nodes by setting parent to None
        if not self.is_leaf_node():
            self.tree.n_split_nodes_ -= 1
            nodes_to_check = [self.left_node_, self.right_node_]
            i = 0
            while i < len(nodes_to_check):
                node = nodes_to_check[i]
                assert node is not None, "Node is None"
                # Set parent to None -> marks deletion
                node.parent_node = None
                if not node.is_leaf_node():
                    self.tree.n_split_nodes_ -= 1
                    nodes_to_check.append(node.left_node_)
                    nodes_to_check.append(node.right_node_)
                else:
                    self.tree.n_leaf_nodes_ -= 1
                i += 1
        else:
            self.tree.n_leaf_nodes_ -= 1
        # Delete labels from all parents
        parent_node_to_check = self.parent_node
        while True:
            for l in self.labels:
                if l in parent_node_to_check.labels:
                    parent_node_to_check.labels.remove(l)
            if parent_node_to_check.parent_node is None:
                break
            parent_node_to_check = parent_node_to_check.parent_node
        # Update parent by replacing with sibling
        sibling = self.get_sibling()
        if self.parent_node is self.tree.root_node_:
            # Create new root
            sibling.parent_node = None
            self.tree.root_node_ = sibling
        else:
            # Change parent of sibling and child of parents parent
            new_parent = self.parent_node.parent_node
            assert new_parent is not None, "Parent of parent is None"
            sibling.parent_node = new_parent
            if new_parent.left_node_ is self.parent_node:
                new_parent.left_node_ = sibling
            else:
                new_parent.right_node_ = sibling
            # Set parent to None -> marks deletion
            self.parent_node.parent_node = None
        self.tree.n_split_nodes_ -= 1  # The parent
        # Finally delete this node by setting parent to None
        self.parent_node = None
        return sibling

    def get_sibling(self) -> _ClusterTreeNode:
        """
        Get the sibling of this node.

        Returns
        -------
        sibling : _ClusterTreeNode
            The sibling node
        """
        assert self.parent_node is not None, "This node has no parent and thus no sibling"
        if self == self.parent_node.left_node_:
            sibling = self.parent_node.right_node_
        else:
            sibling = self.parent_node.left_node_
        assert sibling is not None, "This node has no sibling"
        return sibling

    def __str__(self) -> str:
        """
        Return this node as str.

        Returns
        -------
        to_str : str
            The string
        """
        if self.is_leaf_node():
            if self is self.tree.root_node_:
                return str(self.labels)
            else:
                return (
                    str(self.labels[0])
                    if len(self.labels) == 1
                    else "(" + str(self.labels).replace("[", "").replace("]", "") + ")"
                )
        else:
            to_str = "[{0}, {1}]".format(self.left_node_, self.right_node_)
        return to_str


class BinaryClusterTree:
    """
    A Binary Cluster Tree that saves a cluster hierarchy.
    In the beginning it only contains a root node.
    Each node in the tree contains a set of labels and can have a left and a right child node.
    If a node has children it is called split node, else it is referred to as leaf node.

    Parameters
    ----------
    cluster_tree_node_class : type[_ClusterTreeNode]
        the class used to create new cluster tree nodes (default: _ClusterTreeNode)

    Attributes
    ----------
    root_node_ : _ClusterTreeNode
        the root node of the cluster tree. Contains the label 0
    node_id_counter_ : int
        the current id for creating new nodes
    n_leaf_nodes_ : int
        the number of leaf nodes contained in the tree
    n_split_nodes_ : int
        the number of split nodes contained in the tree
    """

    def __init__(self, cluster_tree_node_class: type[_ClusterTreeNode] = _ClusterTreeNode):
        self.node_id_counter_ = 0
        self.n_leaf_nodes_ = 0
        self.n_split_nodes_ = 0
        self.cluster_tree_node_class = cluster_tree_node_class
        self.root_node_ = cluster_tree_node_class([0], self, None)

    def get_current_node_id_counter(self) -> int:
        """
        Cet current ID counter for new nodes in the tree and raise the counter by one.

        Returns
        -------
        current_counter : int
            The current counter
        """
        current_counter = self.node_id_counter_
        self.node_id_counter_ += 1
        return current_counter

    def split_cluster(
        self, split_cluster_id: int, new_cluster_id: int | None = None
    ) -> tuple[_ClusterTreeNode, _ClusterTreeNode]:
        """
        Split a specific cluster in the tree by creating two new nodes; one containing the split_cluster_id label and one with new the new_cluster_id label.

        Parameters
        ----------
        split_cluster_id : int
            The cluster id to split
        new_cluster_id : int | None
            the new cluster ID that should be added to the tree. If None, it will be specified automatically (default: None)

        Returns
        -------
        tuple : tuple[_ClusterTreeNode, _ClusterTreeNode]
            The two newly created cluster tree nodes
        """
        assert (
            split_cluster_id in self.root_node_.labels
        ), "split_cluster_id ({0}) is not contained in the tree. Following labels are contained: {1}".format(
            split_cluster_id, self.root_node_.labels
        )
        assert (
            new_cluster_id not in self.root_node_.labels
        ), "new_cluster_id ({0}) is already contained in the tree. Following labels are contained: {1}".format(
            new_cluster_id, self.root_node_.labels
        )
        new_cluster_id = (
            len(self.root_node_.labels) if new_cluster_id is None else new_cluster_id
        )
        new_left_node, new_right_node = self.root_node_.split_cluster(
            split_cluster_id, new_cluster_id, self.cluster_tree_node_class
        )
        return new_left_node, new_right_node

    def prune_to_n_leaf_nodes(
        self, n_leaf_nodes_to_keep: int, labels: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Prune the tree by only keeping the first n_leaf_nodes_to_keep leaf nodes in the tree.
        If labels are specified, they will be adjusted to the new structure.

        Parameters
        ----------
        n_leaf_nodes_to_keep : int
            The number of leaf nodes to keep in the cluster tree
        labels : np.ndarray | None
            an optional labels array that should be adjusted by considering the new structure (default: None)

        Returns
        -------
        labels : np.ndarray
            The adjusted labels array (if specified, else None)
        """
        assert n_leaf_nodes_to_keep > 0, "n_nodes_to_keep must be larger than 0"
        if labels is not None:
            labels = labels.copy()
        leaf_nodes, split_nodes = self.get_leaf_and_split_nodes()
        n_total_nodes_to_keep = n_leaf_nodes_to_keep * 2 - 1
        highest_node_id_to_keep = np.unique(
            [node.node_id_ for node in leaf_nodes + split_nodes]
        )[n_total_nodes_to_keep - 1]
        nodes_to_check = [self.root_node_]
        i = 0
        while i < len(nodes_to_check):
            node = nodes_to_check[i]
            if not node.is_leaf_node():
                left_node = node.left_node_
                right_node = node.right_node_
                assert left_node is not None and right_node is not None
                if (
                    left_node.node_id_ <= highest_node_id_to_keep
                    or right_node.node_id_ <= highest_node_id_to_keep
                ) and len(nodes_to_check) < n_total_nodes_to_keep:
                    nodes_to_check.append(left_node)
                    nodes_to_check.append(right_node)
                else:
                    left_node.parent_node = None
                    node.left_node_ = None
                    right_node.parent_node = None
                    node.right_node_ = None
                    min_label = np.min(node.labels)
                    if labels is not None:
                        labels[np.isin(labels, node.labels)] = min_label
            i += 1
        self.n_leaf_nodes_ = n_leaf_nodes_to_keep
        self.n_split_nodes_ = n_leaf_nodes_to_keep - 1
        LE = LabelEncoder()
        labels_pruned = LE.fit_transform(labels)
        return labels_pruned

    def get_least_common_ancestor(
        self, label_1: int, label_2: int
    ) -> _ClusterTreeNode:
        """
        Get the first node that contains label_1 and label_2.

        Parameters
        ----------
        label_1 : int
            the first label
        label_2 : int
            the second label

        Returns
        -------
        least_common_ancestor : _ClusterTreeNode
            The first ancestor node
        """
        assert (
            label_1 in self.root_node_.labels
        ), "label {0} is not contained in the tree".format(label_1)
        assert (
            label_2 in self.root_node_.labels
        ), "label {0} is not contained in the tree".format(label_2)
        least_common_ancestor = self.root_node_
        # Check if child still contains both labels
        while True:
            if least_common_ancestor.is_leaf_node():
                break
            left_node = least_common_ancestor.left_node_
            right_node = least_common_ancestor.right_node_
            assert left_node is not None and right_node is not None
            all_in_left = (
                label_1 in left_node.labels
                and label_2 in left_node.labels
            )
            # Both labels are contained in the left part of the tree
            if all_in_left:
                least_common_ancestor = left_node
            else:
                all_in_right = (
                    label_1 in right_node.labels
                    and label_2 in right_node.labels
                )
                # Both labels are contained in the right part of the tree
                if all_in_right:
                    least_common_ancestor = right_node
                else:
                    break
        return least_common_ancestor

    def get_leaf_and_split_nodes(self) -> tuple[list, list]:
        """
        Get all leaf and split nodes of the tree.

        Returns
        -------
        tuple : tuple[list, list]
            A list containing all leaf nodes,
            A list containing all split nodes
        """
        leaf_nodes = []
        split_nodes = []
        nodes_to_check = [self.root_node_]
        i = 0
        while i < len(nodes_to_check):
            node = nodes_to_check[i]
            if node.is_leaf_node():
                leaf_nodes.append(node)
            else:
                assert node.left_node_ is not None and node.right_node_ is not None
                split_nodes.append(node)
                nodes_to_check.append(node.left_node_)
                nodes_to_check.append(node.right_node_)
            i += 1
        return leaf_nodes, split_nodes

    def __str__(self) -> str:
        """
        Return this tree as str

        Returns
        -------
        to_str : str
            The string
        """
        return str(self.root_node_)

    def export_sklearn_dendrogram(self, distance_function: Callable | None = None) -> np.ndarray:
        """
        Export the tree in the format expected by ``sklearn``/``scipy`` dendrograms
        (linkage matrix).  Each row corresponds to a non-leaf node and contains:

        [child1_id, child2_id, distance, num_nodes_in_subtree]

        * ``child*_id``: the id of the left/right child (based on the labels for all leaves).
        * ``distance``: by default uniformly 0.0, can be overriden by providing a distance function on tree nodes.
        * ``num_nodes_in_subtree``: number of original leaf nodes beneath the
          node.

        Parameters
        ----------
        distance_function : Callable | None
            a specific distance function to use for the thrid column in the dendogram (default: None)

        Returns
        -------
        np.ndarray
            A ``(n_split_nodes, 4)`` linkage matrix.
        """
        rows = list()
        leaves, split_nodes = self.get_leaf_and_split_nodes()
        node_numbers = {leaf: (leaf.labels[0], 1) for leaf in leaves}

        if distance_function is None:
            distance_function = lambda x, y: 0.0

        while split_nodes:
            node = split_nodes.pop(0)
            if (
                node.left_node_ in node_numbers.keys()
                and node.right_node_ in node_numbers.keys()
            ):
                # both children have already been processed, so we can proceed
                rows.append(
                    (
                        node_numbers[node.left_node_][0],
                        node_numbers[node.right_node_][0],
                        distance_function(node.left_node_, node.right_node_),
                        node_numbers[node.left_node_][1]
                        + node_numbers[node.right_node_][1],
                    )
                )
                node_numbers[node] = (len(node_numbers), rows[-1][3])
            else:
                # we need to wait until all children are processed
                split_nodes.append(node)
        return np.array(rows)

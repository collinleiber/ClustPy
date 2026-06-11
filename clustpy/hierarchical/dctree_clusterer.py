"""
@authors:
Pascal Weber
"""

# - Author: Pascal Weber
# - Source: https://github.com/pasiweber/SHADE

from __future__ import annotations
import numpy as np
import sys
from clustpy.utils.dctree import DCTree, _DCNode
from typing import Optional
from sklearn.base import ClusterMixin, BaseEstimator
from clustpy.utils.checks import check_parameters


sys.setrecursionlimit(1000000000)


class DCTree_Clusterer(ClusterMixin, BaseEstimator):
    """
    The DCTree clustering algorithm.
    Identifies stable nodes within the DCTree and labels the data accordingly.

    Parameters
    ----------
    min_points : int
        the minimum number of points (default: 5)
    use_less_memory: bool
      Use less memory when constructing the DCTree.
      This will, however, increase the runtime (default: False)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    dc_tree_ : BinaryClusterTree
        The resulting cluster tree
    n_features_in_ : int
        the number of features used for the fitting

    References
    ----------
    SHADE: Deep Density-based Clustering
    Anna Beer; Pascal Weber; Lukas Miklautz; Collin Leiber; Walid Durani; Christian Böhm
    IEEE International Conference on Data Mining (ICDM), Abu Dhabi, United Arab Emirates, 2024, pp. 675-680, doi: 10.1109/ICDM59182.2024.
    """

    def __init__(self, min_points: int = 5, use_less_memory: bool = False):
        self.min_points = min_points
        self.use_less_memory = use_less_memory

    def fit(self, X: np.ndarray, y: np.ndarray=None) -> 'DCTree_Clusterer':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DCTree_Clusterer
            this instance of the DCTree_Clusterer algorithm
        """
        X, _, _ = check_parameters(X=X, y=y)
        self.dc_tree_ = DCTree(X, min_points=self.min_points, use_less_memory=self.use_less_memory)
        condensed_root = self._condense(self.dc_tree_.root)
        stable_nodes = self._get_stable_nodes(condensed_root)
        labels = np.full(self.dc_tree_.n, -1 if stable_nodes else 0, dtype=np.int32)
        self.n_clusters_ = len(stable_nodes) if stable_nodes else 1
        for idx, node in enumerate(stable_nodes):
            labels[node.leaves] = idx
        self.labels_ = labels
        self.n_features_in_ = X.shape[1]
        return self

    def _condense(self, node: Optional[_DCNode]) -> Optional[_DCNode]:
        """
        Condense the tree to nodes, where both children contain at least min_points leaves.
        Uses a recursive strategy that checks each node separately.

        Parameters
        ----------
        node : _DCNode
            the node that is checked for its children

        Returns
        -------
        condensed_node : _DCNode
            either a condensed node or None
        """
        if node is None or len(node.leaves) < self.min_points:
            return None
        # Process children
        L = self._condense(node.left)
        R = self._condense(node.right)
        # If both children have min_points children, keep this branch.
        size_L = len(node.left.leaves) if node.left is not None else 0
        size_R = len(node.right.leaves) if node.right is not None else 0
        if size_L >= self.min_points and size_R >= self.min_points:
            if (L is not None and R is not None) or (L is None and R is None):
                return _DCNode(node.id, node.dist, node.leaves, L, R)
            elif L is not None:
                return L
            else:
                return R
        elif L is not None:
            return _DCNode(node.id, L.dist, node.leaves, L.left, L.right)
        elif R is not None:
            return _DCNode(node.id, R.dist, node.leaves, R.left, R.right)
        return None

    def _get_stable_nodes(self, node: Optional[_DCNode], parent_dist: float = None) -> list:
        """
        Identify stable nodes in the tree.

        Parameters
        ----------
        node : _DCNode
            the node that is checked.
        parent_dist : float
            Distance in the parent node. Can be None in the case of the root (default: None)

        Returns
        -------
        stable_nodes : list
            list of stable nodes
        """
        if node is None: # In case root is None
            return []
        node.stability_ = (1.0/node.dist - 1.0/parent_dist) * len(node.leaves) if parent_dist is not None else 0
        # Calculate stability for children
        sum_child_stabilities = 0
        child_results = []
        for child in [node.left, node.right]:
            if child is not None:
                child_results += self._get_stable_nodes(child, node.dist)
                sum_child_stabilities += child.stability_
        # Flag stable nodes
        if node.stability_ >= sum_child_stabilities:
            stable_nodes = [node]
        else:
            node.stability_ = sum_child_stabilities
            stable_nodes = child_results
        return stable_nodes

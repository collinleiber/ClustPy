"""
@authors:
Pascal Weber
"""

# - Author: Pascal Weber
# - Source: https://github.com/pasiweber/SHADE

from __future__ import annotations

import numpy as np

from collections import deque
from .dctree import DCTree, _DCNode as _DCDist_DCNode
from typing import List, Literal, Optional, Sequence, Tuple, Union

from sklearn.base import ClusterMixin, BaseEstimator


class DCTree_Clusterer(ClusterMixin, BaseEstimator):
    """
    Parameters
    ----------
    points : np.ndarray
        points, of which the dc_distances should be computed of.

    min_points : int, optional
        min_points parameter used for the computation of the dc_distances, by default 5.
    """

    min_points: int
    increase_inter_cluster_distance: bool

    dc_tree: DCTree
    labels_: np.ndarray

    def __init__(
        self,
        min_points: int = 5,
        increase_inter_cluster_distance: bool = True,
    ):
        self.min_points = min_points
        self.increase_inter_cluster_distance = increase_inter_cluster_distance

    def fit(self, X, y=None):
        self.dc_tree = DCTree(X, min_points=self.min_points)
        self.labels_ = self.clustering(self.dc_tree)
        self.n_clusters = len(np.where(np.unique(self.labels_) >= 0)[0])

        if self.increase_inter_cluster_distance:
            self._increase_inter_cluster_distance(self.dc_tree.root)

        return self

    def _condense_tree(self, node: Optional[_DCDist_DCNode]) -> Optional[_DCNode]:
        if node is None:
            return None

        if node.left is None:
            right = self._condense_tree(node.right)
            if right is not None:
                right.id = node.id
                right.leaves = node.leaves
            return right

        if node.right is None:
            left = self._condense_tree(node.left)
            if left is not None:
                left.id = node.id
                left.leaves = node.leaves
            return left

        left_size = len(node.left.leaves)
        right_size = len(node.right.leaves)

        if right_size >= self.min_points and left_size >= self.min_points:
            node_left = self._condense_tree(node.left)
            node_right = self._condense_tree(node.right)
            if node_left is None and node_right is None:
                return _DCNode(node.id, node.dist, node.leaves, node_left, node_right)
            if node_left is None:
                return node_right
            if node_right is None:
                return node_left
            return _DCNode(node.id, node.dist, node.leaves, node_left, node_right)

        if right_size < self.min_points and left_size < self.min_points:
            return None

        if right_size < self.min_points:
            left = self._condense_tree(node.left)
            if left is not None:
                left.id = node.id
                left.leaves = node.leaves
            return left

        if left_size < self.min_points:
            right = self._condense_tree(node.right)
            if right is not None:
                right.id = node.id
                right.leaves = node.leaves
            return right

    def _calculate_stability(self, node: _DCNode):
        if node.left is None and node.right is None:
            return

        if node.left is not None:
            self._calculate_stability(node.left)
            node.left.stability = (1.0 / node.left.dist - 1.0 / node.dist) * len(node.left.leaves)

        if node.right is not None:
            self._calculate_stability(node.right)
            node.right.stability = (1.0 / node.right.dist - 1.0 / node.dist) * len(
                node.right.leaves
            )

    def _stable_node_flagging(self, node: Optional[_DCNode]):
        if node is None:
            return

        node.is_stable = False

        if node.left is None and node.right is None:
            node.is_stable = True
            return

        if node.left is None:
            self._stable_node_flagging(node.right)
            if node.stability < node.right.stability:
                node.stability = node.right.stability
                node.is_stable = False
                return
            else:
                node.is_stable = True
                return

        if node.right is None:
            self._stable_node_flagging(node.left)
            if node.stability < node.left.stability:
                node.stability = node.left.stability
                node.is_stable = False
                return
            else:
                node.is_stable = True
                return

        self._stable_node_flagging(node.right)
        self._stable_node_flagging(node.left)

        if node.stability < node.left.stability + node.right.stability:
            node.stability = node.left.stability + node.right.stability
            node.is_stable = False
            return
        else:
            # Node is a true cluster
            node.is_stable = True
            return

    def _get_stable_clusters(self, node: Optional[_DCNode]) -> list[_DCNode]:
        if node is None:
            return []

        if node.is_stable:
            self.dc_tree[node.id].is_stable = True
            return [node]

        else:
            return self._get_stable_clusters(node.left) + self._get_stable_clusters(node.right)

    def _increase_inter_cluster_distance(self, root: _DCDist_DCNode):
        queue = deque()
        queue.append(root)
        while len(queue):
            node = queue.popleft()

            if node.left is None and node.right is None:
                continue

            if hasattr(node, "is_stable") and node.is_stable:
                continue

            node.dist += max(
                node.left.dist * len(node.left.leaves), node.right.dist * len(node.right.leaves)
            )
            # node.dist += max(
            #     (node.dist - node.left.dist) * len(node.left.leaves), (node.dist - node.right.dist) * len(node.right.leaves)
            # )
            queue.append(node.left)
            queue.append(node.right)

    def stable_nodes(self, dc_tree: DCTree) -> list[_DCNode]:
        new_root = self._condense_tree(dc_tree.root)
        if new_root is None:
            return []

        self._calculate_stability(new_root)
        new_root.stability = 0

        self._stable_node_flagging(new_root)
        new_root.is_stable = False

        stable_nodes = self._get_stable_clusters(new_root)

        stable_nodes_ = []
        for node in stable_nodes:
            if len(node.leaves) >= self.min_points:
                stable_nodes_.append(node)

        return stable_nodes_

    def clustering(self, dc_tree: DCTree):
        stable_nodes = self.stable_nodes(dc_tree)

        labels = np.full(dc_tree.n, -1)

        idx = 0
        for i in range(len(stable_nodes)):
            labels[stable_nodes[i].leaves] = idx
            idx += 1
        return labels


class _DCNode(_DCDist_DCNode):
    is_stable: Optional[Union[Literal[True], Literal[False]]]
    stability: Optional[float]
    left: _DCNode
    right: _DCNode

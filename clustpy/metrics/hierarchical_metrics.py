from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import clustpy.metrics
import numpy as np
import torch
from scipy.special import comb
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             normalized_mutual_info_score)


class PredictionClusterNode:
    """
    Represents a node in the prediction cluster tree.

    Attributes
    ----------
    id : int
        The ID of the node.
    split_id : int
        The split ID of the node.
    center : np.ndarray
        The center of the cluster represented by the node.
    parent : PredictionClusterNode, optional
        The parent node.
    left_child : PredictionClusterNode, optional
        The left child node.
    right_child : PredictionClusterNode, optional
        The right child node.
    assigned_indices : List[int]
        The indices of data points assigned to this node.
    """

    def __init__(
        self,
        id: int,
        split_id: int,
        center: np.ndarray,
        parent: "PredictionClusterNode" = None,
        left_child: "PredictionClusterNode" = None,
        right_child: "PredictionClusterNode" = None,
    ) -> "PredictionClusterNode":
        self.id = id
        self.split_id = split_id
        self.parent: PredictionClusterNode = parent
        self.left_child: PredictionClusterNode = left_child
        self.right_child: PredictionClusterNode = right_child
        self.assigned_indices: List[int] = []
        self.center = center

    def assign_batch(
        self,
        dataset_indices: torch.Tensor,
        assigned_batch_indices: Union[torch.Tensor, None],
    ):
        """
        Assigns a batch of data points to this node.

        Parameters
        ----------
        dataset_indices : torch.Tensor
            Indices of the entire dataset.
        assigned_batch_indices : Union[torch.Tensor, None]
            Indices of the batch assigned to this node.
        """
        if assigned_batch_indices is not None:
            self.assigned_indices.extend(
                dataset_indices[assigned_batch_indices].tolist()
            )

    @property
    def assignments(self):
        """
        Returns the sorted list of assigned indices.

        Returns
        -------
        List[int]
            The sorted list of assigned indices.
        """
        return sorted(self.assigned_indices)

    @property
    def is_leaf(self):
        """
        Checks if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return self.left_child is None and self.right_child is None


def count_values_in_sequence(seq: np.ndarray):
    """
    Counts the occurrences of each value in the sequence.

    Parameters
    ----------
    seq : np.ndarray
        The input sequence.

    Returns
    -------
    dict
        A dictionary with the count of each value in the sequence.
    """
    res = defaultdict(int)
    for key in seq:
        res[key] += 1
    return dict(res)


def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculates the accuracy between true and predicted labels.

    Parameters
    ----------
    true_labels : np.ndarray
        The true labels.
    predicted_labels : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The accuracy score.
    """
    return accuracy_score(true_labels, predicted_labels)


def weighted_avg_and_std(values, weights):
    """
    Calculates the weighted average and standard deviation.

    Parameters
    ----------
    values : np.ndarray
        The values to average.
    weights : np.ndarray
        The weights for the values.

    Returns
    -------
    tuple
        The weighted average and standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


class PredictionClusterTree:
    """
    Represents a prediction cluster tree.

    Attributes
    ----------
    root : PredictionClusterNode
        The root node of the tree.
    """

    def __init__(self, root_node: "PredictionClusterNode") -> None:
        self.root = root_node

    def __getitem__(self, id):
        """
        Gets a node by its ID.

        Parameters
        ----------
        id : int
            The ID of the node.

        Returns
        -------
        PredictionClusterNode
            The node with the given ID.

        Raises
        ------
        IndexError
            If the node with the given ID is not found.
        """
        def find_idx_recursive(node: PredictionClusterNode):
            if node.id == id:
                return node
            if not node.is_leaf:
                left = find_idx_recursive(node.left_child)
                if left is not None:
                    return left
                right = find_idx_recursive(node.right_child)
                if right is not None:
                    return right
            return None

        found_node = find_idx_recursive(self.root)
        if found_node is not None:
            return found_node
        raise IndexError(f"Node with id: {id} not found")

    @property
    def leaf_nodes(self) -> List[PredictionClusterNode]:
        """
        Gets the list of all leaf nodes in the tree.

        Returns
        -------
        List[PredictionClusterNode]
            The list of all leaf nodes.
        """
        def get_nodes_recursive(node: PredictionClusterNode):
            result = []
            if node.is_leaf:
                result.append(node)
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    @property
    def nodes(self):
        """
        Gets the list of all nodes in the tree.

        Returns
        -------
        List[PredictionClusterNode]
            The list of all nodes.
        """
        def get_nodes_recursive(node: PredictionClusterNode):
            result = [node]
            if node.is_leaf:
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    def __aggregate_assignments(self):
        """
        Aggregates the assignments from leaf nodes to inner nodes.
        """
        def aggregate_nodes_recursive(node: PredictionClusterNode):
            if node.is_leaf:
                return node.assigned_indices
            node.assigned_indices.clear()
            node.assigned_indices.extend(aggregate_nodes_recursive(node.left_child))
            node.assigned_indices.extend(aggregate_nodes_recursive(node.right_child))
            return node.assigned_indices

        aggregate_nodes_recursive(self.root)

    def get_k_clusters(self, k: int) -> List[PredictionClusterNode]:
        """
        Gets the k clusters from the tree.

        Parameters
        ----------
        k : int
            The number of clusters to retrieve.

        Returns
        -------
        List[PredictionClusterNode]
            The list of k cluster nodes.
        """
        self.__aggregate_assignments()
        result_nodes = []
        max_split_level = sorted(list(set([node.split_id for node in self.nodes])))[
            k - 1
        ]

        # the leaf nodes after the first <k> - 1 growing steps (splits) are the nodes representing the <k> clusters
        def get_nodes_at_split_level(node: PredictionClusterNode):
            if (
                node.is_leaf or node.left_child.split_id > max_split_level
            ) and node.split_id <= max_split_level:
                result_nodes.append(node)
                return
            get_nodes_at_split_level(node.left_child)
            get_nodes_at_split_level(node.right_child)

        get_nodes_at_split_level(self.root)
        # consistency check
        assert (
            len(result_nodes) == k
        ), "Number of cluster nodes doesn't correspond to number of classes"
        return result_nodes

    def get_k_cluster_predictions(self, ground_truth: np.ndarray, k: int):
        """
        Gets the predictions for k clusters.

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth labels.
        k : int
            The number of clusters.

        Returns
        -------
        np.ndarray
            The predicted labels for k clusters.
        """
        predictions = np.zeros_like(ground_truth, dtype=np.int32)
        for i, cluster in enumerate(self.get_k_clusters(k)):
            predictions[cluster.assignments] = i
        return predictions

    def dendrogram_purity(self, ground_truth: np.ndarray):
        """
        Calculates the dendrogram purity.

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth labels.

        Returns
        -------
        float
            The dendrogram purity.
        """
        return dendrogram_purity(self, ground_truth)

    def leaf_purity(self, ground_truth: np.ndarray):
        """
        Calculates the leaf purity.

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth labels.

        Returns
        -------
        tuple
            The weighted average and standard deviation of the leaf purity.
        """
        return leaf_purity(self, ground_truth)

    def flat_accuracy(self, ground_truth: np.ndarray, n_clusters: int):
        """
        Calculates the flat accuracy.

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth labels.
        n_clusters : int
            The number of clusters.

        Returns
        -------
        float
            The flat accuracy.
        """
        return clustpy.metrics.unsupervised_clustering_accuracy(
            ground_truth, self.get_k_cluster_predictions(ground_truth, n_clusters)
        )

    def flat_nmi(self, ground_truth: np.ndarray, n_clusters: int):
        """
        Calculates the flat normalized mutual information (NMI).

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth labels.
        n_clusters : int
            The number of clusters.

        Returns
        -------
        float
            The flat NMI.
        """
        return normalized_mutual_info_score(
            ground_truth, self.get_k_cluster_predictions(ground_truth, n_clusters)
        )

    def flat_ari(self, ground_truth: np.ndarray, n_clusters: int):
        """
        Calculates the flat adjusted rand index (ARI).

        Parameters
        ----------
        ground_truth : np.ndarray
            The ground truth labels.
        n_clusters : int
            The number of clusters.

        Returns
        -------
        float
            The flat ARI.
        """
        return adjusted_rand_score(
            ground_truth, self.get_k_cluster_predictions(ground_truth, n_clusters)
        )


def leaf_purity(tree: PredictionClusterTree, ground_truth: np.ndarray):
    """
    Calculates the leaf purity of the tree.

    Parameters
    ----------
    tree : PredictionClusterTree
        The prediction cluster tree.
    ground_truth : np.ndarray
        The ground truth labels.

    Returns
    -------
    tuple
        The weighted average and standard deviation of the leaf purity.
    """
    values = []
    weights = []

    def get_leaf_purities(node: PredictionClusterNode):
        nonlocal values
        nonlocal weights
        if node.is_leaf:
            node_total_dp_count = len(node.assignments)
            node_per_label_counts = count_values_in_sequence(
                [ground_truth[id] for id in node.assignments]
            )
            if node_total_dp_count > 0:
                purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
            else:
                purity_rate = 1.0
            values.append(purity_rate)
            weights.append(node_total_dp_count)
        else:
            get_leaf_purities(node.left_child)
            get_leaf_purities(node.right_child)

    get_leaf_purities(tree.root)

    return weighted_avg_and_std(values, weights)


def dendrogram_purity(tree: PredictionClusterTree, ground_truth: np.ndarray) -> float:
    """
    Calculates the dendrogram purity of the tree.

    Parameters
    ----------
    tree : PredictionClusterTree
        The prediction cluster tree.
    ground_truth : np.ndarray
        The ground truth labels.

    Returns
    -------
    float
        The dendrogram purity.
    """
    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {
        k: comb(v, 2, True) for k, v in total_per_label_frequencies.items()
    }
    total_n_of_pairs = sum(total_per_label_pairs_count.values())

    one_div_total_n_of_pairs = 1.0 / total_n_of_pairs

    purity = 0.0

    def calculate_purity(node: PredictionClusterNode) -> Tuple[Dict[Any, int], int]:
        nonlocal purity
        if node.is_leaf:
            node_total_dp_count = len(node.assignments)
            node_per_label_frequencies = count_values_in_sequence(
                [ground_truth[id] for id in node.assignments]
            )
            node_per_label_pairs_count: Dict[Any, int] = {
                k: comb(v, 2, True) for k, v in node_per_label_frequencies.items()
            }

        else:  # it is an inner node
            left_child_per_label_freq, left_child_total_dp_count = calculate_purity(
                node.left_child
            )
            right_child_per_label_freq, right_child_total_dp_count = calculate_purity(
                node.right_child
            )
            node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
            node_per_label_frequencies: Dict[Any, int] = {
                k: left_child_per_label_freq.get(k, 0)
                + right_child_per_label_freq.get(k, 0)
                for k in set(left_child_per_label_freq)
                | set(right_child_per_label_freq)
            }

            node_per_label_pairs_count: Dict[Any, int] = {
                k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k)
                for k in set(left_child_per_label_freq)
                & set(right_child_per_label_freq)
            }

        for label, pair_count in node_per_label_pairs_count.items():
            label_freq = node_per_label_frequencies[label]
            label_pairs = pair_count
            purity += (
                one_div_total_n_of_pairs
                * label_freq
                / node_total_dp_count
                * label_pairs
            )
        return node_per_label_frequencies, node_total_dp_count

    calculate_purity(tree.root)
    return purity

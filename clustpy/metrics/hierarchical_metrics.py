from clustpy.hierarchical._cluster_tree import BinaryClusterTree
from clustpy.metrics import purity
import numpy as np


def leaf_purity(labels_true: np.ndarray, labels_pred: np.ndarray, tree: BinaryClusterTree) -> float:
    """
    Calculates the leaf purity of the tree.
    Uses labels fromm leafs in the tree to calculate the purity (see clustpy.metrics.purity).

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm
    tree : BinaryClusterTree
        The clustering tree

    Returns
    -------
    leaf_purity : float
        The leaf purity

    References
    -------
    Mautz, Dominik, Claudia Plant, and Christian Böhm. "Deepect: The deep embedded cluster tree."
    Data Science and Engineering 5 (2020): 419-432.
    """
    leaf_nodes, _ = tree.get_leaf_and_split_nodes()
    labels_pred_adj = -np.ones(labels_pred.shape[0])
    for i, leaf_node in enumerate(leaf_nodes):
        labels_pred_adj[np.isin(labels_pred, leaf_node.labels)] = i
    leaf_purity = purity(labels_true, labels_pred_adj)
    return leaf_purity


def dendrogram_purity(labels_true: np.ndarray, labels_pred: np.ndarray, tree: BinaryClusterTree) -> float:
    """
    Calculates the dendrogram purity of the tree.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm
    tree : BinaryClusterTree
        The clustering tree

    Returns
    -------
    dendrogram_purity : float
        The dendrogram purity

    References
    -------
    Heller, Katherine A., and Zoubin Ghahramani. "Bayesian hierarchical clustering."
    Proceedings of the 22nd international conference on Machine learning. 2005.

    or

    Kobren, Ari, et al. "A hierarchical algorithm for extreme clustering."
    Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017.
    """
    cluster_ids_true, cluster_sizes_true = np.unique(labels_true, return_counts=True)
    total_per_label_pairs_count = np.sum([cluster_size * (cluster_size - 1) / 2 for cluster_size in cluster_sizes_true])
    purity_sum = 0
    for id_true in cluster_ids_true:
        points_in_true_cluster = (labels_true == id_true)
        pred_labels_in_cluster, pred_labels_counts_in_cluster = np.unique(labels_pred[points_in_true_cluster],
                                                                          return_counts=True)
        for i, id_pred_1 in enumerate(pred_labels_in_cluster):
            for j in range(i, len(pred_labels_counts_in_cluster)):
                id_pred_2 = pred_labels_in_cluster[j]
                if i == j:
                    # Get all pairs with same cluster label
                    occurrences_of_pair = pred_labels_counts_in_cluster[i] * (pred_labels_counts_in_cluster[i] - 1) / 2
                else:
                    # Get all pairs with different cluster label
                    occurrences_of_pair = pred_labels_counts_in_cluster[i] * pred_labels_counts_in_cluster[j]
                ancestor_labels = tree.get_least_common_ancestor(id_pred_1, id_pred_2).labels
                contained_in_ancestor_labels = np.isin(labels_pred, ancestor_labels)
                intersection_size = np.sum(contained_in_ancestor_labels & points_in_true_cluster)
                purity_sum += occurrences_of_pair * (intersection_size / np.sum(contained_in_ancestor_labels))
    dendrogram_purity = purity_sum / total_per_label_pairs_count
    return dendrogram_purity

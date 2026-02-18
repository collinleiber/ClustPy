from clustpy.hierarchical._cluster_tree import BinaryClusterTree, _ClusterTreeNode
from clustpy.metrics.confusion_matrix import ConfusionMatrix
from clustpy.metrics._metrics_utils import _check_labels_arrays
import numpy as np


def node_purity(node: _ClusterTreeNode, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the purity of this node within a Cluster Tree.
    A leaf with no assigned points receives a purity score of 0.

    Parameters
    ----------
    node: _ClusterTreeNode
        The node of a clustering tree
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    node_purity : float
        The node purity
    """
    labels_true, labels_pred = _check_labels_arrays(labels_true, labels_pred)
    samples_in_leaf = np.isin(labels_pred, node.labels)
    if np.any(samples_in_leaf):
        sizes_gt_matches = np.unique(labels_true[samples_in_leaf], return_counts=True)[1]
        node_purity = sizes_gt_matches.max() / samples_in_leaf.sum()
    else:
        node_purity = 0.
    return node_purity


def leaf_purity(
    tree: BinaryClusterTree, labels_true: np.ndarray, labels_pred: np.ndarray
) -> float:
    """
    Calculates the leaf purity of the tree.
    The leaf purity is equal to a weighted average of the maximum class purity across all leaves.
    Uses labels from leafs in the tree to identify the most frequent ground truth class and weights the score by the size of the leaf.
    If each label contains a single label, this is equal to the standard purity metric.

    Parameters
    ----------
    tree : BinaryClusterTree
        The clustering tree
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    leaf_purity : float
        The leaf purity

    References
    -------
    Mautz, Dominik, Claudia Plant, and Christian Böhm. "Deepect: The deep embedded cluster tree."
    Data Science and Engineering 5 (2020): 419-432.
    """
    cm = ConfusionMatrix(labels_true, labels_pred)
    leaf_nodes, _ = tree.get_leaf_and_split_nodes()
    leaf_purity = 0
    for leaf_node in leaf_nodes:
        relevant_columns = np.isin(cm.pred_clusters, leaf_node.labels)
        column_sum = cm.confusion_matrix[:, relevant_columns].sum(1)
        leaf_purity += column_sum.max()
    leaf_purity = leaf_purity / len(labels_true)
    return leaf_purity


def dendrogram_purity(
    dendrogram: np.ndarray | BinaryClusterTree, labels_true: np.ndarray, labels_pred: np.ndarray | None = None
) -> float:
    """
    Calculates the dendrogram purity of the tree.

    Parameters
    ----------
    dendrogram: np.ndarray | BinaryClusterTree
        BinaryClusterTree or sklearn/scipy-style dendrogram (e.g. AgglomerativeClustering -> children_)
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray | None
        The labels as predicted by a clustering algorithm.
        If None, we assume a full tree with each leaf node defining its own cluster containing a single point (default: None)

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
    if labels_pred is None:
        labels_pred = np.arange(labels_true.shape[0])
    labels_true, labels_pred = _check_labels_arrays(labels_true, labels_pred)
    if type(dendrogram) is BinaryClusterTree:
        # Transform ClusterTree to sklearn dendrogram
        dendrogram = dendrogram.export_sklearn_dendrogram()
    num_true_classes = labels_true.max() + 1
    num_base_clusters =  dendrogram.shape[0] + 1
    parent_matrix = _get_parent_matrix(dendrogram=dendrogram)
    # counts_per_class contains for every true class label the number of datapoints
    # with this label for each base class. For a full hierarchy this is either 0 or 1.
    # For an overclustering, larger values are possible.
    counts_per_class = ConfusionMatrix(labels_true, labels_pred, (num_true_classes, num_base_clusters)).confusion_matrix

    # parent_matrix[i,:] contains a 1 at position j if j is a descendant of i.
    # the matrix multiplication thus sums up how many original nodes per GT class are at each
    # node in the tree.
    node_counts = counts_per_class @ parent_matrix.T

    purity = 0.0
    total_pairs = 0

    for true_label_index in range(num_true_classes):
        # within-leaf purities (always 0 for complete hierarchies)
        for base_cluster_index in range(num_base_clusters):
            class_count = counts_per_class[true_label_index, base_cluster_index]
            sum_counts_per_class = np.sum(counts_per_class[:, base_cluster_index])
            if sum_counts_per_class > 0:
                leaf_purity = class_count / sum_counts_per_class
                weight = class_count * (class_count - 1) / 2
                purity += leaf_purity * weight
                total_pairs += weight
        # purity where inner nodes of the dendrogram serve as LCA
        for index, (child1, child2) in enumerate(dendrogram[:, :2]):
            counts_child1 = node_counts[true_label_index, int(child1)]
            counts_child2 = node_counts[true_label_index, int(child2)]
            weight = counts_child1 * counts_child2

            class_count = node_counts[true_label_index, index + num_base_clusters]
            sum_node_counts = np.sum(node_counts[:, index + num_base_clusters])
            if sum_node_counts > 0:
                node_purity = class_count / sum_node_counts

                purity += node_purity * weight
                total_pairs += weight

    purity /= total_pairs
    return purity


def _get_parent_matrix(dendrogram: np.ndarray) -> np.ndarray:
    """
    The parent matrix stores for each cluster which of the basic elements/clusters form it.

    Parameters
    ----------
    dendrogram: np.ndarray
        sklearn/scipy-style dendrogram

    Returns
    -------
    parent_matrix : np.ndarray
        The parent matrix
    """
    num_base_clusters = dendrogram.shape[0] + 1
    # each line of the dendrogram corresponds to a cluster, plus we need the initial clusters
    parent_matrix = np.zeros(
        (dendrogram.shape[0] + num_base_clusters, num_base_clusters), dtype=np.int8
    )
    # each initial cluster consists just of itself
    parent_matrix[:num_base_clusters] = np.eye(num_base_clusters)
    for i, (child1, child2) in enumerate(dendrogram[:, :2]):
        parent_matrix[i + num_base_clusters] = parent_matrix[int(child1)] + parent_matrix[int(child2)]
    assert np.max(parent_matrix) == 1, "input dendrogram does not correspond to a tree"
    assert np.all(parent_matrix[-1] == 1), "input dendrogram does not correspond to a tree"
    return parent_matrix

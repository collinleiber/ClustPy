"""
@authors:
Collin Leiber
"""

from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from scipy.spatial.distance import pdist, squareform
from clustpy.hierarchical._cluster_tree import BinaryClusterTree
import copy


def _diana(X: np.ndarray, n_clusters: int, distance_threshold: float, construct_full_tree: bool, metric: str) -> (
        np.ndarray, BinaryClusterTree):
    """
    Start the actual DIANA clustering procedure on the input data set.
    
    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        The number of clusters (can be None)
    distance_threshold : float
        The distance thresholds defines the minimum diameter that is considered (can be 0)
    construct_full_tree : bool
        Defines whether the full tree should be constructed after n_clusters has been reached
    metric : str
        Metric used to compute the dissimilarity. Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed" (see scipy.spatial.distance.pdist)

    Returns
    -------
    tuple : (np.ndarray, BinaryClusterTree)
        The final cluster labels,
        The resulting tree containing the cluster hierarchy
    """
    labels = np.zeros(X.shape[0], dtype=np.int32)
    final_labels = np.zeros(X.shape[0], dtype=np.int32)
    # Calculate pairwise distances (must only be done once)
    global_distance_matrix = squareform(pdist(X, metric=metric))
    # Start with a single cluster
    current_n_clusters = 1
    tree = BinaryClusterTree()
    while current_n_clusters < n_clusters or construct_full_tree:
        # Get cluster with maximum diameter (largest distance between two poinst within a cluster)
        split_cluster_id, cluster_distance_matrix = _get_cluster_with_max_diameter(
            global_distance_matrix, labels, current_n_clusters, distance_threshold)
        # Check if we only have clusters of size one or only clusters with diameter < distance_threshold
        if split_cluster_id is None:
            break
        else:
            # Split cluster by updating labels and tree
            labels_new = _split_cluster(cluster_distance_matrix, split_cluster_id, current_n_clusters)
            labels[labels == split_cluster_id] = labels_new
            tree.split_cluster(split_cluster_id)
            current_n_clusters += 1
        if current_n_clusters == n_clusters:
            # Save current labels in final labels -> relevant if n_clusters is specified and construct_full_tree is True
            final_labels = labels.copy()
    return final_labels, tree


def _get_cluster_with_max_diameter(global_distance_matrix: np.ndarray, labels: np.ndarray, n_clusters: int,
                                   distance_threshold: float) -> (int, np.ndarray):
    """
    Identify the cluster with the largest diameter, i.e. with the largest distance between two objects assigned to this cluster.
    Here, only diameters which are larger than distance_threshold are taken into account.
    If only clusters of size one occur or all diameters are below distance_threshold, all return values will be None.

    Parameters
    ----------
    global_distance_matrix : np.ndarray
        The global distance matrix containing the pairwise distances of all objects
    labels : np.ndarray
        The current cluster labels
    n_clusters : int
        The current number of clusters
    distance_threshold : float
        The distance thresholds defines the minimum diameter that is considered

    Returns
    -------
    tuple: (int, np.ndarray)
        The id of the cluster that should be split,
        The pariwise distances of all points within that cluster
    """
    max_diameter = -1
    split_cluster_id = None
    resulting_cluster_distance_matrix = None
    # Search cluster with largest diamter (two objects within a cluster with largest distance)
    for cluster_id in range(n_clusters):
        points = labels == cluster_id
        # Cluster must contain more than one object
        if np.sum(points) > 1:
            # Get pairwise distances of all points within cluster
            cluster_distance_matrix = global_distance_matrix[np.ix_(points, points)]
            diameter = np.max(cluster_distance_matrix)
            if diameter > max_diameter and diameter >= distance_threshold:
                # Save parameters of current cluster with largest diameter
                max_diameter = diameter
                split_cluster_id = cluster_id
                resulting_cluster_distance_matrix = cluster_distance_matrix
    return split_cluster_id, resulting_cluster_distance_matrix


def _split_cluster(cluster_distance_matrix: np.ndarray, split_cluster_id: int, new_cluster_id: int) -> np.ndarray:
    """
    Split the specified cluster into two.
    Therefore, it repeatedly calculates the average dissimilarity of the objects to the two subclusters.
    If the subclusters do not change for an iteration the splitting procedure terminates.

    Parameters
    ----------
    cluster_distance_matrix : np.ndarray
        The distance matrix of the specified cluster containing the pairwise distances of respective objects
    split_cluster_id: int
        The id of the cluster that should be split
    new_cluster_id : int
        The resulting id of the new cluster

    Returns
    -------
    labels_new : np.ndarray
        The updated cluster labels
    """
    # Create labels
    labels_new = np.zeros(cluster_distance_matrix.shape[0], dtype=np.int32) + split_cluster_id
    # Initialize sum of distances for second subcluster
    sum_distances_1 = np.sum(cluster_distance_matrix, axis=1)
    sum_distances_2 = np.zeros(cluster_distance_matrix.shape[0])
    splinter_group = np.array([np.argmax(sum_distances_1)])
    # Start splitting procedure
    size_group_1 = cluster_distance_matrix.shape[0] - 1
    size_group_2 = 0
    while splinter_group.shape[0] > 0:
        # Update labels
        labels_new[splinter_group] = new_cluster_id
        # Update sum of distances for each subcluster
        size_group_1 -= splinter_group.shape[0]
        size_group_2 += splinter_group.shape[0]
        sum_splinter_group = np.sum(cluster_distance_matrix[:, splinter_group], axis=1)
        sum_distances_1 -= sum_splinter_group
        sum_distances_2 += sum_splinter_group
        if size_group_1 > 0:
            # Get new splinter group (only checks objects of the original cluster)
            splinter_group = np.where((labels_new == split_cluster_id) &
                                      (sum_distances_1 / size_group_1 > sum_distances_2 / size_group_2))[0]
        else:
            break
    return labels_new


class Diana(BaseEstimator, ClusterMixin):
    """
    The DIvisive ANAlysis (DIANA) clustering algorithm.
    DIANA build a top-down clustering hierarchy by considering pairwise dissimilarity of objects.
    It recursively splits the clusters with maximum dissimilarity, whereby the dissimilarity is based on a specified distance metric (e.g., Euclidean distance).

    Parameters
    ----------
    n_clusters : int
        The number of clusters. If n_clusters is None the tree will be constructed until the max diamater is below distance_threshold (default: None)
    distance_threshold : float
        The distance thresholds defines the minimum diameter that is considered. Must be 0 if n_clusters is specified (default: 0)
    construct_full_tree : bool
        Defines whether the full tree should be constructed after n_clusters has been reached (default: False)
    metric : str
        Metric used to compute the dissimilarity. Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed" (see scipy.spatial.distance.pdist) (default: euclidean)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    tree_ : BinaryClusterTree
        The resulting cluster tree

    References
    ----------
    Kaufman, Rousseeuw "Divisive Analysis (Program DIANA)"
    Chapter six from Finding Groups in Data: An Introduction to Cluster Analysis. 1990.
    """

    def __init__(self, n_clusters: int = None, distance_threshold: float = 0, construct_full_tree: bool = False,
                 metric: str = "euclidean"):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.construct_full_tree = construct_full_tree
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Diana':
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
        self : Diana
            this instance of the Diana algorithm
        """
        assert self.n_clusters is None or self.distance_threshold == 0, "If n_clusters is set, distance_threshold must be 0. Else the number of identified clusters can be incorrect"
        if self.n_clusters is None or self.n_clusters > X.shape[0]:
            self.n_clusters = X.shape[0]
        labels, tree = _diana(X, self.n_clusters, self.distance_threshold, self.construct_full_tree, self.metric)
        self.labels_ = labels
        self.tree_ = tree
        return self

    def flat_clustering(self, n_leaf_nodes_to_keep: int) -> np.ndarray:
        """
        Transform the predicted labels into a flat clustering result by only keeping n_leaf_nodes_to_keep leaf nodes in the tree.
        Returns labels as if the clustering procedure would have stopped at the specified number of nodes.
        Note that each leaf node corresponds to a cluster.

        Parameters
        ----------
        n_leaf_nodes_to_keep : int
            The number of leaf nodes to keep in the cluster tree

        Returns
        -------
        labels_pruned : np.ndarray
            The new cluster labels
        """
        assert self.labels_ is not None, "The DIANA algorithm has not run yet. Use the fit() function first."
        tree_copy = copy.deepcopy(self.tree_)
        labels_pruned = tree_copy.prune_to_n_leaf_nodes(n_leaf_nodes_to_keep, self.labels_)
        return labels_pruned

"""
@authors:
Collin Leiber
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from clustpy.utils._information_theory import bic_costs

"""
HELPERS also used by other classes
"""


def _initial_kmeans_clusters(X: np.ndarray, n_clusters_init: int, random_state: np.random.RandomState) -> (
        int, np.ndarray, np.ndarray, float):
    """
    Get the initial cluster centers and cluster labels based on the n_clusters_init parameter.
    If n_clusters_init is an integer, the cluster parameters are identified by KMeans with n_clusters_init als single input.
    If n_clusters_init is of type np.ndarray, the cluster parameters are identified by KMeans with the initial cluster centers given by n_clusters_init.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray, float)
        The initial number of clusters,
        The initial cluster labels,
        The initial cluster centers,
        The Kmeans error of the initial clustering result
    """
    if type(n_clusters_init) is int and n_clusters_init == 1:
        n_clusters = n_clusters_init
        labels = np.zeros(X.shape[0], dtype=np.int32)
        centers = np.mean(X, axis=0).reshape(1, -1)
        kmeans_error = np.sum((X - centers) ** 2)
    else:
        if type(n_clusters_init) is int:
            # Normally, n_clusters_init is int
            n_clusters = n_clusters_init
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        else:
            # If n_clusters_init is array, this should be equal to the initial cluster centers
            n_clusters = n_clusters_init.shape[0]
            kmeans = KMeans(n_clusters=n_clusters, init=n_clusters_init, n_init=1, random_state=random_state)
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        kmeans_error = kmeans.inertia_
    return n_clusters, labels, centers, kmeans_error


def _execute_two_means(X: np.ndarray, ids_in_each_cluster: list, cluster_id_to_split: int, centers: np.ndarray,
                       n_split_trials: int, random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, float):
    """
    Execute 2-Means.
    Splits a cluster into two by first selecting a random object from the data set as first new cluster and then selects the coordinate on the opposite site of the original center as the second new center.
    Afterwards, KMeans will be executed.
    This procedure is repeated n_split_trials times and the result with the lowest KMeans-error will be returned.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    ids_in_each_cluster : list
        List that contains for each cluster an array with the ids of all objects within this cluster
    cluster_id_to_split : int
        The id of the cluster that should be split
    centers : np.ndarray
        The original cluster centers
    n_split_trials : int
        Number tries to split a cluster. For each try 2-KMeans is executed with different cluster centers
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, float)
        The resulting cluster labels,
        The resuling cluster centers,
        The Kmeans error of the clustering result
    """
    assert X.shape[0] >= 2, "X must contain at least 2 elements"
    # Prepare cluster for splitting
    old_center = centers[cluster_id_to_split, :]
    reduced_centers = np.delete(centers, cluster_id_to_split, axis=0)
    ids_in_cluster = ids_in_each_cluster[cluster_id_to_split]
    # Try to find kmeans result with smallest squared distances
    best_kmeans = None
    # Get random points in cluster as new centers
    n_split_trials = min(n_split_trials, ids_in_cluster.shape[0])
    random_centers = X[random_state.choice(ids_in_cluster, n_split_trials, replace=False), :]
    # Calculate second new centers as: new2 = old - (new1 - old)
    adjusted_centers = old_center - (random_centers - old_center)
    # Get Kmeans result with minimum Kmeans-error
    for i in range(n_split_trials):
        # Run kmeans with new centers
        tmp_centers = np.r_[reduced_centers, [random_centers[i]], [adjusted_centers[i]]]
        kmeans = KMeans(n_clusters=tmp_centers.shape[0], init=tmp_centers, n_init=1, random_state=random_state)
        kmeans.fit(X)
        # Check squared distances to find best kmeans result
        if best_kmeans is None or kmeans.inertia_ < best_kmeans.inertia_:
            best_kmeans = kmeans
    return best_kmeans.labels_, best_kmeans.cluster_centers_, best_kmeans.inertia_


"""
Actual XMeans methods
"""


def _xmeans(X: np.ndarray, n_clusters_init: int, max_n_clusters: int, check_global_score: bool, allow_merging: bool,
            n_split_trials: int, random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray):
    """
    Start the actual XMeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init
    check_global_score : bool
        Defines whether the global BIC score should be checked after the 'Improve-Params' step. Some implementations skip this step
    allow_merging : bool
        Try to merge clusters after the regular XMeans algorithm terminated. See Ishioka et al. for more information
    n_split_trials : int
        Number tries to split a cluster. For each try 2-KMeans is executed with different cluster centers
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The final number of clusters,
        The labels as identified by XMeans,
        The cluster centers as identified by XMeans
    """
    assert max_n_clusters >= n_clusters_init, "max_n_clusters can not be smaller than n_clusters_init"
    n_dims = X.shape[1]
    n_clusters, labels, centers, global_variance = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    # Get parameters of all clusters
    ids_in_each_cluster = [np.where(labels == c)[0] for c in range(n_clusters)]
    cluster_sizes = np.array([ids_in_cluster.shape[0] for ids_in_cluster in ids_in_each_cluster])
    cluster_variances = np.array([np.sum((X[ids_in_each_cluster[c]] - centers[c]) ** 2) / (
            cluster_sizes[c] - 1) for c in range(n_clusters)])  # Only used if allow_merging is True
    if check_global_score:
        # Get initial global variance
        global_variance = global_variance / (X.shape[0] - n_clusters)
        # Get initial global BIC score
        best_global_bic_score = _bic_score(X.shape[0], cluster_sizes, n_dims, global_variance)
        # Save best result
        best_result = (n_clusters, labels, centers, ids_in_each_cluster, cluster_sizes, cluster_variances)
    while n_clusters <= max_n_clusters:
        n_clusters_old = n_clusters
        # Split Clusters => Improve-Structure
        for c in range(n_clusters_old):
            ids_in_cluster = ids_in_each_cluster[c]
            original_cluster_size = cluster_sizes[c]
            if ids_in_cluster.shape[0] <= 2:
                # Cluster can not be split because it is too small
                continue
            # Get variance of original cluster
            cluster_variance = cluster_variances[c]
            # Get BIC score of original cluster
            cluster_bic_score = _bic_score(original_cluster_size, original_cluster_size, n_dims, cluster_variance)
            # Split cluster into two
            labels_split, centers_split, split_variance = _execute_two_means(X[ids_in_cluster],
                                                                             [np.arange(original_cluster_size)], 0,
                                                                             np.array([centers[c]]), n_split_trials,
                                                                             random_state)
            # Get variance of splitted clusters
            split_variance = split_variance / (ids_in_cluster.shape[0] - 2)
            cluster_sizes_split = np.array([np.sum(labels_split == c) for c in range(2)])
            # Get BIC score of splitted clusters
            split_cluster_bic_score = _bic_score(original_cluster_size, cluster_sizes_split, n_dims, split_variance)
            if cluster_bic_score < split_cluster_bic_score:
                # Keep new clusters
                centers[c] = centers_split[0]
                centers = np.r_[centers, [centers_split[1]]]
                labels[ids_in_cluster[labels_split == 1]] = n_clusters
                n_clusters += 1
        # If no cluster changed, XMeans terminates
        if n_clusters == n_clusters_old:
            break
        else:
            # Prepare the clusters for the next iteration => Improve-Params
            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, random_state=random_state)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            # Update parameters of all clusters
            ids_in_each_cluster = [np.where(labels == c)[0] for c in range(n_clusters)]
            cluster_sizes = np.array([ids_in_cluster.shape[0] for ids_in_cluster in ids_in_each_cluster])
            cluster_variances = [np.sum((X[ids_in_each_cluster[c]] - centers[c]) ** 2) / (
                    cluster_sizes[c] - 1) if cluster_sizes[c] > 1 else 0 for c in
                                 range(n_clusters)]  # Only used if allow_merging is True
            if check_global_score:
                # Get new global variance
                global_variance = kmeans.inertia_ / (X.shape[0] - n_clusters)
                # Get new global BIC score
                new_global_bic_score = _bic_score(X.shape[0], cluster_sizes, n_dims, global_variance)
                if best_global_bic_score < new_global_bic_score:
                    # If score improved, save new best model
                    best_global_bic_score = new_global_bic_score
                    best_result = (
                        n_clusters, labels.copy(), centers.copy(), ids_in_each_cluster.copy(), cluster_sizes.copy(),
                        cluster_variances.copy())
    if check_global_score:
        # Exchange latest result with best overall result
        n_clusters, labels, centers, ids_in_each_cluster, cluster_sizes, cluster_variances = best_result
    # OPTIONAL: try to merge clusters
    if allow_merging:
        n_clusters, labels, centers = _merge_clusters(X, n_clusters, labels, centers, ids_in_each_cluster,
                                                      cluster_sizes, cluster_variances)
    return n_clusters, labels, centers


def _merge_clusters(X: np.ndarray, n_clusters: int, labels: np.ndarray, centers: np.ndarray, ids_in_each_cluster: list,
                    cluster_sizes: np.ndarray, cluster_variances: np.ndarray) -> (int, np.ndarray, np.ndarray):
    """
    Addition to XMeans by Ishioka et al..
    Attempts to repair errors caused by an unfortunate splitting order by merging clusters.
    Tests all pairwise combinations of clusters starting with the smallest clusters.
    Here, each original cluster can only be merged once.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        The number of clusters
    labels : np.ndarray
        The cluster labels
    centers : np.ndarray
        The cluster centers
    ids_in_each_cluster : list
        List containing the ids of the samples of a cluster
    cluster_sizes : np.ndarray
        The sizes of the clusters
    cluster_variances : np.ndarray
        The variances of the clusters

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The updated number of clusters,
        The updated labels,
        The updated cluster centers

    References
    ----------
    Ishioka, Tsunenori. "An expansion of X-means for automatically determining the optimal number of clusters."
    Proceedings of International Conference on Computational Intelligence. Vol. 2. 2005.
    """
    n_dims = X.shape[1]
    argsorted_sizes = np.argsort(cluster_sizes)
    already_merged = [False] * n_clusters
    n_cluster_old = n_clusters
    # Check each combination of clusters, starting with the smallest
    for c1_not_sorted in range(n_cluster_old):
        c1 = argsorted_sizes[c1_not_sorted]
        if already_merged[c1]:
            continue
        for c2_not_sorted in range(c1_not_sorted + 1, n_cluster_old):
            c2 = argsorted_sizes[c2_not_sorted]
            if already_merged[c2]:
                continue
            combined_cluster_size = cluster_sizes[c1] + cluster_sizes[c2]
            # Get BIC score of non-merged clusters
            cluster_1_and_2_variance = (cluster_variances[c1] * (cluster_sizes[c1] - 1) +
                                        cluster_variances[c2] * (cluster_sizes[c2] - 1)) / (combined_cluster_size - 2)
            cluster_1_and_2_bic_score = _bic_score(combined_cluster_size, cluster_sizes[[c1, c2]], n_dims,
                                                   cluster_1_and_2_variance)
            # Get BIC of merged cluster
            new_center = (centers[c1] * cluster_sizes[c1] + centers[c2] * cluster_sizes[c2]) / combined_cluster_size
            cluster_merged_variance = np.sum(
                (X[np.r_[ids_in_each_cluster[c1], ids_in_each_cluster[c2]]] - new_center) ** 2) / (
                                              combined_cluster_size - 1)
            cluster_merged_bic_score = _bic_score(combined_cluster_size, combined_cluster_size,
                                                  n_dims, cluster_merged_variance)
            # Is merge improving the local BIC score?
            if cluster_merged_bic_score > cluster_1_and_2_bic_score:
                # Update labels and centers
                min_cluster_id = min(c1, c2)
                max_cluster_id = max(c1, c2)
                labels[labels == max_cluster_id] = min_cluster_id
                centers[min_cluster_id] = new_center
                cluster_sizes[max_cluster_id] = 0
                # Set already_merged for both cluster to True so they can not be merged again
                already_merged[c1] = True
                already_merged[c2] = True
                n_clusters -= 1
                break
    # Remove empty clusters. Needs to be done from max cluster id to min cluster id
    for c in range(n_cluster_old - 1, -1, -1):
        if cluster_sizes[c] == 0:
            labels[labels > c] -= 1
            centers = np.delete(centers, c, axis=0)
    return n_clusters, labels, centers


def _bic_score(n_points: int, cluster_sizes: np.ndarray, n_dims: int, variance: float) -> float:
    """
    Calculate the BIC score of a clustering result.
    For more information see: 'X-means: Extending k-means with efficient estimation of the number of clusters'

    Parameters
    ----------
    n_points : int
        Number of samples in the data set
    cluster_sizes : np.ndarray
        Number of samples in each cluster. Can also by of type int in case of a single cluster
    n_dims : int
        Number of features in the data set
    variance : float
        The variance across all clusters

    Returns
    -------
    bic_total : float
        The BIC score of the cluster
    """
    n_clusters = cluster_sizes.shape[0] if type(cluster_sizes) is np.ndarray else 1
    # BIC of the free parameters
    n_free_params = n_clusters * (n_dims + 1)  # Equal to: (n_clusters - 1) + n_clusters * n_dims + 1
    bic_free_params = n_free_params * bic_costs(n_points, False)
    # BIC of the data using the loglikelihood
    bic_loglikelihood = np.sum(cluster_sizes * (np.log(cluster_sizes) - np.log(n_points) - np.log(
        2.0 * np.pi) / 2 - n_dims * np.log(variance) / 2) - (cluster_sizes - n_clusters) / 2)
    # Combine BIC score components
    bic_total = bic_loglikelihood - bic_free_params
    return bic_total


class XMeans(BaseEstimator, ClusterMixin):
    """

    Parameters
    ----------
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified (default: 2)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init (default: np.inf)
    check_global_score : bool
        Defines whether the global BIC score should be checked after the 'Improve-Params' step. Some implementations skip this step (default: True)
    allow_merging : bool
        Try to merge clusters after the regular XMeans algorithm terminated. See Ishioka et al. for more information.
         Normally, if allow_merging is True, check_global_score should be False (default: False)
    n_split_trials : int
        Number tries to split a cluster. For each try 2-KMeans is executed with different cluster centers (default: 10)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers

    Examples
    ----------
    >>> from clustpy.partition import XMeans
    >>> from sklearn.datasets import make_blobs
    >>> from clustpy.utils import plot_with_transformation
    >>> rs = np.random.RandomState(11)
    >>> X, L = make_blobs(500, 2, centers=1, cluster_std=2, random_state=rs)
    >>> X2, L2 = make_blobs(1000, 2, centers=4, cluster_std=0.5, random_state=rs)
    >>> X = np.r_[X, X2]
    >>> for b in [False, True]:
    >>>     xm = XMeans(allow_merging=b, random_state=rs)
    >>>     xm.fit(X)
    >>>     plot_with_transformation(X, xm.labels_, xm.cluster_centers_)

    References
    ----------
    Pelleg, Dan, and Andrew W. Moore. "X-means: Extending k-means with efficient estimation of the number of clusters."
    Icml. Vol. 1. 2000.

    and

    Ishioka, Tsunenori. "An expansion of X-means for automatically determining the optimal number of clusters."
    Proceedings of International Conference on Computational Intelligence. Vol. 2. 2005.
    """

    def __init__(self, n_clusters_init: int = 2, max_n_clusters: int = np.inf, check_global_score: bool = True,
                 allow_merging: bool = False, n_split_trials: int = 10, random_state: np.random.RandomState = None):
        self.n_clusters_init = n_clusters_init
        self.max_n_clusters = max_n_clusters
        self.check_global_score = check_global_score
        self.allow_merging = allow_merging
        self.n_split_trials = n_split_trials
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'XMeans':
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
        self : XMeans
            this instance of the XMeans algorithm
        """
        n_clusters, labels, centers = _xmeans(X, self.n_clusters_init, self.max_n_clusters, self.check_global_score,
                                              self.allow_merging, self.n_split_trials, self.random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

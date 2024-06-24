"""
@authors:
Collin Leiber
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from clustpy.utils import dip_test, dip_pval, dip_boot_samples
from clustpy.partition.xmeans import _initial_kmeans_clusters, _execute_two_means
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dipmeans(X: np.ndarray, significance: float, split_viewers_threshold: float, pval_strategy: str, n_boots: int,
              n_split_trials: int, n_clusters_init: int, max_n_clusters: int, random_state: np.random.RandomState) -> (
        int, np.ndarray, np.ndarray):
    """
    Start the actual DipMeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    split_viewers_threshold : float
        Threshold to decide whether a cluster has a unimodal or multimodal structure. Must be within [0, 1]
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap'
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap'
    n_split_trials : int
        Number tries to split a cluster. For each try 2-KMeans is executed with different cluster centers
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The final number of clusters,
        The labels as identified by DipMeans,
        The cluster centers as identified by DipMeans
    """
    assert max_n_clusters >= n_clusters_init, "max_n_clusters can not be smaller than n_clusters_init"
    assert significance >= 0 and significance <= 1, "significance must be a value in the range [0, 1]"
    # Calculate distance matrix
    data_dist_matrix = squareform(pdist(X, 'euclidean'))
    # Initialize parameters
    n_clusters, labels, centers, _ = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    while n_clusters <= max_n_clusters:
        # Default score is 0 for all clusters
        cluster_scores = np.zeros(n_clusters)
        ids_in_each_cluster = [np.where(labels == c)[0] for c in range(n_clusters)]
        for c in range(n_clusters):
            ids_in_cluster = ids_in_each_cluster[c]
            # Get pairwise distances of points in cluster
            cluster_dist_matrix = data_dist_matrix[np.ix_(ids_in_cluster, ids_in_cluster)]
            # Calculate dip values for the distances of each point
            cluster_dips = np.array([dip_test(cluster_dist_matrix[p, :], just_dip=True, is_data_sorted=False) for p in
                                     range(ids_in_cluster.shape[0])])
            # Calculate p-values
            if pval_strategy == "bootstrap":
                # Bootstrap values here so it is not needed for each pval separately
                boot_dips = dip_boot_samples(ids_in_cluster.shape[0], n_boots, random_state)
                cluster_pvals = np.array([np.mean(point_dip <= boot_dips) for point_dip in cluster_dips])
            else:
                cluster_pvals = np.array([dip_pval(point_dip, ids_in_cluster.shape[0], pval_strategy=pval_strategy,
                                                   random_state=random_state) for point_dip in cluster_dips])
            # Get split viewers (points with dip-p-value of < significance)
            split_viewers = cluster_dips[cluster_pvals < significance]
            # Check if percentage share of split viewers in cluster is larger than threshold
            if split_viewers.shape[0] / ids_in_cluster.shape[0] > split_viewers_threshold:
                # Calculate cluster score
                cluster_scores[c] = np.mean(split_viewers)
        # Get cluster with maximum score
        cluster_id_to_split = np.argmax(cluster_scores)
        # Check if any cluster has to be split
        if cluster_scores[cluster_id_to_split] > 0:
            # Split cluster using bisecting kmeans
            labels, centers, _ = _execute_two_means(X, ids_in_each_cluster, cluster_id_to_split, centers,
                                                    n_split_trials, random_state)
            n_clusters += 1
        else:
            break
    return n_clusters, labels, centers


class DipMeans(BaseEstimator, ClusterMixin):
    """
    Execute the DipMeans clustering procedure.
    In contrast to other algorithms (e.g. KMeans) DipMeans is able to identify the number of clusters by itself.
    Therefore, it uses the dip-dist criterion.
    It calculates the dip-value of the distances of each point within a cluster to all other points in that cluster and checks how many points are assigned a dip-value below the threshold.
    If that amount of so called split viewers is above the split_viewers_threshold, the cluster will be split using 2-Means.
    The algorithm terminates if all clusters show a unimdoal behaviour.

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal (default: 0.001)
    split_viewers_threshold : float
        Threshold to decide whether a cluster has a unimodal or multimodal structure. Must be within [0, 1] (default: 0.01)
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap' (default: 'table')
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap' (default: 1000)
    n_split_trials : int
        Number tries to split a cluster. For each try 2-KMeans is executed with different cluster centers (default: 10)
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified (default: 1)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init (default: np.inf)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers

    References
    ----------
    Kalogeratos, Argyris, and Aristidis Likas. "Dip-means: an incremental clustering method for estimating the number of clusters."
    Advances in neural information processing systems. 2012.
    """

    def __init__(self, significance: float = 0.001, split_viewers_threshold: float = 0.01,
                 pval_strategy: str = "table", n_boots: int = 1000, n_split_trials: int = 10, n_clusters_init: int = 1,
                 max_n_clusters: int = np.inf, random_state: np.random.RandomState | int = None):
        self.significance = significance
        self.split_viewers_threshold = split_viewers_threshold
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.n_split_trials = n_split_trials
        self.n_clusters_init = n_clusters_init
        self.max_n_clusters = max_n_clusters
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DipMeans':
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
        self : DipMeans
            this instance of the DipMeans algorithm
        """
        n_clusters, labels, centers = _dipmeans(X, self.significance, self.split_viewers_threshold,
                                                self.pval_strategy, self.n_boots, self.n_split_trials,
                                                self.n_clusters_init, self.max_n_clusters, self.random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

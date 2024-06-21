"""
@authors:
Collin Leiber
"""

import numpy as np
from sklearn.decomposition import PCA
from clustpy.utils import dip_test, dip_pval
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from clustpy.partition.xmeans import _initial_kmeans_clusters, _execute_two_means


def _proj_dipmeans(X: np.ndarray, significance: float, n_random_projections: int, pval_strategy: str, n_boots: int,
                   n_split_trials: int, n_clusters_init: int, max_n_clusters: int,
                   random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray):
    """
    Start the actual ProjectedDipMeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    n_random_projections : int
        Number of random projections that should be applied in addition to the projections from PCA
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
        The labels as identified by ProjectedDipMeans,
        The cluster centers as identified by ProjectedDipMeans
    """
    assert max_n_clusters >= n_clusters_init, "max_n_clusters can not be smaller than n_clusters_init"
    assert significance >= 0 and significance <= 1, "significance must a value in the range [0, 1]"
    # Initialize parameters
    n_clusters, labels, centers, _ = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    while n_clusters <= max_n_clusters:
        # Default score is 0 for all clusters
        cluster_scores = np.zeros(n_clusters)
        ids_in_each_cluster = []
        for c in range(n_clusters):
            ids_in_cluster = np.where(labels == c)[0]
            ids_in_each_cluster.append(ids_in_cluster)
            # Get projections
            projected_data = _get_projected_data(X[ids_in_cluster], n_random_projections, random_state)
            # Calculate dip values for the distances of each point
            cluster_dips = np.array([dip_test(projected_data[:, p], just_dip=True, is_data_sorted=False) for p in
                                     range(projected_data.shape[1])])
            # Calculate p-values of maximum dip
            pval = dip_pval(np.max(cluster_dips), ids_in_cluster.shape[0], pval_strategy=pval_strategy, n_boots=n_boots,
                            random_state=random_state)
            # Calculate cluster score
            cluster_scores[c] = pval
        # Get cluster with minimum pval
        cluster_id_to_split = np.argmin(cluster_scores)
        # Check if any cluster has to be split
        if cluster_scores[cluster_id_to_split] < significance:
            # Split cluster using bisecting kmeans
            labels, centers, _ = _execute_two_means(X, ids_in_each_cluster, cluster_id_to_split, centers,
                                                    n_split_trials, random_state)
            n_clusters += 1
        else:
            break
    return n_clusters, labels, centers


def _get_projected_data(X: np.ndarray, n_random_projections: int, random_state: np.random.RandomState) -> np.ndarray:
    """
    Get the objects of a cluster projcted onto different projection axes.
    First projection are the original features.
    Furhter, the objects will be projected onto each component of a PCA.
    Additionally, the objects can be projected onto random axes.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_random_projections : int
        Number of random projections that should be applied in addition to the projections from PCA
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    projections_data : np.ndarray
        The data projected onto multiple projection axes
    """
    # Execute PCA
    pca = PCA()
    pca_X = pca.fit_transform(X) if X.shape[0] > 1 else np.empty((X.shape[0], 0))
    # Add random projections
    random_projections = random_state.rand(X.shape[1], n_random_projections)
    random_X = np.matmul(X, random_projections)
    # Combine data
    projected_data = np.c_[X, pca_X, random_X]
    return projected_data


class ProjectedDipMeans(BaseEstimator, ClusterMixin):
    """
    Execute the Projected DipMeans clustering procedure.
    It repeatedly creates random projection axes for each cluster and tests whether the data projected onto that projection axis is unimodal.
    If the probability of unimodality is below significance another the cluster will be split.
    This is done by using 2-Means.
    The algorithm terminates if all clusters show a unimdoal behaviour on all projection axes.

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal (default: 0.001)
    n_random_projections : int
        Number of random projections that should be applied in addition to the original features and the components from a PCA (default: 0)
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
    Chamalis, Theofilos, and Aristidis Likas. "The projected dip-means clustering algorithm."
    Proceedings of the 10th Hellenic Conference on Artificial Intelligence. 2018.
    """

    def __init__(self, significance: float = 0.001, n_random_projections: int = 0, pval_strategy: str = "table",
                 n_boots: int = 1000, n_split_trials: int = 10, n_clusters_init: int = 1, max_n_clusters: int = np.inf,
                 random_state: np.random.RandomState | int = None):
        self.significance = significance
        self.n_random_projections = n_random_projections
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.n_split_trials = n_split_trials
        self.n_clusters_init = n_clusters_init
        self.max_n_clusters = max_n_clusters
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ProjectedDipMeans':
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
        self : ProjectedDipMeans
            this instance of the ProjectedDipMeans algorithm
        """
        n_clusters, labels, centers = _proj_dipmeans(X, self.significance, self.n_random_projections,
                                                     self.pval_strategy, self.n_boots, self.n_split_trials,
                                                     self.n_clusters_init, self.max_n_clusters, self.random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

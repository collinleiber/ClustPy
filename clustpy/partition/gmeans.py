"""
@authors:
Collin Leiber
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from clustpy.partition.xmeans import _initial_kmeans_clusters, _execute_two_means
from scipy.stats import anderson


def _gmeans(X: np.ndarray, significance: float, n_clusters_init: int, max_n_clusters: int, n_split_trials: int,
            random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray):
    """
    Start the actual GMeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the Anderson Darling Test indicates a Gaussian distribution
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init
    n_split_trials : int
        Number tries to split a cluster. For each try 2-KMeans is executed with different cluster centers
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The final number of clusters,
        The labels as identified by GMeans,
        The cluster centers as identified by GMeans
    """
    assert max_n_clusters >= n_clusters_init, "max_n_clusters can not be smaller than n_clusters_init"
    assert significance >= 0 and significance <= 1, "significance must be a value in the range [0, 1]"
    # Initialize parameters
    n_clusters, labels, centers, _ = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    while n_clusters <= max_n_clusters:
        n_clusters_old = n_clusters
        for c in range(n_clusters_old):
            ids_in_cluster = np.where(labels == c)[0]
            if ids_in_cluster.shape[0] < 2:
                continue
            # Split cluster into two
            labels_split, centers_split, _ = _execute_two_means(X[ids_in_cluster], [np.arange(ids_in_cluster.shape[0])], 0,
                                                             np.array([centers[c]]), n_split_trials, random_state)
            # Project data form cluster onto resulting connection axis
            projected_data = np.dot(X[ids_in_cluster], centers_split[0] - centers_split[1])
            # Use Anderson Darling to test if data is Gaussian
            ad_result = anderson(projected_data, "norm")
            p_value = _anderson_darling_statistic_to_prob(ad_result.statistic, len(ids_in_cluster))
            if p_value < significance:
                # If data is not Gaussian, keep the newly created cluster centers
                centers[c] = centers_split[0]
                centers = np.r_[centers, [centers_split[1]]]
                labels[ids_in_cluster[labels_split == 1]] = n_clusters
                n_clusters += 1
        # If no cluster changed, GMeans terminates
        if n_clusters == n_clusters_old:
            break
        else:
            # Prepare the cluster for the next iteration
            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, random_state=random_state)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
    return n_clusters, labels, centers


def _anderson_darling_statistic_to_prob(statistic: float, n_points: int) -> float:
    """
    Transform the statistic returned by the Anderson Darling test into a p_value.
    First the adjusted statistic will be calculated.
    Afterwards, the actual p-value can be obtained.

    Parameters
    ----------
    statistic : float
        The original statistic from the Anderson Darling test.
    n_points : int
        The number of samples

    Returns
    -------
    p_value : float
        The p-value

    References
    ----------
    D'Agostino, Ralph B., and Michael A. Stephens. "Goodness-of-fit techniques."
    Statistics: Textbooks and Monographs (1986).
    """
    adjusted_stat = statistic * (1 + (.75 / n_points) + 2.25 / (n_points ** 2))
    if adjusted_stat < 0.2:
        # is log q => therefore add 1 - ...
        p_value = 1 - np.exp(-13.436 + 101.14 * adjusted_stat - 223.73 * (adjusted_stat ** 2))
    elif adjusted_stat < 0.34:
        # is log q => therefore add 1 - ...
        p_value = 1 - np.exp(-8.318 + 42.796 * adjusted_stat - 59.938 * (adjusted_stat ** 2))
    elif adjusted_stat < 0.6:
        p_value = np.exp(0.9177 - 4.279 * adjusted_stat - 1.38 * (adjusted_stat ** 2))
    else:
        p_value = np.exp(1.2937 - 5.709 * adjusted_stat - 0.0186 * (adjusted_stat ** 2))
    return p_value


class GMeans(BaseEstimator, ClusterMixin):
    """
    Execute the GMeans clustering procedure.
    Determines the number of clusters by repeatedly trying to split a cluster into two clusters.
    Therefore, the data is projected onto the axis connecting the two resulting centers.
    If the Anderson Darling test does not assume a Gaussian distribution for this projection, the new clusters are retained.
    Otherwise, the cluster remains as it was originally. This is repeated until no cluster changes.

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the Anderson Darling Test indicates a Gaussian distribution (default: 0.001)
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified (default: 1)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init (default: np.inf)
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

    References
    ----------
    Hamerly, Greg, and Charles Elkan. "Learning the k in k-means."
    Advances in neural information processing systems. 2004.

    and

    D'Agostino, Ralph B., and Michael A. Stephens. "Goodness-of-fit techniques."
    Statistics: Textbooks and Monographs (1986).
    """

    def __init__(self, significance: float = 0.001, n_clusters_init: int = 1, max_n_clusters: int = np.inf,
                 n_split_trials: int = 10, random_state: np.random.RandomState = None):
        self.significance = significance
        self.n_clusters_init = n_clusters_init
        self.max_n_clusters = max_n_clusters
        self.n_split_trials = n_split_trials
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'GMeans':
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
        self : GMeans
            this instance of the GMeans algorithm
        """
        n_clusters, labels, centers = _gmeans(X, self.significance, self.n_clusters_init, self.max_n_clusters,
                                              self.n_split_trials, self.random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

"""
@authors:
Collin Leiber
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


def _gap_statistic(X: np.ndarray, min_n_clusters: int, max_n_clusters: int, n_boots: int,
                   use_principal_components: bool, use_log: bool, random_state: np.random.RandomState) -> (
        int, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Start the actual Gap Statistic procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    min_n_clusters : int
        Minimum number of clusters. Must be smaller than max_n_clusters
    max_n_clusters : int
        Maximum number of clusters. Must be larger than min_n_clusters
    n_boots : int
        Number of random data sets that should be created to calculate Gap Statistic
    use_principal_components : bool
        True, if the random data sets should be created using the feature-wise minimum and maximum value of the Principle Components.
        Else, the minimum and maximum value of the regular data set will be used
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used
        For more information see Mohajer et al.
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The first number of clusters that fulfills the Gap condition (can be None),
        The labels as identified by the Gap Statistic (can be None),
        The cluster centers as identified by the Gap Statistic (Can be None),
        The Gap values,
        The sk values
    """
    assert max_n_clusters >= min_n_clusters, "max_n_clusters can not be smaller than min_n_clusters"
    assert n_boots > 0, "n_boots must be larger than 0"
    # Get min and max values for each dimension
    if use_principal_components:
        pca = PCA(n_components=X.shape[1])
        X_transformed = pca.fit_transform(X)
    else:
        pca = None
        X_transformed = X
    mins = np.min(X_transformed, axis=0)
    maxs = np.max(X_transformed, axis=0)
    # Prepare parameters
    gaps = np.zeros(max_n_clusters + 2 - min_n_clusters)
    sks = np.zeros(max_n_clusters + 2 - min_n_clusters)
    all_labels = np.zeros((X.shape[0], max_n_clusters + 2 - min_n_clusters), dtype=np.int32)
    random_datasets = [_generate_random_data(X.shape, mins, maxs, pca, random_state) for _ in
                       range(n_boots)]
    for n_clusters in range(min_n_clusters, max_n_clusters + 2):  # +1 because we need to calculate Gap(k+1)
        # Execute KMeans on the original data
        labels, W_k = _execute_kmeans(X, n_clusters, use_log, random_state)
        # Save labels
        all_labels[:, n_clusters - min_n_clusters] = labels
        # Create random data
        W_kbs = np.zeros(n_boots)
        for b in range(n_boots):
            # Execute KMeans on random data
            labels, W_kb = _execute_kmeans(random_datasets[b], n_clusters, use_log, random_state)
            # Save within cluster dispersion
            W_kbs[b] = W_kb
        # Calculate Gap Statistic
        gaps[n_clusters - min_n_clusters] = np.mean(W_kbs) - W_k
        sks[n_clusters - min_n_clusters] = np.std(W_kbs) * np.sqrt(1 + 1 / n_boots)
    # Check if any result fulfills gap condition
    fulfills_gap = gaps[:-1] >= gaps[1:] - sks[1:]
    # Prepare final result
    if np.any(fulfills_gap):
        best_index = np.where(fulfills_gap)[0][0]
        best_n_clusters = best_index + min_n_clusters
        best_labels = all_labels[:, best_index]
        best_centers = np.array([np.mean(X[best_labels == c], axis=0) for c in range(best_n_clusters)])
    else:
        best_n_clusters = None
        best_labels = None
        best_centers = None
    return best_n_clusters, best_labels, best_centers, gaps, sks


def _generate_random_data(data_shape: tuple, mins: np.ndarray, maxs: np.ndarray, pca: PCA,
                          random_state: np.random.RandomState) -> np.ndarray:
    """
    Create a random data set using a uniform distribution and the feature-wise min and max values of the data set.
    If a PCA was used, rotate the data set back into the original feature space.

    Parameters
    ----------
    data_shape : tuple
        The data shape
    mins : np.ndarray
        The feature-wise minimum values
    maxs : np.ndarray
        The feature-wise maximum values
    pca : PCA
        The PCA object used to calculate mins and maxs. Can be None, if principle components are not used
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    random_samples : np.ndarray
        The randomly created data set
    """
    random_dataset = random_state.random(size=data_shape) * (maxs - mins) + mins
    if pca is not None:
        random_dataset = pca.inverse_transform(random_dataset)
    return random_dataset


def _execute_kmeans(X: np.ndarray, n_clusters: int, use_log: bool, random_state: np.random.RandomState) -> (
        np.ndarray, float):
    """
    Execute KMeans on the given data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        The number of clusters
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used
        For more information see Mohajer et al.
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, float)
        The cluster labels,
        The within cluster dispersion
    """
    if n_clusters > 1:
        kmeans = KMeans(n_clusters, random_state=random_state)
        kmeans.fit(X)
        labels = kmeans.labels_
        # Calculate within cluster dispersion
        W_k = np.log(kmeans.inertia_) if use_log else kmeans.inertia_  # Equal to D_k = sum_k(D_r / (2n))
    else:
        labels = np.zeros(X.shape[0])
        # Calculate within cluster dispersion
        W_k = np.sum(pdist(X, metric="sqeuclidean")) / X.shape[0]
        W_k = np.log(W_k) if use_log else W_k
    return labels, W_k


class GapStatistic(BaseEstimator, ClusterMixin):
    """
    Estimate the number of cluster for KMeans using the Gar Statistic.
    Calculate the Gap Statistic by comparing within cluster dispersion of the input data set with that of ranomly sampled data.
    The Gap Statistic is evaluated for multiple numebers of clusters.
    First clustering result that fulfills the Gap condition 'Gap(k) >= Gap(k+1)-s_{k+1}' will be returned.
    Beware: Result can be None if no clustering result fulfills that condition!

    Parameters
    ----------
    min_n_clusters : int
        Minimum number of clusters. Must be smaller than max_n_clusters (default: 1)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than min_n_clusters (default: 10)
    n_boots : int
        Number of random data sets that should be created to calculate Gap Statistic (default: 10)
    use_principal_components : bool
        True, if the random data sets should be created using the feature-wise minimum and maximum value of the Principle Components.
        Else, the minimum and maximum value of the regular data set will be used (default: True)
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used.
        For more information see Mohajer et al. (default: True)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The first number of clusters that fulfills the Gap condition (can be None)
    labels_ : np.ndarray
        The labels as identified by the Gap Statistic (can be None)
    cluster_centers_ : np.ndarray
        The cluster centers as identified by the Gap Statistic (Can be None)
    gaps_ : np.ndarray
        The Gap values,
    sks_ : np.ndarray
        The sk values

    Examples
    ----------
    >>> from sklearn.datasets import make_blobs
    >>> X, L = make_blobs(1000, 2, centers=7, cluster_std=0.7)
    >>> gs = GapStatistic()
    >>> gs.fit(X)
    >>> print(gs.n_clusters_)
    >>> gs.plot()

    References
    ----------
    Tibshirani, Robert, Guenther Walther, and Trevor Hastie. "Estimating the number of clusters in a data set via the gap statistic."
    Journal of the Royal Statistical Society: Series B (Statistical Methodology) 63.2 (2001): 411-423.

    and

    Mohajer, Mojgan, Karl-Hans Englmeier, and Volker J. Schmid. "A comparison of Gap statistic definitions with and without logarithm function."
    arXiv preprint arXiv:1103.4767 (2011).
    """

    def __init__(self, min_n_clusters: int = 1, max_n_clusters: int = 10, n_boots: int = 10,
                 use_principal_components: bool = True, use_log: bool = True,
                 random_state: np.random.RandomState | int = None):
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.n_boots = n_boots
        self.use_principal_components = use_principal_components
        self.use_log = use_log
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'GapStatistic':
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
        self : GapStatistic
            this instance of the GapStatistic algorithm
        """
        n_clusters, labels, centers, gaps, sks = _gap_statistic(X, self.min_n_clusters, self.max_n_clusters,
                                                                self.n_boots,
                                                                self.use_principal_components, self.use_log,
                                                                self.random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.gaps_ = gaps
        self.sks_ = sks
        return self

    def plot(self) -> None:
        """
        Plot the result of the Gap Statistic.
        Shows the number of the clusters on the x-axis and the Gap values on the y-axis.
        """
        assert hasattr(self, "gaps_"), "The Gap Statistic algorithm has not run yet. Use the fit() function first."
        plt.plot(np.arange(self.min_n_clusters, self.max_n_clusters + 1), self.gaps_[:-1])
        plt.errorbar(np.arange(self.min_n_clusters, self.max_n_clusters + 1), self.gaps_[:-1], self.sks_[:-1],
                     capsize=3, linestyle='None')
        plt.ylabel("Gap Statistic")
        plt.xlabel("n_clusters")
        plt.show()

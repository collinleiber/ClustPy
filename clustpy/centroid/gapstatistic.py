"""
@authors:
Collin Leiber
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt
from clustpy.utils.checks import check_parameters
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils import check_random_state
from clustpy.metrics._metrics_utils import _check_length_data_and_labels
import inspect


def gap_statistic_score(X: np.ndarray, labels: np.ndarray, n_clusters: int | None = None, return_sk : bool = False,
                  use_log: bool = True, weighted: bool = False, use_principal_components: bool = True, 
                  inertia: float | None = None, cluster_centers : np.ndarray | None = None, 
                  clustering_algorithm : ClusterMixin = KMeans, clustering_params: dict | None = None, 
                  bootstrapped_data: np.ndarray | None = None, n_boots: int = 50, 
                  random_state: np.random.RandomState | int | None = None) -> float | tuple[float, float]:
    """
    Calculate the Gap Statistic for the given clustering result.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        the given cluster labels
    n_clusters : int | None
        The number of clusters. If None, it will be extraced from the labels (default: None)
    return_sk : bool
        Return the s_k value in addition to the Gap (default: False)
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used.
        For more information see Mohajer et al. "A comparison of Gap statistic definitions with and without logarithm function." (default: True)
    weighted : bool
        True if weighted Gap should be calculated.
        For more information see Yan and Ye "Determining the Number of Clusters Using the Weighted Gap Statistic." (default: False)
    use_principal_components : bool
        True, if the random data sets should be created using the feature-wise minimum and maximum value of the Principle Components.
        Else, the minimum and maximum value of the regular data set will be used (default: True)
    inertia : float | None
        The inertia of the given clustering result. If None, it will be calculated from the labels (default: None)
    cluster_centers : np.ndarray | None
        The cluster centers of the given clustering result. If None, it will be calculated from the labels (default: None)
    clustering_algorithm : ClusterMixin
        The clustering algorithm run on the bootstraped data (default: KMeans)
    clustering_params : dict | None
        The parameters for the clustering algorithm. If None, it will be equal to {} (default: None)
    bootstrapped_data : np.ndarray | None
        The bootstrapped data sets. If None, a new set of n_boots datasets will be created (default: None)
    n_boots : int
        Number of random data sets that should be created to calculate Gap Statistic. Has to match bootstrapped_data (default: 50)
    random_state : np.random.RandomState | int | None
        use a fixed random state to get a repeatable solution (default: None)

    Returns
    -------
    tuple : float | tuple[float, float]
        The Gap value,
        Optionally the s_k value

    References
    ----------
    Tibshirani, Robert, Guenther Walther, and Trevor Hastie. "Estimating the number of clusters in a data set via the gap statistic."
    Journal of the Royal Statistical Society: Series B (Statistical Methodology) 63.2 (2001): 411-423.

    and

    Yan, Mingjin, and Keying Ye. "Determining the number of clusters using the weighted gap statistic." 
    Biometrics 63.4 (2007): 1031-1037.

    and

    Mohajer, Mojgan, Karl-Hans Englmeier, and Volker J. Schmid. "A comparison of Gap statistic definitions with and without logarithm function."
    arXiv preprint arXiv:1103.4767 (2011).
    """
    X, labels = _check_length_data_and_labels(X, labels, allow_single_cluster=True)
    random_state = check_random_state(random_state)
    if n_clusters is None:
        if cluster_centers is not None:
            n_clusters = cluster_centers.shape[0]
        else:
            n_clusters = len(np.unique(labels))
    # Get reference data
    assert n_boots > 0, "n_boots must be larger than 0"
    if bootstrapped_data is None:
        # Get min and max values for each dimension
        mins, maxs, pca = _get_data_min_max(X, use_principal_components)
        # Create bootstrapped data
        bootstrapped_data = [_generate_random_data(X.shape, mins, maxs, pca, random_state) for _ in range(n_boots)]
    assert len(bootstrapped_data) == n_boots, f"len(bootstrapped_data) must be equal to n_boots. Your values: {len(bootstrapped_data)} vs {n_boots}"
    # Get within-cluster disperion for the input data
    W_k = _get_within_cluster_dispersion(X, labels, cluster_centers, inertia, use_log, weighted)
    # Get within-dispersion measures
    W_kbs = np.zeros(n_boots)
    for b in range(n_boots):
        # Execute clustering algorithm on random data
        labels_boot, centers_boot, inertia_boot = _execute_clusterer(bootstrapped_data[b], n_clusters, clustering_algorithm, clustering_params, random_state)
        # Save within cluster dispersion
        W_kbs[b] = _get_within_cluster_dispersion(bootstrapped_data[b], labels_boot, centers_boot, inertia_boot, use_log, weighted)
    # Calculate Gap Statistic
    gap = np.mean(W_kbs) - W_k
    if return_sk:
        sk = np.sqrt(1 + 1 / n_boots) * np.std(W_kbs)
        return gap, sk
    else:
        return gap


def _gap_statistic_clusterer(X: np.ndarray, min_n_clusters: int, max_n_clusters: int, stopping_criterion: str, use_log: bool, 
                             weighted: bool, use_principal_components: bool, clustering_algorithm : ClusterMixin, 
                             clustering_params : dict | None, n_boots: int, 
                             random_state: np.random.RandomState) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Start the actual Gap Statistic procedure on the input data set.
    If no satisfactory result can be identified, the number of clusters will be set to one.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    min_n_clusters : int
        Minimum number of clusters. Must be smaller than max_n_clusters
    max_n_clusters : int
        Maximum number of clusters. Must be larger than min_n_clusters
    stopping_criterion : str
        Defines when to stop the process. Can be:
        - "original": Abort the process when a result with Gap[k-1] >= Gap[k] - sks[k] is identified
        - "original-all": Returns the same result as original, but calculates all Gaps until max_n_clusters
        - "max": Returns the maximum gap identified
        - "ddgap": Returns the maximum DDGap value, defined as 2 * Gap[k] - Gap[k-1] - Gap[k+1]
        (see Yan and Ye "Determining the Number of Clusters Using the Weighted Gap Statistic.")
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used.
        For more information see Mohajer et al. "A comparison of Gap statistic definitions with and without logarithm function."
    weighted : bool
        True if weighted Gap should be calculated.
        For more information see Yan and Ye "Determining the Number of Clusters Using the Weighted Gap Statistic."
    use_principal_components : bool
        True, if the random data sets should be created using the feature-wise minimum and maximum value of the Principle Components.
        Else, the minimum and maximum value of the regular data set will be used
    clustering_algorithm : ClusterMixin
        The clustering algorithm run on the bootstraped data
    clustering_params : dict | None
        The parameters for the clustering algorithm. If None, it will be equal to {}
    n_boots : int
        Number of random data sets that should be created to calculate Gap Statistic. Has to match bootstrapped_data
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
        The first number of clusters that fulfills the Gap condition (can be None),
        The labels as identified by the Gap Statistic (can be None),
        The cluster centers as identified by the Gap Statistic (Can be None),
        The Gap values,
        The sk values
    """
    assert max_n_clusters >= min_n_clusters, "max_n_clusters can not be smaller than min_n_clusters"
    max_n_clusters = min(max_n_clusters, X.shape[0] - 1)
    assert n_boots > 0, "n_boots must be larger than 0"
    # Get min and max values for each dimension
    mins, maxs, pca = _get_data_min_max(X, use_principal_components)
    # Create bootstrapped data
    bootstrapped_data = [_generate_random_data(X.shape, mins, maxs, pca, random_state) for _ in range(n_boots)]
    # Prepare parameters
    gaps = []
    sks = []
    all_labels = []
    fulfills_gap_idx = None
    for n_clusters in range(min_n_clusters, max_n_clusters + 2): # +1 because max_n_clusters should be a potential output (we need to calculate Gap(k+1)) 
        # Execute clustering on the original data
        labels, centers, inertia = _execute_clusterer(X, n_clusters, clustering_algorithm, clustering_params, random_state)
        # Save labels
        all_labels.append(labels)
        gap_boot, sk_boot = gap_statistic_score(X, labels, n_clusters, True, use_log, weighted, use_principal_components, inertia, centers, 
                                                clustering_algorithm, clustering_params, bootstrapped_data, n_boots, random_state)
        # Save Gap Statistic
        gaps.append(gap_boot)
        sks.append(sk_boot)
        if fulfills_gap_idx is None and n_clusters != min_n_clusters and gaps[-2] >= gaps[-1] - sks[-1]:
            fulfills_gap_idx = len(gaps) - 2
            if stopping_criterion == "original":
                break
    gaps = np.array(gaps)
    sks = np.array(sks)
    ddgaps = None
    if stopping_criterion == "max":
        fulfills_gap_idx = np.argmax(gaps)
    elif stopping_criterion == "ddgap":
        ddgaps = 2 * gaps[1:-1] - gaps[:-2] - gaps[2:]
        fulfills_gap_idx = np.argmax(ddgaps) + 1
    # Prepare final result
    if fulfills_gap_idx is not None:
        best_n_clusters = fulfills_gap_idx + min_n_clusters
        best_labels = all_labels[fulfills_gap_idx]
    else:
        best_n_clusters = 1
        best_labels = np.zeros(X.shape[0], dtype=np.int32)
    best_centers = np.array([np.mean(X[best_labels == c], axis=0) for c in range(best_n_clusters)])
    return best_n_clusters, best_labels, best_centers, gaps, sks, ddgaps


def _get_data_min_max(X: np.ndarray, use_principal_components: bool) -> tuple[np.ndarray, np.ndarray, PCA | None]:
    """
    Get min and max values for each feature of the given data.
    If use_principal_components is True, the dataset will be rotated according to the principal components before calculating the mins and maxs.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    use_principal_components : bool
        True, if the random data sets should be created using the feature-wise minimum and maximum value of the Principle Components.
        Else, the minimum and maximum value of the regular data set will be used

    Returns
    -------
    tuple : tuple[np.ndarray, np.ndarray, PCA | None]
        The min values, max values and the pca object (can be None)
    """
    # Get min and max values for each dimension
    if use_principal_components and X.shape[0] >= X.shape[1]:
        pca = PCA(n_components=X.shape[1])
        X_transformed = pca.fit_transform(X)
    else:
        pca = None
        X_transformed = X
    mins = np.min(X_transformed, axis=0)
    maxs = np.max(X_transformed, axis=0)
    return mins, maxs, pca


def _generate_random_data(data_shape: tuple, mins: np.ndarray, maxs: np.ndarray, pca: PCA | None,
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
    pca : PCA | None
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


def _execute_clusterer(X: np.ndarray, n_clusters: int, clustering_algorithm: ClusterMixin, clustering_params: dict | None, 
                       random_state: np.random.RandomState) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """
    Execute KMeans on the given data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        The number of clusters
    clustering_algorithm : ClusterMixin
        The clustering algorithm run on the bootstraped data
    clustering_params : dict | None
        Parameters for the clustering algorithm
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : tuple[np.ndarray, np.ndarray | None, float | None]
        The cluster labels,
        The cluster centers,
        The inertia
    """
    inertia = None
    centers = None
    if n_clusters > 1:
        if clustering_params is None:
            params_copy = {}
        else:
            params_copy = clustering_params.copy()
        algo_input_params = inspect.getfullargspec(clustering_algorithm).args + inspect.getfullargspec(clustering_algorithm).kwonlyargs
        if "n_clusters" in algo_input_params and "n_clusters" not in params_copy:
            params_copy["n_clusters"] = n_clusters
        elif "n_components" in algo_input_params and "n_components" not in params_copy:
            params_copy["n_components"] = n_clusters
        if "random_state" in algo_input_params and "random_state" not in params_copy:
            params_copy["random_state"] = random_state
        clusterer = clustering_algorithm(**params_copy)
        labels = clusterer.fit_predict(X)
        if hasattr(clusterer, "inertia_"):
            inertia = clusterer.inertia_ # Equal to D_k = sum_k(D_r / (2n))
        if hasattr(clusterer, "cluster_centers_"):
            centers = clusterer.cluster_centers_
        elif hasattr(clusterer, "means_"):
            centers = clusterer.means_
    else:
        labels = np.zeros(X.shape[0], dtype=np.int32)
    return labels, centers, inertia


def _get_within_cluster_dispersion(X: np.ndarray, labels : np.ndarray, centers : np.ndarray | None, 
                                   inertia : float | None, use_log: bool, weighted: bool) -> float:
    """
    Get the within cluster disperion (inertia).
    Calculated as sum_k sum_{x in C_k} |x-mu_k|^2 = sum_k 1/(2|C_k|) sum_{x,z in C_k} |x-z|^2

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        the given cluster labels
    centers : np.ndarray | None
        The cluster centers of the given clustering result. If None, it will be calculated from the labels
    inertia : float | None
        The inertia of the given clustering result. If None, it will be calculated from the labels
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used.
        For more information see Mohajer et al. "A comparison of Gap statistic definitions with and without logarithm function."
    weighted : bool
        True if weighted Gap should be calculated.
        For more information see Yan and Ye "Determining the Number of Clusters Using the Weighted Gap Statistic."

    Returns
    -------
    tuple : float
        The inertia
    """
    if inertia is None or weighted:
        if centers is None:
            n_clusters = len(np.unique(labels))
            centers = np.array([np.mean(X[labels == l], axis=0) for l in range(n_clusters)])
        else:
            n_clusters = centers.shape[0]
        # Calculate within cluster dispersion
        assert centers.ndim == 2, f"Centers have to be of shape (k, d) but the shaepe is {centers.shape}"
        if weighted:
            cluster_sizes = np.array([np.sum(labels == l) for l in range(n_clusters)])
            squared_distances = [np.sum((X[labels == l] - centers[l]) ** 2) / (cluster_sizes[l] - 1) if cluster_sizes[l] != 1 else 0 for l in range(n_clusters)]
        else:
            squared_distances = [np.sum((X[labels == l] - centers[l]) ** 2) for l in range(n_clusters)]
        inertia = np.sum(squared_distances)
    W_k = np.log(inertia) if use_log else inertia
    return W_k


class GapStatistic(ClusterMixin, BaseEstimator):
    """
    Estimate the number of cluster using the Gap Statistic.
    Calculate the Gap Statistic by comparing within cluster dispersion of the input data set with that of ranomly sampled data.
    The Gap Statistic is evaluated for multiple numebers of clusters.
    First clustering result that fulfills the Gap condition 'Gap(k) >= Gap(k+1)-s_{k+1}' will be returned.
    Beware: n_clusters will be 1 if no clustering result fulfills that condition!

    Parameters
    ----------
    min_n_clusters : int
        Minimum number of clusters. Must be smaller than max_n_clusters (default: 1)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than min_n_clusters (default: 10)
    stopping_criterion : str
        Defines when to stop the process. Can be:
        - "original": Abort the process when a result with Gap[k-1] >= Gap[k] - sks[k] is identified
        - "original-all": Returns the same result as original, but calculates all Gaps until max_n_clusters
        - "max": Returns the maximum gap identified
        - "ddgap": Returns the maximum DDGap value, defined as 2 * Gap[k] - Gap[k-1] - Gap[k+1]
        (see Yan and Ye "Determining the Number of Clusters Using the Weighted Gap Statistic.") (default: original)
    use_log : bool
        True, if the logarithm of the within cluster dispersion should be used.
        For more information see Mohajer et al. "A comparison of Gap statistic definitions with and without logarithm function." (default: True)
    weighted : bool
        True if weighted Gap should be calculated.
        For more information see Yan and Ye "Determining the Number of Clusters Using the Weighted Gap Statistic." (default: False)
    use_principal_components : bool
        True, if the random data sets should be created using the feature-wise minimum and maximum value of the Principle Components.
        Else, the minimum and maximum value of the regular data set will be used (default: True)
    clustering_algorithm : ClusterMixin
        The clustering algorithm run on the bootstraped data (default: KMeans)
    clustering_params : dict | None
        The parameters for the clustering algorithm. If None, it will be equal to {} (default: None)
    n_boots : int
        Number of random data sets that should be created to calculate Gap Statistic (default: 50)
    random_state : np.random.RandomState | int | None
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The first number of clusters that fulfills the Gap condition (one if no results fulfills the statistic)
    labels_ : np.ndarray
        The labels
    cluster_centers_ : np.ndarray
        The cluster centers 
    gaps_ : np.ndarray
        The Gap values,
    sks_ : np.ndarray
        The sk values
    n_features_in_ : int
        the number of features used for the fitting

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

    Yan, Mingjin, and Keying Ye. "Determining the number of clusters using the weighted gap statistic." 
    Biometrics 63.4 (2007): 1031-1037.

    and

    Mohajer, Mojgan, Karl-Hans Englmeier, and Volker J. Schmid. "A comparison of Gap statistic definitions with and without logarithm function."
    arXiv preprint arXiv:1103.4767 (2011).
    """

    def __init__(self, min_n_clusters: int = 1, max_n_clusters: int = 10, stopping_criterion: str = "original", 
                 use_log: bool = True, weighted: bool= False, use_principal_components: bool = True, 
                 clustering_algorithm: ClusterMixin = KMeans,
                 clustering_params: dict | None = None, n_boots: int = 50, random_state: np.random.RandomState | int = None):
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.stopping_criterion = stopping_criterion
        self.use_log = use_log
        self.weighted = weighted
        self.use_principal_components = use_principal_components
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params
        self.n_boots = n_boots
        self.random_state = random_state

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
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        stopping_criterion = self.stopping_criterion.lower()
        assert stopping_criterion in ["original", "original-all", "max", "ddgap"]
        n_clusters, labels, centers, gaps, sks, ddgaps = _gap_statistic_clusterer(X, self.min_n_clusters, self.max_n_clusters,
                                                                          stopping_criterion, self.use_log, self.weighted, 
                                                                          self.use_principal_components, 
                                                                          self.clustering_algorithm, self.clustering_params, 
                                                                          self.n_boots, random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.gaps_ = gaps
        self.sks_ = sks
        self.ddgaps_ = ddgaps
        self.n_features_in_ = X.shape[1]
        return self

    def plot(self, add_ddgap: bool = False) -> None:
        """
        Plot the result of the Gap Statistic.
        Shows the number of the clusters on the x-axis and the Gap values on the y-axis.

        Parameters
        ----------
        add_ddgap : bool
            add the DDGap statistic to the plot (default: False)
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        max_n_cluster_tested = self.min_n_clusters + len(self.gaps_)
        plt.plot(np.arange(self.min_n_clusters, max_n_cluster_tested), self.gaps_, c="blue")
        plt.errorbar(np.arange(self.min_n_clusters, max_n_cluster_tested), self.gaps_, self.sks_,
                     capsize=3, linestyle='None', c="orange")
        if add_ddgap:
            if self.ddgaps_ is None:
                ddgaps = 2 * self.gaps_[1:-1] - self.gaps_[2:] - self.gaps_[:-2]
            else:
                ddgaps = self.ddgaps_
            plt.plot(np.arange(self.min_n_clusters + 1, max_n_cluster_tested - 1), ddgaps, c="green")
        plt.ylabel("Gap Statistic")
        plt.xlabel("n_clusters")
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of an input dataset. For this method the results from the fit() method will be used.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        predicted_labels : np.ndarray
            the predicted labels of the input data set
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        predicted_labels, _ = pairwise_distances_argmin_min(X=X, Y=self.cluster_centers_,
                                                          metric='euclidean',
                                                          metric_kwargs={'squared': True})
        predicted_labels = predicted_labels.astype(np.int32)
        return predicted_labels

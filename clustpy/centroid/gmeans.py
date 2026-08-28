"""
@authors:
Collin Leiber
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from clustpy.centroid.xmeans import _initial_kmeans_clusters, _execute_two_means
from scipy.stats import anderson
from clustpy.utils.checks import check_parameters
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


def _gmeans(X: np.ndarray, significance: float, n_clusters_init: int, max_n_clusters: int, n_split_trials: int,
            pval_strategy: str, n_boots: int, random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray):
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
    pval_strategy: str
        Strategy to calculate the p-value. 
        Can be 'equation' (using piecewise curve-fitting regression equations), 'bootstrap' (as described in the paper, 
        including monte-carlo simulation) or 'interpolate' (interpolation strategy from scipy)
    n_boots : int
        Number of bootstraps used to calculate the p-values. Only necessary if pval_strategy is 'original'
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
    if pval_strategy == "bootstrap":
        ad_simulation = _anderson_bootstraps(X.shape[0], n_boots, random_state)
    n_clusters, labels, centers, _ = _initial_kmeans_clusters(X, n_clusters_init, random_state)
    while n_clusters < max_n_clusters:
        n_clusters_old = n_clusters
        for c in range(n_clusters_old):
            ids_in_cluster = np.where(labels == c)[0]
            cluster_size = ids_in_cluster.shape[0]
            if cluster_size < 2:
                continue
            # Split cluster into two
            labels_split, centers_split, _ = _execute_two_means(X[ids_in_cluster], [np.arange(cluster_size)], 0,
                                                             np.array([centers[c]]), n_split_trials, random_state)
            # Project data form cluster onto resulting connection axis
            projection_vector = centers_split[0] - centers_split[1]
            vector_norm = np.linalg.norm(projection_vector)
            if vector_norm != 0:
                projection_vector /= vector_norm
            projected_data = np.dot(X[ids_in_cluster], projection_vector)
            # Use Anderson Darling to test if data is Gaussian (method includes standardization)
            ad_result = anderson(projected_data, "norm", method="interpolate")
            if pval_strategy == "interpolate":
                p_value = ad_result.pvalue
            elif pval_strategy == "equation":
                p_value = _anderson_darling_statistic_to_prob(ad_result.statistic, cluster_size)
            else:
                ad_statistic = _gmeans_ad_statistic(ad_result.statistic, cluster_size)
                # larger statistic -> less normal
                count_ge = np.sum(ad_simulation >= ad_statistic)
                p_value = count_ge / n_boots
            if p_value < significance:
                # If data is not Gaussian, keep the newly created cluster centers
                centers[c] = centers_split[0]
                centers = np.r_[centers, [centers_split[1]]]
                labels[ids_in_cluster[labels_split == 1]] = n_clusters
                n_clusters += 1
                # If maximum number of clusters is reached, stop iterating over the current clusters
                if n_clusters == max_n_clusters:
                    break
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


def _anderson_darling_statistic_to_prob(statistic: float, n_samples: int) -> float:
    """
    Transform the statistic returned by the Anderson Darling test into a p_value.
    First the adjusted statistic will be calculated.
    Afterwards, the actual p-value can be obtained.

    Parameters
    ----------
    statistic : float
        The original statistic from the Anderson Darling test.
    n_samples : int
        the number of samples

    Returns
    -------
    p_value : float
        The p-value

    References
    ----------
    D'Agostino, Ralph B., and Michael A. Stephens. "Goodness-of-fit techniques."
    Statistics: Textbooks and Monographs (1986).
    """
    adjusted_stat = statistic * (1 + (.75 / n_samples) + 2.25 / (n_samples ** 2))
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


def _gmeans_ad_statistic(statistic: float, n_samples: int) -> float:
    """
    Adjust the ad statistic as described in the GMeans paper.

    Parameters
    ----------
    statistic : float
        The original statistic from the Anderson Darling test.
    n_samples : int
        the number of samples

    Returns
    -------
    adjusted_stat : float
        the adjusted ad statistic
    """
    adjusted_stat = statistic * (1 + 4/n_samples - 25/(n_samples**2))
    return adjusted_stat


def _anderson_bootstraps(n_samples: int, n_boots: int, random_state: np.random.RandomState) -> list[float]:
    """
    Adjust the ad statistic as described in the GMeans paper.

    Parameters
    ----------
    n_samples : int
        the number of samples
    n_boots : int
        the number of bootstraps
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    gmeans_ads : list[float]
        the bootstraped ad statistic
    """
    samples = random_state.normal(size=(n_boots, n_samples))
    simulation_ads = [anderson(samples[i], "norm", method="interpolate").statistic for i in range(n_boots)]
    gmeans_ads = [_gmeans_ad_statistic(ad, n_samples) for ad in simulation_ads]
    return gmeans_ads


class GMeans(ClusterMixin, BaseEstimator):
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
    pval_strategy: str
        Strategy to calculate the p-value. 
        Can be 'equation' (using piecewise curve-fitting regression equations), 'bootstrap' (as described in the paper, 
        including monte-carlo simulation) or 'interpolate' (interpolation strategy from scipy) (default: 'equation')
    n_boots : int
        Number of bootstraps used to calculate the p-values. Only necessary if pval_strategy is 'original' (default: 1000)
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
    n_features_in_ : int
        the number of features used for the fitting

    References
    ----------
    Hamerly, Greg, and Charles Elkan. "Learning the k in k-means."
    Advances in neural information processing systems. 2004.

    and

    D'Agostino, Ralph B., and Michael A. Stephens. "Goodness-of-fit techniques."
    Statistics: Textbooks and Monographs (1986).
    """

    def __init__(self, significance: float = 0.0001, n_clusters_init: int = 1, max_n_clusters: int = np.inf,
                 n_split_trials: int = 10, pval_strategy: str = "equation", n_boots: int = 1000,
                 random_state: np.random.RandomState | int = None):
        self.significance = significance
        self.n_clusters_init = n_clusters_init
        self.max_n_clusters = max_n_clusters
        self.n_split_trials = n_split_trials
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.random_state = random_state

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
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        pval_strategy = self.pval_strategy.lower()
        assert pval_strategy in ["equation", "bootstrap", "interpolate"]
        n_clusters, labels, centers = _gmeans(X, self.significance, self.n_clusters_init, self.max_n_clusters,
                                              self.n_split_trials, pval_strategy, self.n_boots, random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.n_features_in_ = X.shape[1]
        return self

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

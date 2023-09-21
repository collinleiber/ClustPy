"""
@authors:
Collin Leiber
"""

import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _pgmeans(X, significance, n_projections, n_samples, n_new_centers, amount_random_centers, n_clusters_init,
             max_n_clusters, random_state):
    """
    Start the actual PGMeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the Kolmogorov Smirnov Test indicates a Gaussian Mixture Model
    n_projections : int
        Number of projection axes to test different projected GMMs on.
        Can be None, in that case it will be set to: -2.6198 * log(significance)
    n_samples : int
        Number of samples generated from the fitted GMM and used to execute the Kolmogorov Smirnov Test.
        If it is chosen larger than the number of data samples, it will be equal to this value.
        Can be None, in that case it will be set to: 3 / significance
    n_new_centers : int
        Nummber of centers to test when a new center should be added to the current GMM model.
        The additional center producing the best new GMM will be used for subsequent iterations
    amount_random_centers : float
        Amount of random centers tested. Must be a value in the range [0, 1].
        In total (n_new_centers * amount_random_centers) random centers will be tested.
        The other possible centers will be chosen based on the probability densities of the current GMM model
    n_clusters_init : int
        The initial number of clusters. Can also be of type np.ndarray if initial cluster centers are specified
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The final number of clusters,
        The labels as identified by PGMeans,
        The cluster centers as identified by PGMeans
    """
    assert max_n_clusters >= n_clusters_init, "max_n_clusters can not be smaller than n_clusters_init"
    assert significance >= 0 and significance <= 1, "significance must be a value in the range [0, 1]"
    assert amount_random_centers >= 0 and amount_random_centers <= 1, "amount_random_centers must be a value in the range [0, 1]"
    # Start parameters
    n_new_random_centers = int(n_new_centers * amount_random_centers)
    n_new_non_random_centers = n_new_centers - n_new_random_centers
    n_clusters, current_gmm = _initial_gmm_clusters(X, n_clusters_init, n_new_centers, random_state)
    while n_clusters <= max_n_clusters:
        gmm_matches = True
        for _ in range(n_projections):
            # Get random projection
            projection_vector = random_state.rand(X.shape[1])
            # Project data
            projected_X = np.matmul(X, projection_vector)
            # Project model - Alternative: Sample directly from model and project samples (should be slower)
            proj_gmm = _project_model(current_gmm, projection_vector, n_clusters, random_state)
            projected_samples, _ = proj_gmm.sample(n_samples)
            projected_samples = projected_samples.reshape(-1, )
            # Execute Kolmogorov-Smirnov test
            _, p_value = ks_2samp(projected_X, projected_samples)
            # Is hypothesis being rejected?
            if p_value < significance:
                gmm_matches = False
                break
        if gmm_matches:
            break
        else:
            # Add new center and update GMM
            n_clusters += 1
            current_gmm = _update_gmm_with_new_center(X, n_clusters, current_gmm, n_new_non_random_centers,
                                                      n_new_random_centers, random_state)
    # Get values from GMM
    labels = current_gmm.predict(X).astype(np.int32)
    centers = current_gmm.means_
    return n_clusters, labels, centers


def _project_model(gmm: GMM, projection_vector: np.ndarray, n_clusters: int,
                   random_state: np.random.RandomState) -> GMM:
    """
    Project the current Gaussian mixture (GMM) onto the specific projection axis.

    Parameters
    ----------
    gmm : GMM
        The current GMM
    projection_vector : np.ndarray
        The projection axis
    n_clusters : int
        The current number of clusters
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    proj_gmm : GMM
        The projected univariate GMM
    """
    # Project the model parameters
    proj_cov = np.array(
        [np.matmul(projection_vector.T, np.matmul(cov, projection_vector)) for cov in gmm.covariances_]).reshape(
        (-1, 1, 1))
    proj_mean = np.matmul(gmm.means_, projection_vector).reshape((-1, 1))
    # Create new 1d GMM
    proj_gmm = GMM(n_components=n_clusters, random_state=random_state)
    proj_gmm.covariances_ = proj_cov
    proj_gmm.means_ = proj_mean
    proj_gmm.weights_ = gmm.weights_
    return proj_gmm


def _update_gmm_with_new_center(X: np.ndarray, n_clusters: int, current_gmm: GMM, n_new_non_random_centers: int,
                                n_new_random_centers: int, random_state: np.random.RandomState) -> GMM:
    """
    Update the current GMM by adding a new center.
    To receive a high-quality model multiple additional centers will be tested.
    These can be a random sample from the data set or samples with a low probability density within the current GMM.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        The updated number of clusters (number of clusters in current_gmm + 1)
    current_gmm : GMM
        The current GMM
    n_new_non_random_centers : int
        Number of non-random centers that should be tested
    n_new_random_centers : int
        Number of random centers that should be tested
    random_state : np.ndarray
        use a fixed random state to get a repeatable solution

    Returns
    -------
    best_gmm : GMM
        The updated GMM with an additional center added
    """
    best_gmm = None
    best_log_likelihood = np.inf
    if n_new_non_random_centers > 0:
        # Non-random centers are chosen through lowest probability regarding current GMM
        max_probability_densities = np.max(current_gmm.predict_proba(X), axis=1)
        # Get minimum max probabilities
        possible_non_random_samples = np.argsort(max_probability_densities)
    for c in range(n_new_non_random_centers + n_new_random_centers):
        if c < n_new_non_random_centers:
            # Add non random centers
            new_center = X[possible_non_random_samples[c]]
        else:
            # Add random centers
            new_center = X[random_state.choice(np.arange(X.shape[0]))]
        new_gmm = GMM(n_components=n_clusters, n_init=1, means_init=np.r_[current_gmm.means_, [new_center]],
                      random_state=random_state)
        new_gmm.fit(X)
        # Check error of new GMM
        if new_gmm.lower_bound_ < best_log_likelihood:
            best_log_likelihood = new_gmm.lower_bound_
            best_gmm = new_gmm
    return best_gmm


def _initial_gmm_clusters(X: np.ndarray, n_clusters_init: int, gmm_repetitions: int,
                          random_state: np.random.RandomState) -> (int, GMM):
    """
    Get the initial Gaussian Mixture Model based on the n_clusters_init parameter.
    If n_clusters_init is an integer, the cluster parameters are identified by a GMM with init_n_clusters als single input.
    If n_clusters_init is of type np.ndarray, the cluster parameters are identified by a GMM with the initial cluster centers given by n_clusters_init.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified
    gmm_repetitions : int
        Number of repetitions for the initial GMM
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, GMM)
        The initial number of clusters,
        The initial GMM
    """
    if type(n_clusters_init) is int and n_clusters_init == 1:
        # Convert n_cluster_init to initial cluster center. GMM will be created below
        n_clusters_init = np.mean(X, axis=0).reshape(1, -1)
    # Create initial GMM
    if type(n_clusters_init) is int:
        # Normally, init_n_clusters is int
        n_clusters = n_clusters_init
        initial_gmm = GMM(n_components=n_clusters, n_init=gmm_repetitions, random_state=random_state)
        initial_gmm.fit(X)
    else:
        # If init_n_clusters is array, this should be equal to the initial cluster centers
        n_clusters = n_clusters_init.shape[0]
        initial_gmm = GMM(n_components=n_clusters, means_init=n_clusters_init, n_init=1, random_state=random_state)
        initial_gmm.fit(X)
    return n_clusters, initial_gmm


class PGMeans(BaseEstimator, ClusterMixin):
    """
    Execute the PGMeans clustering procedure.
    Determines the number of clusters by executing the EM algorithm multiple times to create different Gaussian Mixtures (GMMs).
    For each GMM it projects the GMM model onto random projection axes and uses the Kolmogorov Smirnov Test to decide whether the data matches the fitted model.
    If this is not the case, a new center will be added and the next GMM will be fitted.
    This is repeated until no cluster are added anymore.

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the Kolmogorov Smirnov Test indicates a Gaussian Mixture Model (default: 0.001)
    n_projections : int
        Number of projection axes to test different projected GMMs on.
        Can be None, in that case it will be set to: -2.6198 * log(significance) (default: None)
    n_samples : int
        Number of samples generated from the fitted GMM and used to execute the Kolmogorov Smirnov Test.
        If it is chosen larger than the number of data samples, it will be equal to this value.
        Can be None, in that case it will be set to: 3 / significance (default: None)
    n_new_centers : int
        Nummber of centers to test when a new center should be added to the current GMM model.
        The additional center producing the best new GMM will be used for subsequent iterations (default: 10)
    amount_random_centers : float
        Amount of random centers tested. Must be a value in the range [0, 1].
        In total (n_new_centers * amount_random_centers) random centers will be tested.
        The other possible centers will be chosen based on the probability densities of the current GMM modal (default: 0.5)
    n_clusters_init : int
        The initial number of clusters. Can also by of type np.ndarray if initial cluster centers are specified (default: 1)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init (default: np.inf)
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
    Feng, Yu, and Greg Hamerly. "PG-means: learning the number of clusters in data."
    Advances in neural information processing systems. 2007.
    """

    def __init__(self, significance: float = 0.001, n_projections: int = None, n_samples: int = None,
                 n_new_centers: int = 10, amount_random_centers: float = 0.5, n_clusters_init: int = 1,
                 max_n_clusters: int = np.inf, random_state: np.random.RandomState = None):
        self.significance = significance
        if n_projections is None:
            n_projections = int(-2.6198 * np.log(significance)) + 1
        self.n_projections = n_projections
        if n_samples is None:
            n_samples = int(3 / self.significance) + 1
        self.n_samples = n_samples
        self.n_new_centers = n_new_centers
        self.n_clusters_init = n_clusters_init
        self.max_n_clusters = max_n_clusters
        self.amount_random_centers = amount_random_centers
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'PGMeans':
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
        self : PGMeans
            this instance of the PGMeans algorithm
        """
        self.n_samples = min(self.n_samples, X.shape[0])
        n_clusters, labels, centers = _pgmeans(X, self.significance, self.n_projections, self.n_samples,
                                               self.n_new_centers, self.amount_random_centers, self.n_clusters_init,
                                               self.max_n_clusters, self.random_state)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

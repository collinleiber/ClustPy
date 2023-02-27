from clustpy.utils._information_theory import bic_costs, mdl_costs_probability, \
    integer_costs, mdl_costs_gmm_multiple_covariances, mdl_costs_gmm_common_covariance, _mdl_costs_gaussian, \
    mdl_costs_gaussian_diagonal_covariance, mdl_costs_gaussian_full_covariance, mdl_costs_gaussian_spherical_covariance
from sklearn.mixture import GaussianMixture as GMM
import numpy as np

"""
Helpers
"""


def _create_Gaussian_mixture(n_clusters):
    rs = np.random.RandomState(1)
    n_points = 100
    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.1, 0.2],
                    [0.1, 1.5, 0.5],
                    [0.2, 0.5, 3]])
    X = rs.multivariate_normal(mean, cov, n_points)
    L = np.array([0] * n_points)
    if n_clusters > 1:
        n_points = 120
        mean = np.array([3, 4, 5])
        cov = np.array([[1.5, 0.5, 0.1],
                        [0.5, 1.5, 0.3],
                        [0.1, 0.3, 1]])
        X = np.r_[X, rs.multivariate_normal(mean, cov, n_points)]
        L = np.r_[L, [1] * n_points]
    if n_clusters > 2:
        n_points = 180
        mean = np.array([9, 9, 7])
        cov = np.array([[1, 0, 0],
                        [0, 0.8, 0],
                        [0, 0, 1.2]])
        X = np.r_[X, rs.multivariate_normal(mean, cov, n_points)]
        L = np.r_[L, [2] * n_points]
    if n_clusters > 3:
        raise Exception("n_clusters can not be larger than 3")
    return X, L


def _real_costs_of_single_gauss(X, means, covariance_type):
    gmm = GMM(n_components=1, covariance_type=covariance_type, means_init=[means])
    gmm.fit(X)
    # Convert variances to covariance matrices
    if covariance_type == "spherical":
        cov = np.identity(X.shape[1]) * gmm.covariances_[0]
    elif covariance_type == "diag":
        cov = np.identity(X.shape[1]) * gmm.covariances_[0]
    else:
        cov = gmm.covariances_[0]
    # Get probabilities of samples
    dist_to_center = X - gmm.means_[0]
    probabilities = [np.exp(
        -0.5 * np.matmul(np.matmul(x.T, np.linalg.inv(cov)), x)) / np.sqrt(
        (2 * np.pi) ** X.shape[1] * np.linalg.det(cov)) for x in dist_to_center]
    mdl_costs = -np.sum(np.log2(probabilities))
    return mdl_costs


def _real_costs_of_multiple_gauss(X, means, n_clusters, L):
    gmm = GMM(n_components=n_clusters, covariance_type="tied", means_init=means)
    gmm.fit(X)
    # If covariance_type is 'tied' we only have one covariance matrix
    cov = gmm.covariances_
    # Get probabilities of samples
    mdl_costs = 0
    for i in range(n_clusters):
        dist_to_center = X[L == i] - gmm.means_[i]
        probabilities = [np.exp(
            -0.5 * np.matmul(np.matmul(x.T, np.linalg.inv(cov)), x)) / np.sqrt(
            (2 * np.pi) ** X.shape[1] * np.linalg.det(cov)) for x in dist_to_center]
        mdl_costs += -np.sum(np.log2(probabilities))
    return mdl_costs


"""
Tests
"""


def test_bic_costs():
    # Use log2
    costs = bic_costs(3, True)
    assert costs == 0.792481250360578
    # Use ln
    costs = bic_costs(3, False)
    assert costs == 0.5493061443340549


def test_integer_costs():
    costs = integer_costs(77)
    assert abs(costs - 12.328150766) < 1e-9


def test_mdl_costs_probability():
    costs = mdl_costs_probability(1 / 33)
    assert abs(costs - 5.044394119) < 1e-9


def test_mdl_costs_gmm_multiple_covariances():
    n_clusters = 3
    X, L = _create_Gaussian_mixture(n_clusters)
    # Get scatter matrices and centers
    means = np.array([np.mean(X[L == i], axis=0) for i in range(n_clusters)])
    scatters = np.zeros((n_clusters, X.shape[1], X.shape[1]))
    for i in range(n_clusters):
        dist_to_mean = X[L == i] - means[i]
        scatters[i] = np.matmul(dist_to_mean.T, dist_to_mean)
    # Single variance
    mdl_real = np.sum([_real_costs_of_single_gauss(X[L == i], means[i], "spherical") for i in range(n_clusters)])
    mdl_our = mdl_costs_gmm_multiple_covariances(X.shape[1], scatters, L, None, "spherical")
    assert abs(mdl_real - mdl_our) < 1e-9
    # Diagonal
    mdl_real = np.sum([_real_costs_of_single_gauss(X[L == i], means[i], "diag") for i in range(n_clusters)])
    mdl_our = mdl_costs_gmm_multiple_covariances(X.shape[1], scatters, L, None, "diag")
    assert abs(mdl_real - mdl_our) < 1e-9
    # Full
    mdl_real = np.sum([_real_costs_of_single_gauss(X[L == i], means[i], "full") for i in range(n_clusters)])
    mdl_our = mdl_costs_gmm_multiple_covariances(X.shape[1], scatters, L, None, "full")
    assert abs(mdl_real - mdl_our) < 1e-9


def test_mdl_costs_gmm_common_covariance():
    n_clusters = 3
    X, L = _create_Gaussian_mixture(n_clusters)
    # Get scatter matrices and centers
    means = np.array([np.mean(X[L == i], axis=0) for i in range(n_clusters)])
    scatters = np.zeros((n_clusters, X.shape[1], X.shape[1]))
    for i in range(n_clusters):
        dist_to_mean = X[L == i] - means[i]
        scatters[i] = np.matmul(dist_to_mean.T, dist_to_mean)
    # Full
    mdl_real = _real_costs_of_multiple_gauss(X, means, n_clusters, L)
    mdl_our = mdl_costs_gmm_common_covariance(X.shape[1], scatters, X.shape[0], None, "full")
    mdl_our2 = mdl_costs_gmm_common_covariance(X.shape[1], np.sum(scatters, 0), X.shape[0], None, "full")
    assert mdl_our == mdl_our2
    assert abs(mdl_real - mdl_our) < 0.5  # Error is larger because GMM covariances are a little different


def test_mdl_costs_gaussian():
    n_clusters = 1
    X, L = _create_Gaussian_mixture(n_clusters)
    # Get scatter matrix and center
    mean = np.mean(X, axis=0)
    dist_to_mean = X - mean
    scatter = np.matmul(dist_to_mean.T, dist_to_mean)
    # Single variance
    mdl_real = _real_costs_of_single_gauss(X, mean, "spherical")
    mdl_our = _mdl_costs_gaussian(X.shape[1], scatter, X.shape[0], None, "spherical")
    mdl_our2 = mdl_costs_gaussian_spherical_covariance(X.shape[1], scatter, X.shape[0], None)
    assert abs(mdl_real - mdl_our) < 1e-9
    assert mdl_our == mdl_our2
    # Diagonal
    mdl_real = _real_costs_of_single_gauss(X, mean, "diag")
    mdl_our = _mdl_costs_gaussian(X.shape[1], scatter, X.shape[0], None, "diag")
    mdl_our2 = mdl_costs_gaussian_diagonal_covariance(X.shape[1], scatter, X.shape[0], None)
    assert abs(mdl_real - mdl_our) < 1e-9
    assert mdl_our == mdl_our2
    # Full
    mdl_real = _real_costs_of_single_gauss(X, mean, "full")
    mdl_our = _mdl_costs_gaussian(X.shape[1], scatter, X.shape[0], None, "full")
    mdl_our2 = mdl_costs_gaussian_full_covariance(X.shape[1], scatter, X.shape[0], None)
    assert abs(mdl_real - mdl_our) < 1e-9
    assert mdl_our == mdl_our2

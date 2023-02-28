import numpy as np

"""
Often used constants
"""
_LOG_2_PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)


def bic_costs(n_points: int, use_log2: bool = False) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC) costs for parameters.
    Is equal to: 1/2 * log(n_points)

    Parameters
    ----------
    n_points : int
        Number of samples
    use_log2 : bool
        Defines whether log2 should be used instead of ln (default: False)

    Returns
    -------
    costs : float
        The BIC costs
    """
    assert n_points > 1, "The number of points must be larger than 0 to calculate the BIC costs. Your input:\n{0}".format(
        n_points)
    if use_log2:
        bic_costs = 0.5 * np.log2(n_points)
    else:
        bic_costs = 0.5 * np.log(n_points)
    return bic_costs


def integer_costs(integer: int) -> float:
    """
    Calculate the costs to encode an integer value. Uses following formula:
    log(integer) + log(log(integer)) + log(log(log(integer))) + ... + log(const), where const = 2.865064.

    Parameters
    ----------
    integer : int
        The integer value to encode

    Returns
    -------
    costs : float
        The encoding costs of the integer
    """
    assert type(integer) is int or type(integer) is np.int32 or type(
        integer) is np.int64, "The input to calculate the mdl costs of must be an integer. Your input:\n{0} (type: {1})".format(
        integer, type(integer))
    costs = 0
    if integer != 0:
        last_interim_result = np.log2(integer)
        while last_interim_result > 0:
            costs += last_interim_result
            last_interim_result = np.log2(last_interim_result)
    costs = costs + np.log2(2.865064)
    return costs


"""
========== MDL ==========
=========================

Discrete Distribution
"""


def mdl_costs_probability(probability: float) -> float:
    """
    Encode an object using a probability and MDL.

    Parameters
    ----------
    probability : float
        The probability

    Returns
    -------
    costs : float
        The encoding costs
    """
    assert probability >= 0 and probability <= 1, "probability must be a value in the range [0, 1]"
    if probability == 0:
        return 0
    return -np.log2(probability)


"""
Gaussian Mixture Models
"""


def mdl_costs_gmm_multiple_covariances(n_dims: int, scatter_matrices: np.ndarray, labels: np.ndarray,
                                       rotation: np.ndarray = None, covariance_type: str = "spherical") -> float:
    """
    Calculate the coding costs of all points within a Gaussian Mixture Model (GMM).
    Calculates log(pdf(X)) with help of the scatter matrices of the clusters.
    Method calls _mdl_costs_gaussian for each cluster.

    Parameters
    ----------
    n_dims : int
        Number of features
    scatter_matrices : np.ndarray
        Array containing the scatter matrix of each cluster
    labels : np.ndarray
        The cluster labels
    rotation : np.ndarray
        An optional rotation matrix for the feature space (default: None)
    covariance_type : str
        The type of covariance matrix.
        Can be 'spherical' (same variance for each feature, no covariances), 'diag' (different variance for each feature, no covariances) or 'full' (different variance for each feature, includes covariances) (default: 'spherical')

    Returns
    -------
    full_pdf_costs : float
        The encoding costs for all points of the GMM
    """
    assert covariance_type in ["spherical", "diag",
                               "full"], "covariance_type must equal 'spherical', 'diag' or 'full'"
    if n_dims == 0:
        return 0
    full_pdf_costs = 0
    # Get costs for each cluster
    for cluster_index, scatter_matrix_cluster in enumerate(scatter_matrices):
        # Get number of points in this cluster
        n_points_in_cluster = np.sum(labels == cluster_index)
        full_pdf_costs += _mdl_costs_gaussian(n_dims, scatter_matrix_cluster, n_points_in_cluster, rotation,
                                              covariance_type)
    # Return pdf_costs
    return full_pdf_costs


def mdl_costs_gmm_common_covariance(n_dims: int, scatter_matrix: np.ndarray, n_points: int,
                                    rotation: np.ndarray = None, covariance_type: str = "spherical") -> float:
    """
    Calculate the coding costs of all points within a Gaussian Mixture Model (GMM).
    In this special case all Gaussians within the GMM share a common covariance matrix.
    Calculates log(pdf(X)) with help of the scatter matrices of the clusters.
    Method calls _mdl_costs_gaussian for each cluster.


    Parameters
    ----------
    n_dims : int
        Number of features
    scatter_matrix : np.ndarray
        Either a single scatter matrix or an array containing the scatter matrix of each cluster
    n_points : int
        The number of samples in all clusters
    rotation : np.ndarray
        An optional rotation matrix for the feature space (default: None)
    covariance_type : str
        The type of covariance matrix.
        Can be 'spherical' (same variance for each feature, no covariances), 'diag' (different variance for each feature, no covariances) or 'full' (different variance for each feature, includes covariances) (default: 'spherical')

    Returns
    -------
    full_pdf_costs : float
        The encoding costs for all points of the GMM
    """
    assert covariance_type in ["spherical", "diag",
                               "full"], "covariance_type must equal 'spherical', 'diag' or 'full'"
    if n_dims == 0:
        return 0
    # Get costs for each cluster
    if scatter_matrix.ndim == 3:
        scatter_matrix = np.sum(scatter_matrix, 0)
    # Get number of points which are not outliers
    full_pdf_costs = _mdl_costs_gaussian(n_dims, scatter_matrix, n_points, rotation, covariance_type)
    # Return pdf_costs
    return full_pdf_costs


"""
Single Gaussian
"""


def _mdl_costs_gaussian(n_dims: int, scatter_matrix_cluster: np.ndarray, n_points_in_cluster: int,
                        rotation: np.ndarray = None, covariance_type: str = "spherical") -> float:
    """
    Calculate the coding costs of all points within a single Gaussian cluster.
    Calculates log(pdf(X)) with help of the scatter matrix of the cluster.

    Parameters
    ----------
    n_dims : int
        Number of features
    scatter_matrix_cluster : np.ndarray
        The scatter matrix of the cluster
    n_points_in_cluster : int
        The number of samples in the cluster
    rotation : np.ndarray
        An optional rotation matrix for the feature space (default: None)
    covariance_type : str
        The type of covariance matrix.
        Can be 'spherical' (same variance for each feature, no covariances), 'diag' (different variance for each feature, no covariances) or 'full' (different variance for each feature, includes covariances) (default: 'spherical')

    Returns
    -------
    pdf_costs : float
        The encoding costs for all points in the cluster
    """
    if covariance_type == "spherical":
        pdf_costs = mdl_costs_gaussian_spherical_covariance(n_dims, scatter_matrix_cluster,
                                                            n_points_in_cluster, rotation)
    elif covariance_type == "diag":
        pdf_costs = mdl_costs_gaussian_diagonal_covariance(n_dims, scatter_matrix_cluster,
                                                           n_points_in_cluster, rotation)
    elif covariance_type == "full":
        pdf_costs = mdl_costs_gaussian_full_covariance(n_dims, scatter_matrix_cluster,
                                                       n_points_in_cluster, rotation)
    else:
        raise Exception("covariance_type must be 'single', 'diagonal' or 'full'")
    return pdf_costs


def mdl_costs_gaussian_spherical_covariance(n_dims: int, scatter_matrix_cluster: np.ndarray, n_points_in_cluster: int,
                                            rotation: np.ndarray = None) -> float:
    """
    Calculate the coding costs of all points within a single Gaussian cluster.
    This Gaussian has the same variance for each feature and no covariances.
    Calculates log(pdf(X)) with help of the scatter matrix of the cluster.

    Parameters
    ----------
    n_dims : int
        Number of features
    scatter_matrix_cluster : np.ndarray
        The scatter matrix of the cluster
    n_points_in_cluster : int
        The number of samples in the cluster
    rotation : np.ndarray
        An optional rotation matrix for the feature space (default: None)

    Returns
    -------
    pdf_costs : float
        The encoding costs for all points in the cluster
    """
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or n_dims == 0:
        return 0
    if rotation is None:
        rotation = np.identity(scatter_matrix_cluster.shape[0])
    # Calculate the actual costs
    trace = np.trace(np.matmul(np.matmul(rotation.transpose(), scatter_matrix_cluster), rotation))
    assert trace >= -1e-15, "Trace can not be negative! Trace is {0}".format(trace)
    # Can occur if all points in this cluster lie on the same position
    if trace <= 1e-15:
        return 0
    pdf_costs = 1 + _LOG_2_PI - np.log(n_dims * n_points_in_cluster) + np.log(trace)
    pdf_costs *= n_points_in_cluster * n_dims / (2 * _LOG_2)
    return pdf_costs


def mdl_costs_gaussian_diagonal_covariance(n_dims: int, scatter_matrix_cluster: np.ndarray, n_points_in_cluster: int,
                                           rotation: np.ndarray = None) -> float:
    """
    Calculate the coding costs of all points within a single Gaussian cluster.
    This Gaussian has a different variance for each feature and no covariances.
    Calculates log(pdf(X)) with help of the scatter matrix of the cluster.

    Parameters
    ----------
    n_dims : int
        Number of features
    scatter_matrix_cluster : np.ndarray
        The scatter matrix of the cluster
    n_points_in_cluster : int
        The number of samples in the cluster
    rotation : np.ndarray
        An optional rotation matrix for the feature space (default: None)

    Returns
    -------
    pdf_costs : float
        The encoding costs for all points in the cluster
    """
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or n_dims == 0:
        return 0
    if rotation is None:
        rotation = np.identity(scatter_matrix_cluster.shape[0])
    cropped_scatter = np.matmul(np.matmul(rotation.transpose(), scatter_matrix_cluster), rotation)
    pdf_costs = n_dims + n_dims * _LOG_2_PI - n_dims * np.log(n_points_in_cluster) + np.sum(
        np.log(cropped_scatter.diagonal()))
    pdf_costs *= n_points_in_cluster / (2 * _LOG_2)
    return pdf_costs


def mdl_costs_gaussian_full_covariance(n_dims: int, scatter_matrix_cluster: np.ndarray, n_points_in_cluster: int,
                                       rotation: np.ndarray = None) -> float:
    """
    Calculate the coding costs of all points within a single Gaussian cluster.
    This Gaussian has a different variance for each feature and includes all covariances.
    Calculates log(pdf(X)) with help of the scatter matrix of the cluster.

    Parameters
    ----------
    n_dims : int
        Number of features
    scatter_matrix_cluster : np.ndarray
        The scatter matrix of the cluster
    n_points_in_cluster : int
        The number of samples in the cluster
    rotation : np.ndarray
        An optional rotation matrix for the feature space (default: None)

    Returns
    -------
    pdf_costs : float
        The encoding costs for all points in the cluster
    """
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or n_dims == 0:
        return 0
    if rotation is None:
        rotation = np.identity(scatter_matrix_cluster.shape[0])
    cropped_scatter = np.matmul(np.matmul(rotation.transpose(), scatter_matrix_cluster), rotation)
    inv = np.linalg.inv(cropped_scatter)
    det = np.linalg.det(cropped_scatter)
    if det <= 0:
        det = 1e-20
    # calculate sum
    pdf_costs = 0
    for row, _ in enumerate(cropped_scatter):
        for col in range(0, row + 1):
            if row == col:
                pdf_costs += cropped_scatter[row, col] * inv[row, col]
            else:
                pdf_costs += 2 * cropped_scatter[row, col] * inv[row, col]
    pdf_costs += n_dims * _LOG_2_PI - n_dims * np.log(n_points_in_cluster) + np.log(det)
    pdf_costs *= n_points_in_cluster / (2 * _LOG_2)
    return pdf_costs

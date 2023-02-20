import numpy as np

"""
Basic MDL Methods
"""

_LOG_2_PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)


def mdl_costs_orthogonal_matrix(dimensionality, costs_for_element):
    cost = dimensionality * (dimensionality - 1) / 2
    cost *= costs_for_element
    return cost


def mdl_costs_bic(n_points):
    assert n_points > 1, "The number of points must be larger than 0 to calculate the mdl costs of the precision. Your input:\n{0}".format(
        n_points)
    return 0.5 * np.log2(n_points)


def mdl_costs_integer_value(integer):
    """
    Calculate the mdl costs to encode an integer value. Uses following formula:
    log(integer) + log(log(integer)) + log(log(log(integer))) + ... + log(const),
    where const = 2.865064.
    If log(log(...)) is smaller than 0 operation will be aborted.
    :param integer: input integer value
    :return: the mdl coding costs of the integer
    """
    assert type(integer) is int or type(
        integer) is np.int32 or type(
        integer) is np.int64, "The input to calculate the mdl costs of must be an integer. Your input:\n{0} (type: {1})".format(
        integer, type(integer))
    cost = 0
    if integer != 0:
        last_interim_result = np.log2(integer)
        while last_interim_result > 0:
            cost += last_interim_result
            last_interim_result = np.log2(last_interim_result)
    const = np.log2(2.865064)
    return cost + const


"""
Uniform Distribution
"""


def mdl_costs_discrete_probability(probability):
    if probability == 0:
        return 0
    return -np.log2(probability)


"""
Gaussian Mixture Models
"""


def mdl_costs_gmm_multiple_covariances(dimensionality, scatter_matrices, labels, rotation=None,
                                       covariance_type="single"):
    """
    Calculate the coding costs of all points within a subspace_nr. Calculates log(pdf(X)) with help of the scatter matrices
    of the clusters.
    Method calls calculate_single_cluster_costs for each cluster
    :param rotation: cropped orthogonal rotation matrix
    :param dimensionality:
    :param scatter_matrices: scatter matrices of the subspace_nr
    :param labels: labels of the cluster assignments for the subspace_nr. -1 equals outlier
    :return: coding costs for all points for this subspace_nr
    """
    assert covariance_type in ["single", "diagonal",
                               "full"], "covariance_type must equal 'single', 'diagonal' or 'full'"
    if dimensionality == 0:
        return 0
    full_pdf_costs = 0
    # Get costs for each cluster
    for cluster_index, scatter_matrix_cluster in enumerate(scatter_matrices):
        # Get number of points in this cluster
        n_points_in_cluster = len(labels[labels == cluster_index])
        full_pdf_costs += _mdl_costs_gaussian(dimensionality, scatter_matrix_cluster, n_points_in_cluster, rotation,
                                              covariance_type)
    # Return pdf_costs
    return full_pdf_costs


def mdl_costs_gmm_single_covariance(dimensionality, scatter_matrices, n_points, rotation=None,
                                    covariance_type="single"):
    assert covariance_type in ["single", "diagonal",
                               "full"], "covariance_type must equal 'single', 'diagonal' or 'full'"
    if dimensionality == 0:
        return 0
    # Get costs for each cluster
    sum_scatter_matrices = np.sum(scatter_matrices, 0)
    # Get number of points which are not outliers
    full_pdf_costs = _mdl_costs_gaussian(dimensionality, sum_scatter_matrices, n_points, rotation, covariance_type)
    # Return pdf_costs
    return full_pdf_costs


"""
Single Gaussian
"""


def _mdl_costs_gaussian(dimensionality, scatter_matrix_cluster, n_points_in_cluster, rotation=None,
                        covariance_type="single"):
    if covariance_type == "single":
        full_pdf_costs = mdl_costs_gaussian_single_variance(dimensionality, scatter_matrix_cluster,
                                                            n_points_in_cluster, rotation)
    elif covariance_type == "diagonal":
        full_pdf_costs = mdl_costs_gaussian_diagonal_covariance(dimensionality, scatter_matrix_cluster,
                                                                n_points_in_cluster, rotation)
    elif covariance_type == "full":
        full_pdf_costs = mdl_costs_gaussian_full_covariance(dimensionality, scatter_matrix_cluster,
                                                            n_points_in_cluster, rotation)
    else:
        raise Exception("covariance_type must be 'single', 'diagonal' or 'full'")
    return full_pdf_costs


def mdl_costs_gaussian_single_variance(dimensionality, scatter_matrix_cluster, n_points_in_cluster, rotation=None):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or dimensionality == 0:
        return 0
    if rotation is None:
        rotation = np.identity(scatter_matrix_cluster.shape[0])
    # Calculate the actual costs
    trace = np.trace(np.matmul(np.matmul(rotation.transpose(), scatter_matrix_cluster), rotation))
    assert trace >= -1e-20, "Trace can not be negative! Trace is {0}".format(trace)
    # Can occur if all points in this cluster lie on the same position
    if trace <= 1e-20:
        return 0
    pdf_costs = 1 + _LOG_2_PI - np.log(dimensionality * n_points_in_cluster) + np.log(trace)
    pdf_costs *= n_points_in_cluster * dimensionality / (2 * _LOG_2)
    return pdf_costs


def mdl_costs_gaussian_diagonal_covariance(dimensionality, scatter_matrix_cluster, n_points_in_cluster, rotation=None):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or dimensionality == 0:
        return 0
    if rotation is None:
        rotation = np.identity(scatter_matrix_cluster.shape[0])
    cropped_scatter = np.matmul(np.matmul(rotation.transpose(), scatter_matrix_cluster), rotation)
    pdf_costs = dimensionality + dimensionality * _LOG_2_PI - dimensionality * np.log(n_points_in_cluster) + np.sum(
        np.log(cropped_scatter.diagonal()))
    pdf_costs *= n_points_in_cluster / (2 * _LOG_2)
    return pdf_costs


def mdl_costs_gaussian_full_covariance(dimensionality, scatter_matrix_cluster, n_points_in_cluster, rotation=None):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or dimensionality == 0:
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
    pdf_costs += dimensionality * _LOG_2_PI - dimensionality * np.log(n_points_in_cluster) + np.log(det)
    pdf_costs *= n_points_in_cluster / (2 * _LOG_2)
    return pdf_costs

import numpy as np

"""
Basic MDL Methods
"""
LOG_2_PI = np.log(2 * np.pi)
LOG_2 = np.log(2)


def mdl_costs_orthogonal_matrix(dimensionality, costs_for_element):
    cost = dimensionality * (dimensionality - 1) / 2
    cost *= costs_for_element
    return cost


def mdl_costs_float_value(n_points):
    assert n_points > 1, "The number of points must be larger than 0 to calculate the mdl costs of a float. Your input:\n{0}".format(
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
        integer) is np.int32, "The input to calculate the mdl costs of must be an integer. Your input:\n{0}".format(
        integer)
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


def mdl_costs_uniform_distribution(possibilities):
    if possibilities == 0:
        return 0
    return -np.log2(1 / possibilities)


"""
Gaussian Mixtrure Models
"""


def mdl_costs_gmm_multiple_covariances(cropped_V, m_subspace, scatter_matrices_subspace, labels_subspace, covariance_type="single"):
    """
    Calculate the coding costs of all points within a subspace. Calculates log(pdf(X)) with help of the scatter matrices
    of the clusters.
    Method calls calculate_single_cluster_costs for each cluster
    :param cropped_V: cropped orthogonal rotation matrix
    :param m_subspace:
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :param labels_subspace: labels of the cluster assingments for the subspace. -1 equals outlier
    :return: coding costs for all points for this subspace
    """
    assert covariance_type in ["single", "diagonal", "full"], "covariance_type must equal 'single', 'diagonal' or 'full'"
    if m_subspace == 0:
        return 0
    full_pdf_costs = 0
    # Get costs for each cluster
    for cluster_index, scatter_matrix_cluster in enumerate(scatter_matrices_subspace):
        # Get number of points in this cluster
        n_points_in_cluster = len(labels_subspace[labels_subspace == cluster_index])
        if covariance_type == "single":
            full_pdf_costs += mdl_costs_gaussian_single_variance(cropped_V, n_points_in_cluster,
                                                                 scatter_matrix_cluster, m_subspace)
        elif covariance_type == "diagonal":
            full_pdf_costs += mdl_costs_gaussian_diagonal_covariance(cropped_V, n_points_in_cluster,
                                                                 scatter_matrix_cluster, m_subspace)
        elif covariance_type == "full":
            full_pdf_costs += mdl_costs_gaussian_full_covariance(cropped_V, n_points_in_cluster,
                                                                     scatter_matrix_cluster, m_subspace)
    # Return pdf_costs
    return full_pdf_costs


def mdl_costs_gmm_single_covariance(cropped_V, m_subspace, scatter_matrices_subspace, n_points_non_outliers, covariance_type="single"):
    assert covariance_type in ["single", "diagonal", "full"], "covariance_type must equal 'single', 'diagonal' or 'full'"
    if m_subspace == 0:
        return 0
    # Get costs for each cluster
    sum_scatter_matrices = np.sum(scatter_matrices_subspace, 0)
    # Get number of points which are not outliers
    if covariance_type == "single":
        full_pdf_costs = mdl_costs_gaussian_single_variance(cropped_V, n_points_non_outliers,
                                                        sum_scatter_matrices, m_subspace)
    elif covariance_type == "diagonal":
        full_pdf_costs = mdl_costs_gaussian_diagonal_covariance(cropped_V, n_points_non_outliers,
                                                        sum_scatter_matrices, m_subspace)
    elif covariance_type == "full":
        full_pdf_costs = mdl_costs_gaussian_full_covariance(cropped_V, n_points_non_outliers,
                                                        sum_scatter_matrices, m_subspace)
    # Return pdf_costs
    return full_pdf_costs


"""
Single Gaussians
"""


def mdl_costs_gaussian_single_variance(cropped_V, n_points_in_cluster, scatter_matrix_cluster, m_subspace):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or m_subspace == 0:
        return 0
    # Calculate the actual costs
    trace = np.trace(np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix_cluster), cropped_V))
    assert trace >= -1e-20, "Trace can not be negative!"
    # Can occur if all points in this cluster lie on the same position
    if trace <= 1e-20:
        return 0
    pdf_costs = 1 + LOG_2_PI + np.log(trace / m_subspace / n_points_in_cluster)
    pdf_costs *= n_points_in_cluster * m_subspace / 2 / LOG_2
    return pdf_costs


def mdl_costs_gaussian_diagonal_covariance(cropped_V, n_points_in_cluster, scatter_matrix_cluster,
                                           m_subspace):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or m_subspace == 0:
        return 0
    cropped_scatter = np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix_cluster), cropped_V)
    pdf_costs = m_subspace + m_subspace * LOG_2_PI - m_subspace * np.log(n_points_in_cluster) + np.sum(
        np.log(cropped_scatter.diagonal()))
    pdf_costs *= n_points_in_cluster / 2 / LOG_2
    return pdf_costs


def mdl_costs_gaussian_full_covariance(cropped_V, n_points_in_cluster, scatter_matrix_cluster, m_subspace):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or m_subspace == 0:
        return 0
    cropped_scatter = np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix_cluster), cropped_V)
    inv = np.linalg.inv(cropped_scatter)
    det = np.linalg.det(cropped_scatter)
    # calculate sum
    pdf_costs = 0
    for row, _ in enumerate(cropped_scatter):
        for col in range(0, row + 1):
            if row == col:
                pdf_costs += cropped_scatter[row, col] * inv[row, col]
            else:
                pdf_costs += 2 * cropped_scatter[row, col] * inv[row, col]
    pdf_costs += m_subspace * LOG_2_PI - m_subspace * np.log(n_points_in_cluster) + np.log(det)
    pdf_costs *= n_points_in_cluster / 2 / LOG_2
    return pdf_costs

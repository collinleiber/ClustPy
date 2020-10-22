import numpy as np

def mdl_costs_uniform_distribution(possibilities):
    if possibilities == 0:
        return 0
    return -np.log2(1 / possibilities)

def mdl_costs_orthogonal_matrix(n_points, dimensionality):
    cost = dimensionality * (dimensionality - 1) / 2
    cost *= mdl_costs_float_value(n_points)
    return cost

def mdl_costs_float_value(n_points):
    if n_points <= 0:
        raise ValueError(
            "The number of points must be larger than 0 to calculate the mdl costs of a float. Your input:\n" + str(
                n_points))
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
    if type(integer) is not int and type(integer) is not np.int32:
        raise ValueError(
            "The input to calculate the mdl costs of must be an integer. Your input:\n" + str(integer))
    cost = 0
    if integer != 0:
        last_interim_result = np.log2(integer)
        while last_interim_result > 0:
            cost += last_interim_result
            last_interim_result = np.log2(last_interim_result)
    const = np.log2(2.865064)
    return cost + const

def mdl_costs_gmm_single_covariance(cropped_V, m_subspace, scatter_matrices_subspace, n_points_in_cluster):
    if m_subspace == 0:
        return 0
    # Get costs for each cluster
    sum_scatter_matrices = np.sum(scatter_matrices_subspace, 0)
    # Get number of points which are not outliers
    full_pdf_costs = mdl_costs_cluster_single_variance(cropped_V, n_points_in_cluster,
                                                                     sum_scatter_matrices, m_subspace)
    # Return pdf_costs
    return full_pdf_costs


def mdl_costs_cluster_single_variance(cropped_V, n_points_in_cluster, scatter_matrix_cluster, m_subspace):
    # If only one point is in this cluster it is already encoded with the center
    if n_points_in_cluster <= 1 or m_subspace == 0:
        return 0
    # Calculate the actual costs
    trace = np.trace(np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix_cluster), cropped_V))
    if trace < -1e-20:
        raise Exception("Trace can not be negative!")
    # Can occur if all points in this cluster lie on the same position
    if trace <= 1e-20:
        return 0
    pdf_costs = 1 + np.log(2 * np.pi) + np.log(trace / m_subspace / n_points_in_cluster)
    pdf_costs *= n_points_in_cluster * m_subspace / 2 / np.log(2)
    return pdf_costs
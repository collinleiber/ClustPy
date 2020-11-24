"""
Mautz, Dominik, et al. "Discovering non-redundant k-means
clusterings in optimal subspaces." Proceedings of the 24th ACM
SIGKDD International Conference on Knowledge Discovery &
Data Mining. 2018.
"""

import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state
from sklearn.cluster._kmeans import _k_init as kpp
from sklearn.utils.extmath import row_norms
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import normalized_mutual_info_score as nmi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cluspy.utils._mdlcosts as mdl

_ACCEPTED_NUMERICAL_ERROR = 1e-6
_NOISE_SPACE_THRESHOLD = -1e-7


def _nrkmeans(X, n_clusters, V, m, P, centers, mdl_for_noisespace, outliers, max_iter, random_state):
    """
    Execute the nrkmeans algorithm. The algorithm will search for the optimal cluster subspaces and assignments
    depending on the input number of clusters and subspaces. The number of subspaces will automatically be traced by the
    length of the input n_clusters array.
    :param X: input data
    :param n_clusters: list containing number of clusters for each subspace
    :param V: orthogonal rotation matrix
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param centers: list containing the cluster centers for each subspace
    :param mdl_for_noisespace: boolean defining if MDL should be used to identify noise space dimensions
    :param outliers: boolean defining if outliers should be identified
    :param max_iter: maximum number of iterations for the algorithm
    :param random_state: use a fixed random state to get a repeatable solution
    :return: labels, centers, V, m, P, n_clusters (can get lost), scatter_matrices
    """
    V, m, P, centers, random_state, subspaces, labels, scatter_matrices = _initialize_nrkmeans_parameters(
        X, n_clusters, V, m, P, centers, mdl_for_noisespace, outliers, max_iter, random_state)
    # Check if labels stay the same (break condition)
    old_labels = None
    n_outliers = np.zeros(subspaces, dtype=int)
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Execute basic kmeans steps
        for i in range(subspaces):
            # Assign each point to closest cluster center
            labels[:, i] = _assign_labels(X, V, centers[i], P[i])
            # Update centers and scatter matrices depending on cluster assignments
            centers[i], scatter_matrices[i] = _update_centers_and_scatter_matrices(X, n_clusters[i], labels[:, i])
            # Remove empty clusters
            n_clusters[i], centers[i], scatter_matrices[i], labels[:, i] = _remove_empty_cluster(n_clusters[i],
                                                                                                 centers[i],
                                                                                                 scatter_matrices[i],
                                                                                                 labels[:, i])
            # (Optional) Check for outliers
            if outliers:
                labels[:, i], scatter_matrices[i], n_outliers[i] = _check_for_outliers(X, V, centers[i], labels[:, i],
                                                                                       scatter_matrices[i], m[i], P[i])
        # Check if labels have not changed
        if _are_labels_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()
        # Update rotation for each pair of subspaces
        for i in range(subspaces - 1):
            for j in range(i + 1, subspaces):
                # Do rotation calculations
                P_1_new, P_2_new, V_new = _update_rotation(X, V, i, j, n_clusters, P, scatter_matrices, labels,
                                                           mdl_for_noisespace, outliers, n_outliers)
                # Update V, m, P
                m[i] = len(P_1_new)
                m[j] = len(P_2_new)
                P[i] = P_1_new
                P[j] = P_2_new
                V = V_new
        # Handle empty subspaces (no dimensionalities left) -> Should be removed
        subspaces, n_clusters, m, P, centers, labels, scatter_matrices = _remove_empty_subspace(subspaces,
                                                                                                n_clusters,
                                                                                                m, P,
                                                                                                centers,
                                                                                                labels,
                                                                                                scatter_matrices)
    # print("[NrKmeans] Converged in iteration " + str(iteration + 1))
    # Return relevant values
    return labels, centers, V, m, P, n_clusters, scatter_matrices


def _initialize_nrkmeans_parameters(X, n_clusters, V, m, P, centers, mdl_for_noisespace, outliers, max_iter,
                                    random_state):
    """
    Initialize the input parameters form NrKmeans. This means that all input values which are None must be defined.
    Also all input parameters which are not None must be checked, if a correct execution is possible.
    :param X: input data
    :param n_clusters: list containing number of clusters for each subspace
    :param V: orthogonal rotation matrix
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param centers: list containing the cluster centers for each subspace
    :param mdl_for_noisespace: boolean defining if MDL should be used to identify noise space dimensions
    :param outliers: boolean defining if outliers should be identified
    :param max_iter: maximum number of iterations for the algorithm
    :param random_state: use a fixed random state to get a repeatable solution
    :return: checked V, m, P, centers, random_state, number of subspaces, labels, scatter_matrices
    """
    data_dimensionality = X.shape[1]
    random_state = check_random_state(random_state)
    # Check if n_clusters is a list
    if not type(n_clusters) is list:
        raise ValueError(
            "Number of clusters must be specified for each subspace and therefore be a list.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains negative values
    if len([x for x in n_clusters if x < 1]) > 0:
        raise ValueError(
            "Number of clusters must not contain negative values or 0.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains more than one noise space
    nr_noise_spaces = len([x for x in n_clusters if x == 1])
    if nr_noise_spaces > 1:
        raise ValueError(
            "Only one subspace can be the noise space (number of clusters = 1).\nYour input:\n" + str(n_clusters))
    # Check if noise space is not the last member in n_clusters
    if nr_noise_spaces != 0 and n_clusters[-1] != 1:
        raise ValueError(
            "Noise space (number of clusters = 1) must be the last entry in n_clsuters.\nYour input:\n" + str(
                n_clusters))
    # Get number of subspaces
    subspaces = len(n_clusters)
    # Check if V is orthogonal
    if V is None:
        if data_dimensionality > 1:
            V = ortho_group.rvs(dim=data_dimensionality,
                                random_state=random_state)
        else:
            V = np.ones((1, 1))
    if not _is_matrix_orthogonal(V):
        raise ValueError("Your input matrix V is not orthogonal.\nV:\n" + str(V))
    if V.shape[0] != data_dimensionality or V.shape[1] != data_dimensionality:
        raise ValueError(
            "The shape of the input matrix V must equal the data dimensionality.\nShape of V:\n" + str(V.shape))
    # Calculate dimensionalities m
    if m is None and P is None:
        m = [int(data_dimensionality / subspaces)] * subspaces
        if data_dimensionality % subspaces != 0:
            choices = random_state.choice(range(subspaces), data_dimensionality - np.sum(m))
            for choice in choices:
                m[choice] += 1
    # If m is None but P is defined use P's dimensionality
    elif m is None:
        m = [len(x) for x in P]
    if not type(m) is list or not len(m) is subspaces:
        raise ValueError("A dimensionality list m must be specified for each subspace.\nYour input:\n" + str(m))
    # Calculate projections P
    if P is None:
        possible_projections = list(range(data_dimensionality))
        P = []
        for dimensionality in m:
            choices = random_state.choice(possible_projections, dimensionality, replace=False)
            P.append(choices)
            possible_projections = list(set(possible_projections) - set(choices))
    if not type(P) is list or not len(P) is subspaces:
        raise ValueError("Projection lists must be specified for each subspace.\nYour input:\n" + str(P))
    else:
        # Check if the length of entries in P matches values of m
        used_dimensionalities = []
        for i, dimensionality in enumerate(m):
            used_dimensionalities.extend(P[i])
            if not len(P[i]) == dimensionality:
                raise ValueError(
                    "Values for dimensionality m and length of projection list P do not match.\nDimensionality m:\n" + str(
                        dimensionality) + "\nDimensionality P:\n" + str(P[i]))
        # Check if every dimension in considered in P
        if sorted(used_dimensionalities) != list(range(data_dimensionality)):
            raise ValueError("Projections P must include all dimensionalities.\nYour used dimensionalities:\n" + str(
                used_dimensionalities))
    # Define initial cluster centers with kmeans++ for each subspace
    if centers is None:
        centers = [kpp(X, k, row_norms(X, squared=True), random_state) for k in n_clusters]
    if not type(centers) is list or not len(centers) is subspaces:
        raise ValueError("Cluster centers must be specified for each subspace.\nYour input:\n" + str(centers))
    else:
        # Check if number of centers for subspaces matches value in n_clusters
        for i, subspace_centers in enumerate(centers):
            if not n_clusters[i] == len(subspace_centers):
                raise ValueError(
                    "Values for number of clusters n_clusters and number of centers do not match.\nNumber of clusters:\n" + str(
                        n_clusters[i]) + "\nNumber of centers:\n" + str(len(subspace_centers)))
    # Check max iter
    if max_iter is None or type(max_iter) is not int or max_iter <= 0:
        raise ValueError(
            "Max_iter must be an integer larger than 0. Your Max_iter:\n" + str(max_iter))
    # Check outliers value
    if type(outliers) is not bool:
        raise ValueError(
            "outliers must be a boolean. Your outliers:\n" + str(outliers))
    # Check mdl_for_noisespace value
    if type(mdl_for_noisespace) is not bool:
        raise ValueError(
            "mdl_for_noisespace must be a boolean. Your outliers:\n" + str(mdl_for_noisespace))
    # Initial labels and scatter matrices
    labels = np.zeros((X.shape[0], subspaces), dtype=np.int32)
    scatter_matrices = [None] * subspaces
    return V, m, P, centers, random_state, subspaces, labels, scatter_matrices


def _assign_labels(X, V, centers_subspace, P_subspace):
    """
    Assign each point in each subspace to its nearest cluster center.
    :param X: input data
    :param V: orthogonal rotation matrix
    :param centers_subspace: cluster centers of the subspace
    :param P_subspace: projecitons of the subspace
    :return: list with cluster assignments
    """
    cropped_X = np.matmul(X, V[:, P_subspace])
    cropped_centers = np.matmul(centers_subspace, V[:, P_subspace])
    # Find nearest center
    labels, _ = pairwise_distances_argmin_min(X=cropped_X, Y=cropped_centers, metric='euclidean',
                                              metric_kwargs={'squared': True})
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32)
    return labels


def _update_centers_and_scatter_matrices(X, n_clusters_subspace, labels_subspace):
    """
    Update the cluster centers within this subspace depending on the labels of the data points. Also updates the the
    scatter matrix of each cluster by summing up the outer product of the distance between each point and center.
    :param X: input data
    :param n_clusters_subspace: number of clusters of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :return: centers, scatter_matrices - Updated cluster center and scatter matrices (one scatter matrix for each cluster)
    """
    # Create empty matrices
    centers = np.zeros((n_clusters_subspace, X.shape[1]))
    scatter_matrices = np.zeros((n_clusters_subspace, X.shape[1], X.shape[1]))
    # Update cluster parameters
    for center_id, _ in enumerate(centers):
        # Get points in this cluster
        points_in_cluster = np.where(labels_subspace == center_id)[0]
        if len(points_in_cluster) == 0:
            centers[center_id] = np.nan
            continue
        # Update center
        centers[center_id] = np.average(X[points_in_cluster], axis=0)
        # Update scatter matrix
        centered_points = X[points_in_cluster] - centers[center_id]
        for entry in centered_points:
            rank1 = np.outer(entry, entry)
            scatter_matrices[center_id] += rank1
    return centers, scatter_matrices


def _remove_empty_cluster(n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace):
    """
    Check if after label assignemnt and center update a cluster got lost. Empty clusters will be
    removed for the following rotation und iterations. Therefore all necessary lists will be updated.
    :param n_clusters_subspace: number of clusters of the subspace
    :param centers_subspace: cluster centers of the subspace
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :return: n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace (updated)
    """
    # Check if any cluster is lost
    if np.any(np.isnan(centers_subspace)):
        # Get ids of lost clusters
        empty_clusters = np.where(np.any(np.isnan(centers_subspace), axis=1))[0]
        print(
            "[NrKmeans] ATTENTION: Clusters were lost! Number of lost clusters: " + str(
                len(empty_clusters)) + " out of " + str(
                len(centers_subspace)))
        # Update necessary lists
        n_clusters_subspace -= len(empty_clusters)
        for cluster_id in reversed(empty_clusters):
            centers_subspace = np.delete(centers_subspace, cluster_id, axis=0)
            scatter_matrices_subspace = np.delete(scatter_matrices_subspace, cluster_id, axis=0)
            labels_subspace[labels_subspace > cluster_id] -= 1
    return n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace


def _update_rotation(X, V, first_index, second_index, n_clusters, P, scatter_matrices, labels, mdl_for_noisespace,
                     outliers, n_outliers):
    """
    Update the rotation of the subspaces. Updates V and m and P for the input subspaces.
    :param X: input data
    :param V: orthogonal rotation matrix
    :param first_index: index of the first subspace
    :param second_index: index of the second subspace (can be noise space)
    :param n_clusters: list containing number of clusters for each subspace
    :param P: list containing projections for each subspace
    :param scatter_matrices: list containing scatter matrices for each subspace
    :param mdl_for_noisespace: boolean defining if MDL should be used to identify noise space dimensions
    :param outliers: boolean defining if outliers should be identified
    :param n_outliers: number of outliers
    :return: P_1_new, P_2_new, V_new - new P for the first subspace, new P for the second subspace and new V
    """
    # Check if second subspace is the noise space
    is_noise_space = (n_clusters[second_index] == 1)
    # Get combined projections and combined_cropped_V
    P_1 = P[first_index]
    P_2 = P[second_index]
    P_combined = np.append(P_1, P_2)
    # Check if both P's are empty
    if len(P_combined) == 0:
        return P_1, P_2, V
    cropped_V_combined = V[:, P_combined]
    # Prepare input for eigenvalue decomposition.
    if outliers:
        sum_scatter_matrices_1 = np.sum(scatter_matrices[first_index], 0) * X.shape[0] / (
                X.shape[0] - n_outliers[first_index])
        sum_scatter_matrices_2 = np.sum(scatter_matrices[second_index], 0) * X.shape[0] / (
                X.shape[0] - n_outliers[second_index])
    else:
        sum_scatter_matrices_1 = np.sum(scatter_matrices[first_index], 0)
        sum_scatter_matrices_2 = np.sum(scatter_matrices[second_index], 0)
    diff_scatter_matrices = sum_scatter_matrices_1 - sum_scatter_matrices_2
    projected_diff_scatter_matrices = np.matmul(np.matmul(cropped_V_combined.transpose(), diff_scatter_matrices),
                                                cropped_V_combined)
    if not _is_matrix_symmetric(projected_diff_scatter_matrices):
        raise Exception(
            "Input for eigenvalue decomposition is not symmetric.\nInput:\n" + str(projected_diff_scatter_matrices))
    # Get eigenvalues and eigenvectors (already sorted by eigh)
    e, V_C = np.linalg.eigh(projected_diff_scatter_matrices)
    if not _is_matrix_orthogonal(V_C):
        raise Exception("Eigenvectors are not orthogonal.\nEigenvectors:\n" + str(V_C))
    # Use transitions and eigenvectors to build V full
    V_F = _create_full_rotation_matrix(X.shape[1], P_combined, V_C)
    # Calculate new V
    V_new = np.matmul(V, V_F)
    if not _is_matrix_orthogonal(V_new):
        raise Exception("New V is not othogonal.\nNew V:\n" + str(V_new))
    # Use number of negative eigenvalues to get new projections
    n_negative_e = len(e[e < 0])
    if is_noise_space:
        if mdl_for_noisespace:
            P_1_new, P_2_new = _compare_possible_splits(X, V_new, first_index, second_index,
                                                        n_negative_e, P_combined, n_clusters,
                                                        scatter_matrices, labels, outliers, n_outliers)
        else:
            n_negative_e = len(e[e < _NOISE_SPACE_THRESHOLD])
            P_1_new, P_2_new = _update_projections(P_combined, n_negative_e)
    else:
        P_1_new, P_2_new = _update_projections(P_combined, n_negative_e)
    # Return new dimensionalities, projections and V
    return P_1_new, P_2_new, V_new


def _get_cost_function_of_subspace(cropped_V, scatter_matrices_subspace):
    """
    Calculate the result of the NrKmeans loss function for a certain subspace.
    Depends on the rotation and the scattermatrices. Calculates:
    P^T*V^T*S*V*P
    :param cropped_V: cropped orthogonal rotation matrix
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :return: result of the NrKmeans cost function
    """
    scatter_matrix = np.sum(scatter_matrices_subspace, 0)
    return np.trace(np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix),
                              cropped_V))


def _create_full_rotation_matrix(dimensionality, P_combined, V_C):
    """
    Create full rotation matrix out of the found eigenvectors. Set diagonal to 1 and overwrite columns and rows with
    indices in P_combined (consider the oder) with the values from V_C. All other values should be 0.
    :param dimensionality: dimensionality of the full rotation matrix
    :param P_combined: combined projections of the subspaces
    :param V_C: the calculated eigenvectors
    :return: the new full rotation matrix
    """
    V_F = np.identity(dimensionality)
    V_F[np.ix_(P_combined, P_combined)] = V_C
    return V_F


def _update_projections(P_combined, n_negative_e):
    """
    Create the new projections for the subspaces. First subspace gets all as many projections as there are negative
    eigenvalues. Second subspace gets all other projections in reversed order.
    :param P_combined: combined projections of the subspaces
    :param n_negative_e: number of negative eigenvalues
    :return: P_1_new, P_2_new - projections for the subspaces
    """
    P_1_new = np.array([P_combined[x] for x in range(n_negative_e)], dtype=int)
    P_2_new = np.array([P_combined[x] for x in reversed(range(n_negative_e, len(P_combined)))], dtype=int)
    return P_1_new, P_2_new


def _remove_empty_subspace(subspaces, n_clusters, m, P, centers, labels, scatter_matrices):
    """
    Check if after rotation and rearranging the dimensionalities a empty subspaces occurs. Empty subspaces will be
    removed for the next iteration. Therefore all necessary lists will be updated.
    :param subspaces: number of subspaces
    :param n_clusters:
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param centers: list containing the cluster centers for each subspace
    :param labels: list containing cluster assignments for each subspace
    :param scatter_matrices: list containing scatter matrices for each subspace
    :return: subspaces, n_clusters, m, P, centers, labels, scatter_matrices
    """
    if 0 in m:
        np_m = np.array(m)
        empty_spaces = np.where(np_m == 0)[0]
        print(
            "[NrKmeans] ATTENTION: Subspaces were lost! Number of lost subspaces: " + str(
                len(empty_spaces)) + " out of " + str(
                len(m)))
        subspaces -= len(empty_spaces)
        n_clusters = [x for i, x in enumerate(n_clusters) if i not in empty_spaces]
        m = [x for i, x in enumerate(m) if i not in empty_spaces]
        P = [x for i, x in enumerate(P) if i not in empty_spaces]
        centers = [x for i, x in enumerate(centers) if i not in empty_spaces]
        labels = np.delete(labels, empty_spaces, axis=1)
        scatter_matrices = [x for i, x in enumerate(scatter_matrices) if i not in empty_spaces]
    return subspaces, n_clusters, m, P, centers, labels, scatter_matrices


def _is_matrix_orthogonal(matrix):
    """
    Check whether a matrix is orthogonal by comparing the multiplication of the matrix and its transpose and
    the identity matrix.
    :param matrix: input matrix
    :return: True if matrix is orthogonal
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    matrix_product = np.matmul(matrix, matrix.transpose())
    return np.allclose(matrix_product, np.identity(matrix.shape[0]), atol=_ACCEPTED_NUMERICAL_ERROR)


def _is_matrix_symmetric(matrix):
    """
    Check whether a matrix is symmetric by comparing the matrix with its transpose.
    :param matrix: input matrix
    :return: True if matrix is symmetric
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.T, atol=_ACCEPTED_NUMERICAL_ERROR)


def _are_labels_equal(labels_new, labels_old):
    """
    Check if the old labels and new labels are equal. Therefore check the nmi for each subspace. If all are 1, labels
    have not changed.
    :param labels_new: new labels list
    :param labels_old: old labels list
    :return: True if labels for all subspaces are the same
    """
    if labels_new is None or labels_old is None or labels_new.shape[1] != labels_old.shape[1]:
        return False
    return all([nmi(labels_new[:, i], labels_old[:, i], "arithmetic") == 1 for i in range(labels_new.shape[1])])


"""
===================== MDL Additions =====================
"""


def _compare_possible_splits(X, V, cluster_index, noise_index, n_negative_e, P_combined,
                             n_clusters, scatter_matrices, labels, outliers, n_outliers):
    """
    Use MDL to find the cheapest combination of cluster and noise dimensionality. Try raising number of cluster space
    dimensionality until costs increase.
    :param X: input data
    :param V: orthogonal rotation matrix
    :param cluster_index: index of the cluster space
    :param noise_index: index of the noise space
    :param n_negative_e: number of negative eigenvalues
    :param P_combined: combined projections of clsuter and noise space
    :param n_clusters: list containing number of clusters for each subspace
    :param scatter_matrices: list containing scatter matrices for each subspace
    :param outliers: boolean defining if outliers should be identified
    :param n_outliers: number of outliers
    :return: P_cluster, P_noise - projections for cluster and noise space
    """
    best_costs = np.inf
    best_P_cluster, best_P_noise = _update_projections(P_combined, 0)
    # Try raising number of dimensionalities in the cluster space until costs raise
    for m_cluster in range(1, n_negative_e + 1):
        m_noise = len(P_combined) - m_cluster
        P_cluster, P_noise = _update_projections(P_combined, m_cluster)
        # Get costs for this combination of dimensionalities
        costs = _mdl_m_dependant_subspace_costs(X, V, cluster_index, noise_index, m_cluster,
                                               m_noise, P_cluster, P_noise,
                                               scatter_matrices, n_clusters, labels, outliers, n_outliers)
        # If costs are lower, next try. Else break
        if costs < best_costs:
            best_costs = costs
            best_P_cluster = P_cluster.copy()
            best_P_noise = P_noise.copy()
        else:
            break
    return best_P_cluster, best_P_noise


def _check_for_outliers(X, V, centers_subspace, labels_subspace, scatter_matrices_subspace, m_subspace, P_subspace):
    """
    Check for each point if it should be interpreted as an outlier in this subspace. Outliers are defined by the changing
    cluster costs if point is removed from cluster. If it is cheaper to encode the point separately it is an outlier.
    For each outlier the label will be set to -1 and the distance to its old center will be subtracted from the scatter
    matrix.
    :param X: input data
    :param V: orthogonal rotation matrix
    :param centers_subspace: cluster centers of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :param m_subspace: dimensionality of the subspace
    :param P_subspace: projecitons of the subspace
    :return: labels, scatter_matrices, n_outliers - the updated labels and scatter matrices and total number of outliers
    """
    n_points = X.shape[0]
    # Copy labels and scatter matrices lists to update theses based on new outliers
    labels_subspace_copy = labels_subspace.copy()
    scatter_matrices_subspace_copy = scatter_matrices_subspace.copy()
    # Get basic cluster parameters for this subspace
    cropped_V = V[:, P_subspace]
    # Get original costs encoding the points
    sum_scatter_matrices = np.sum(scatter_matrices_subspace, 0)
    cluster_costs = mdl.mdl_costs_gaussian_single_variance(cropped_V, n_points,
                                                           sum_scatter_matrices, m_subspace)
    # Get costs for single outlier in this subspace
    outlier_coding_cost = _mdl_outlier_costs(n_points, m_subspace)
    # Save the most distant non-outlier and nearest outlier (Important for runtime speedup!)
    farthest_non_outliers = 0
    neareast_outliers = np.inf
    # Check each point separately if it is an outlier. If so, copied lists will be updated
    n_outliers_in_cluster = np.zeros(len(centers_subspace))
    n_points_in_cluster = np.zeros(len(centers_subspace))
    for i, x in enumerate(X):
        old_label = labels_subspace[i]
        n_points_in_cluster[old_label] += 1
        is_outlier, outer, farthest_non_outliers, neareast_outliers = _is_point_outlier(x, n_points, cluster_costs,
                                                                                        outlier_coding_cost, cropped_V,
                                                                                        centers_subspace[old_label],
                                                                                        m_subspace,
                                                                                        sum_scatter_matrices,
                                                                                        farthest_non_outliers,
                                                                                        neareast_outliers)
        if is_outlier:
            labels_subspace_copy[i] = -1
            scatter_matrices_subspace_copy[old_label] -= outer
            n_outliers_in_cluster[old_label] += 1
    n_outliers_total = np.sum(n_outliers_in_cluster)
    # Revert changes for clusters that are empty now
    for cluster_index, n_noise_clu in enumerate(n_outliers_in_cluster):
        if n_points_in_cluster[cluster_index] == n_noise_clu:
            n_outliers_total -= n_noise_clu
            # Revert changes
            scatter_matrices_subspace_copy[cluster_index] = scatter_matrices_subspace[cluster_index]
            labels_subspace_copy[labels_subspace == cluster_index] = cluster_index
    # Use new values for labels and scatter matrices
    return labels_subspace_copy, scatter_matrices_subspace_copy, n_outliers_total


def _is_point_outlier(x, n_points, cluster_costs, outlier_coding_cost, cropped_V, center_cluster,
                      m_subspace, scatter_matrix, farthest_non_outliers, neareast_outliers):
    """
    Check for this point whether it is cheaper to encode it separately than combined with all other point in this
    cluster. Execution time is shortened by comparing distance to nearest outlier and farthest non-outlier.
    :param x: data point
    :param n_points: number of points in this cluster
    :param cluster_costs: original costs of all points within a cluster
    :param outlier_coding_cost: costs of encoding a single outlier
    :param cropped_V: cropped orthogonal rotation matrix
    :param center_cluster: cluster center
    :param m_subspace: dimensionality of the subspace
    :param scatter_matrix: scatter matrix of the cluster
    :param farthest_non_outliers: the most distant non-outlier for each cluster
    :param neareast_outliers: the nearest outlier for each cluster
    :return: Pair of: True if point is an outlier, outer product of the distance between the point and the center
    """
    # Get distance from point to center
    distance = x - center_cluster
    cropped_distance = np.matmul(distance, cropped_V)
    distance_squared = np.inner(cropped_distance, cropped_distance)
    # Is distance smaller than the most distant non-outlier
    if distance_squared <= farthest_non_outliers:
        return False, None, farthest_non_outliers, neareast_outliers
    # Get outer product of point to center
    outer = np.outer(distance, distance)
    # Is distance greater than the least distant outlier
    if distance_squared >= neareast_outliers:
        return True, outer, farthest_non_outliers, neareast_outliers
    # Remove squared difference from point to center from scatter matrix
    scatter_matrix_cluster_tmp = scatter_matrix - outer
    # Calculate new pdf costs for cluster
    new_cluster_costs = mdl.mdl_costs_gaussian_single_variance(cropped_V, n_points - 1,
                                                               scatter_matrix_cluster_tmp,
                                                               m_subspace)
    # Check if coding costs would be lower than before
    if new_cluster_costs + outlier_coding_cost < cluster_costs:
        neareast_outliers = distance_squared
        return True, outer, farthest_non_outliers, neareast_outliers
    else:
        farthest_non_outliers = distance_squared
        return False, None, farthest_non_outliers, neareast_outliers


def _mdl_m_dependant_subspace_costs(X, V, cluster_index, noise_index, m_cluster, m_noise,
                               P_cluster, P_noise, scatter_matrices, n_clusters, labels, outliers, n_outliers):
    """
    Get the total costs depending on the subspace dimensionality for one cluster space and the noise space.
    Method can be used to determine the best possible number do dimensionalities to swap from cluster into the noise
    space.
    :param X: the data
    :param V: orthogonal rotation matrix
    :param cluster_index: index of the cluster space
    :param noise_index: index of the noise space
    :param m_cluster: dimensionality of the cluster space
    :param m_noise: dimensionality of the noise space
    :param P_cluster: projections of the cluster space
    :param P_noise: projections of the noise space
    :param scatter_matrices: list containing all scatter matrices of the subspaces
    :param n_clusters: list containing number of clusters for each subspace
    :param outliers: boolean defining if outliers should be identified
    :param n_outliers: number of outliers
    :return: total costs for this selection of dimensionalites
    """
    n_points = X.shape[0]
    # ==== Costs of cluster space ====
    # Costs for cluster dimensionality
    cluster_costs = mdl.mdl_costs_integer_value(m_cluster)
    # Costs for centers
    cluster_costs += n_clusters[cluster_index] * m_cluster * mdl.mdl_costs_float_value(n_points)
    # Costs for point encoding
    cropped_V_cluster = V[:, P_cluster]
    cluster_costs += mdl.mdl_costs_gmm_single_covariance(cropped_V_cluster, m_cluster,
                                                         scatter_matrices[cluster_index],
                                                         n_points - n_outliers[cluster_index])
    # Costs for outliers
    if outliers:
        cluster_costs += _mdl_outlier_costs(n_points, m_cluster) * n_outliers[cluster_index]
    # ==== Costs of noise space ====
    # Costs for noise dimensionality
    noise_costs = mdl.mdl_costs_integer_value(m_noise)
    # Costs for centers
    noise_costs += n_clusters[noise_index] * m_noise * mdl.mdl_costs_float_value(n_points)
    # Costs for point encoding
    cropped_V_noise = V[:, P_noise]
    noise_costs += mdl.mdl_costs_gmm_single_covariance(cropped_V_noise, m_noise,
                                                       scatter_matrices[noise_index],
                                                       n_points - n_outliers[noise_index])
    # Costs for outliers
    if outliers:
        noise_costs += _mdl_outlier_costs(n_points, m_noise) * n_outliers[noise_index]
    # Return costs
    return cluster_costs + noise_costs


def _mdl_costs(X, nrkmeans):
    """
    Calculate the total mdl cost of a non redudant clustering found by NrKmeans.
    Total costs consists of general costs which describe the whole system (number of subspaces, V, data dimensionality)
    and separate costs for each subspace. This include the exact dimensionalities of the subspaces, number of clusters,
    the centers, cluster assignments, cluster variances and coding costs for each point within a cluster and for each
    outlier.
    :param X: the data
    :param n_clusters: list containing number of clusters for each subspace
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param V: orthogonal rotation matrix
    :param scatter_matrices: list containing all scatter matrices of the subspaces
    :param labels: list with the labels of the cluster assingments for each subspace. -1 equals outlier
    :param outliers: boolean defining if outliers should be identified
    :return: total costs (int), subspace costs (list)
    """
    n_points = X.shape[0]
    n_outliers = 0
    subspaces = len(nrkmeans.m)
    # Calculate costs
    global_costs = 0
    # Costs of number of subspaces
    global_costs += mdl.mdl_costs_integer_value(subspaces)
    # Costs of matrix V
    global_costs += mdl.mdl_costs_orthogonal_matrix(n_points, mdl.mdl_costs_float_value(n_points))
    # Costs for each subspace
    all_subspace_costs = []
    for subspace_nr in range(subspaces):
        model_costs = 0
        # Costs for dimensionality
        model_costs += mdl.mdl_costs_integer_value(nrkmeans.m[subspace_nr])
        # Number of clusters in subspace
        model_costs += mdl.mdl_costs_integer_value(nrkmeans.n_clusters[subspace_nr])
        # Costs for cluster centers
        model_costs += nrkmeans.n_clusters[subspace_nr] * nrkmeans.m[subspace_nr] * mdl.mdl_costs_float_value(n_points)
        # Subspace Variance costs
        model_costs += mdl.mdl_costs_float_value(n_points)
        # Coding costs for outliers
        cropped_V = nrkmeans.V[:, nrkmeans.P[subspace_nr]]
        outlier_costs = 0
        if nrkmeans.outliers:
            # Encode number of outliers
            n_outliers = len(nrkmeans.labels_[:, subspace_nr][nrkmeans.labels_[:, subspace_nr] == -1])
            model_costs += mdl.mdl_costs_integer_value(n_outliers)
            # Encode coding costs of outliers
            outlier_costs += _mdl_outlier_costs(n_points, nrkmeans.m[subspace_nr]) * n_outliers
        # Cluster assignment (is 0 for noise space)
        assignment_costs = (n_points - n_outliers) * mdl.mdl_costs_uniform_distribution(nrkmeans.n_clusters[subspace_nr])
        # Coding costs for each point
        coding_costs = mdl.mdl_costs_gmm_single_covariance(cropped_V, nrkmeans.m[subspace_nr],
                                                           nrkmeans.scatter_matrices_[subspace_nr], n_points - n_outliers)
        # Save this subspace costs
        all_subspace_costs.append(model_costs + outlier_costs + assignment_costs + coding_costs)
    # return full and single subspace costs
    total_costs = global_costs + sum(all_subspace_costs)
    return total_costs, global_costs, all_subspace_costs

def _mdl_outlier_costs(n_points, m_subspace):
    return mdl.mdl_costs_float_value(n_points) * m_subspace


"""
==================== NrKmeans Object ====================
"""


class NrKmeans():
    def __init__(self, n_clusters, V=None, m=None, P=None, input_centers=None, mdl_for_noisespace=False, outliers=False,
                 max_iter=300, random_state=None):
        """
        Create new NrKmeans instance. Gives the opportunity to use the fit() method to cluster a dataset.
        :param n_clusters: list containing number of clusters for each subspace
        :param V: orthogonal rotation matrix (optional)
        :param m: list containing number of dimensionalities for each subspace (optional)
        :param P: list containing projections for each subspace (optional)
        :param input_centers: list containing the cluster centers for each subspace (optional)
        :param mdl_for_noisespace: boolean defining if MDL should be used to identify noise space dimensions (default: False)
        :param outliers: boolean defining if outliers should be identified (default: False)
        :param max_iter: maximum number of iterations for the NrKmaens algorithm (default: 300)
        :param random_state: use a fixed random state to get a repeatable solution (optional)
        """
        # Fixed attributes
        self.input_n_clusters = n_clusters.copy()
        self.max_iter = max_iter
        self.random_state = random_state
        self.mdl_for_noisespace = mdl_for_noisespace
        self.outliers = outliers
        # Variables
        self.n_clusters = n_clusters
        self.input_centers = input_centers
        self.V = V
        self.m = m
        self.P = P

    def fit(self, X):
        """
        Cluster the input dataset with the Nr-Kmeans algorithm. Saves the labels, centers, V, m, P and scatter matrices
        in the NrKmeans object.
        :param X: input data
        :return: the Nr-Kmeans object
        """
        labels, centers, V, m, P, n_clusters, scatter_matrices = _nrkmeans(X, self.n_clusters, self.V, self.m,
                                                                           self.P, self.input_centers,
                                                                           self.mdl_for_noisespace,
                                                                           self.outliers, self.max_iter,
                                                                           self.random_state)
        # Update class variables
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.V = V
        self.m = m
        self.P = P
        self.n_clusters = n_clusters
        self.scatter_matrices_ = scatter_matrices
        return self


    # def predict(self, X):
    #     """
    #     Predict the labels of an input dataset. For this method the results from the NrKmeans fit() method will be used.
    #     If Nr-Kmeans was fitted with a defined outlier strategy this will also be applied here.
    #     :param X: input data
    #     :return: Array with the labels of the points for each subspace
    #     """
    #     # Check if Nr-Kmeans model was created
    #     if self.labels is None:
    #         raise Exception("The NrKmeans algorithm has not run yet. Use the fit() function first.")
    #     predicted_labels = [None] * len(self.n_clusters)
    #     # Get labels for each subspace
    #     for sub in range(len(self.n_clusters)):
    #         # Predict the labels
    #         predicted_labels[sub] = _assign_labels(X, self.V, self.centers[sub], self.P[sub])
    #         # Check for outliers
    #         if self.outliers:
    #             n_points = X.shape[0]
    #             # Get basic cluster parameters for this subspace
    #             cropped_V = self.V[:, self.P[sub]]
    #             # Get original costs encoding the points
    #             sum_scatter_matrices = np.sum(self.scatter_matrices[sub], 0)
    #             cluster_costs = mdl._calculate_single_cluster_costs_single_variance(cropped_V, n_points,
    #                                                                                 sum_scatter_matrices, self.m[sub])
    #             # Get costs for single outlier in this subspace
    #             outlier_coding_cost = mdl.single_outlier_costs(X, cropped_V)
    #             # Check each point separately if it is an outlier
    #             for i, x in enumerate(X):
    #                 # Get costs for additional point in this cluster
    #                 old_label = predicted_labels[sub][i]
    #                 # Get outer product of point to center
    #                 distance = x - self.centers[sub][old_label]
    #                 outer = np.outer(distance, distance)
    #                 # Add outer product to original scatter matrix
    #                 scatter_matrix_cluster_tmp = self.scatter_matrices[sub][old_label] + outer
    #                 # Calculate new costs
    #                 new_pdf_costs = mdl._calculate_single_cluster_costs_single_variance(cropped_V, n_points + 1,
    #                                                                                     scatter_matrix_cluster_tmp,
    #                                                                                     self.m[sub])
    #                 # Check if coding costs would be lower
    #                 if outlier_coding_cost + cluster_costs < new_pdf_costs:
    #                     predicted_labels[sub][i] = -1
    #     # Retrun the predicted labels
    #     return predicted_labels

    def transform_full_space(self, X):
        """
        Transfrom the input dataset with the orthogonal rotation matrix V from the Nr-Kmeans object.
        :param X: input data
        :return: the rotated dataset
        """
        return np.matmul(X, self.V)

    def transform_subspace(self, X, subspace_index):
        """
        Transform the input dataset with the orthogonal rotation matrix V projected onto a special subspace.
        :param X: input data
        :param subspace_index: index of the subspace
        :return: the rotated dataset
        """
        cluster_space_V = self.V[:, self.P[subspace_index]]
        return np.matmul(X, cluster_space_V)

    def have_subspaces_been_lost(self):
        """
        Check whether subspaces have been lost during Nr-Kmeans execution.
        :return: True if at least one subspace has been lost
        """
        return len(self.n_clusters) != len(self.input_n_clusters)

    def have_clusters_been_lost(self):
        """
        Check whether clusteres within a subspace have been lost during Nr-Kmeans execution.
        Will also return true if subspaces have been lost (check have_subspaces_been_lost())
        :return: True if at least one cluster has been lost
        """
        return not np.array_equal(self.input_n_clusters, self.n_clusters)

    def get_cluster_count_of_changed_subspaces(self):
        """
        Get the Number of clusters of the changed subspaces. If no subspace/cluster is lost, empty list will be
        returned.
        :return: list with the changed cluster count
        """
        changed_subspace = self.input_n_clusters.copy()
        for x in self.n_clusters:
            if x in changed_subspace:
                changed_subspace.remove(x)
        return changed_subspace

    def plot_subspace(self, X, subspace_index, labels=None, title=None):
        """
        Plot the specified subspace as scatter matrix plot.
        :param X: input data
        :param subspace_index: index of the subspace
        :param labels: the labels to use for the plot (default: labels found by Nr-Kmeans)
        :return: a scatter matrix plot of the input data
        """
        if self.labels_ is None:
            raise Exception("The NrKmeans algorithm has not run yet. Use the fit() function first.")
        if labels is None:
            labels = self.labels_[:, subspace_index]
        if X.shape[0] != labels.shape[0]:
            raise Exception("Number of data objects must match the number of labels.")
        if self.m[subspace_index] > 10:
            print("[NrKmeans] Info: Subspace has a dimensionality > 10 and will not be plotted.")
            return
        sns.set(style="ticks")
        plot_data = pd.DataFrame(self.transform_subspace(X, subspace_index))
        columns = plot_data.columns.values
        plot_data["labels"] = labels
        # If group contains only one element KDE Plot will not work -> Use histrogram instead
        diag = "auto"
        if 1 in np.unique(labels, return_counts=True)[1]:
            diag = "hist"
        g = sns.pairplot(plot_data, hue="labels", vars=columns, diag_kind=diag)
        # Add title
        my_title = "Subspace {0}".format(subspace_index) if title is None else "{0} / Subspace {1}".format(title,
                                                                                                           subspace_index)
        g.fig.suptitle(my_title)
        plt.show()

    # TODO
    # def calculate_mdl_costs(self, X):
    #     """
    #     Calculate the Mdl Costs of this NrKmeans result.
    #     :param X: input data
    #     :return: the MdlCosts object
    #     """
    #     if self.labels is None:
    #         raise Exception("The NrKmeans algorithm has not run yet. Use the fit() function first.")
    #     self.mdl_costs = mdl.MdlCosts(X, self)
    #     return self.mdl_costs

    def calculate_cost_function(self):
        """
        Calculate the result of the NrKmeans loss function. Depends on the rotation and the scattermatrices.
        Calculates for each subspace:
        P^T*V^T*S*V*P
        :return: result of the NrKmeans cost function
        """
        if self.labels_ is None:
            raise Exception("The NrKmeans algorithm has not run yet. Use the fit() function first.")
        costs = np.sum(
            [_get_cost_function_of_subspace(self.V[:, self.P[i]], s) for i, s in enumerate(self.scatter_matrices_)])
        return costs

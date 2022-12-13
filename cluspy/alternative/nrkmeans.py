"""
@authors:
Collin Leiber
"""

import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state
from scipy.spatial.distance import pdist
from sklearn.utils.extmath import row_norms
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.base import BaseEstimator, ClusterMixin
from cluspy.utils.plots import plot_scatter_matrix
import cluspy.utils._mdlcosts as mdl
from cluspy.utils._wrapper_methods import _kmeans_plus_plus as kpp

"""
Defines the numerical error that is accepted to consider a matrix as orthogonal or symmetric.
"""
_ACCEPTED_NUMERICAL_ERROR = 1e-6


def _nrkmeans(X: np.ndarray, n_clusters: list, V: np.ndarray, m: list, P: list, centers: list, mdl_for_noisespace: bool,
              outliers: bool, max_iter: int, threshold_negative_eigenvalue: float, max_distance: float,
              precision: float, random_state: np.random.RandomState, debug: bool) -> (
        np.ndarray, list, np.ndarray, list, list, list, list):
    """
    Start the actual NrKmeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : list
        list containing number of clusters for each subspace
    V : np.ndarray
        the orthonormal rotation matrix. Can be None
    m : list
        list containing the dimensionalities for each subspace. Can be None
    P : list
        list containing projections (ids of corresponding dimensions) for each subspace. Can be None
    centers : list
        list containing the cluster centers for each subspace. Can be None
    mdl_for_noisespace : bool
        defines if MDL should be used to identify noise space dimensions instead of only considering negative eigenvalues
    outliers : bool
        defines if outliers should be identified through MDL
    max_iter : int
        maximum number of iterations for the algorithm
    threshold_negative_eigenvalue : float
        threshold to consider an eigenvalue as negative. Used for the update of the subspace dimensions
    max_distance : float
        distance used to encode cluster centers and outliers. Only relevant if a MDL strategy is used
    precision : float
        precision used to convert probability densities to actual probabilities. Only relevant if a MDL strategy is used
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (np.ndarray, list, np.ndarray, list, list, list, list)
        The labels,
        The cluster centers,
        The orthonormal rotation matrix,
        The dimensionalities of the subpsaces,
        The projections,
        The number of clusters for each subspace (usually the same as the input),
        The scatter matrices
    """
    V, m, P, centers, subspaces, labels, scatter_matrices = \
        _initialize_nrkmeans_parameters(
            X, n_clusters, V, m, P, centers, mdl_for_noisespace, outliers, max_iter, random_state)
    # Check if labels stay the same (break condition)
    old_labels = None
    n_outliers = np.zeros(subspaces, dtype=int)
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Execute basic kmeans steps
        for i in range(subspaces):
            # Center and scatter matrix stays the same for the noise space if not outliers are present
            if n_clusters[i] != 1 or iteration == 0 or n_outliers[i] > 0:
                # Assign each point to closest cluster center
                labels[:, i] = _assign_labels(X, V, centers[i], P[i])
                # Update centers and scatter matrices depending on cluster assignments
                centers[i], scatter_matrices[i] = _update_centers_and_scatter_matrices(X, n_clusters[i], labels[:, i])
                # Remove empty clusters
                n_clusters[i], centers[i], scatter_matrices[i], labels[:, i] = _remove_empty_cluster(n_clusters[i],
                                                                                                     centers[i],
                                                                                                     scatter_matrices[
                                                                                                         i],
                                                                                                     labels[:, i],
                                                                                                     debug)
            # (Optional) Check for outliers
            if outliers:
                labels[:, i], n_outliers[i] = _check_for_outliers(X, V, centers[i], labels[:, i],
                                                                  scatter_matrices[i], m[i], P[i],
                                                                  max_distance)
                # Again update centers and scatter matrices
                centers[i], scatter_matrices[i] = _update_centers_and_scatter_matrices(X, n_clusters[i], labels[:, i])
        # Check if labels have not changed
        if _are_labels_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()
        # Update rotation for each pair of subspaces
        for i in range(subspaces - 1):
            for j in range(i + 1, subspaces):
                # Do rotation calculations
                P_1_new, P_2_new, V_new = _update_rotation(X, V, i, j, n_clusters, P, scatter_matrices,
                                                           threshold_negative_eigenvalue,
                                                           mdl_for_noisespace, outliers, n_outliers,
                                                           max_distance, precision)
                # Update V, m, P
                m[i] = len(P_1_new)
                m[j] = len(P_2_new)
                P[i] = P_1_new
                P[j] = P_2_new
                V = V_new
        # Handle empty subspaces (no dimensionalities left) -> Should be removed
        subspaces, n_clusters, m, P, centers, labels, scatter_matrices = _remove_empty_subspace(n_clusters, m, P,
                                                                                                centers, labels,
                                                                                                scatter_matrices, debug)
    if debug:
        print("[NrKmeans] Converged in iteration " + str(iteration + 1))
    # Return relevant values
    return labels, centers, V, m, P, n_clusters, scatter_matrices


def _initialize_nrkmeans_parameters(X: np.ndarray, n_clusters: list, V: np.ndarray, m: list, P: list, centers: list,
                                    mdl_for_noisespace: bool, outliers: bool, max_iter: int,
                                    random_state: np.random.RandomState) -> (
        np.ndarray, list, list, list, np.random.RandomState, int, np.ndarray, list, float, float):
    """
    Initialize the input parameters of NrKmeans. This means that all input values which are None must be defined.
    Also all input parameters which are not None must be checked, if a correct execution is possible.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : list
        list containing number of clusters for each subspace
    V : np.ndarray
        the orthonormal rotation matrix. Can be None
    m : list
        list containing the dimensionalities for each subspace. Can be None
    P : list
        list containing projections (ids of corresponding dimensions) for each subspace. Can be None
    centers : list
        list containing the cluster centers for each subspace. Can be None
    mdl_for_noisespace : bool
        defines if MDL should be used to identify noise space dimensions instead of only considering negative eigenvalues
    outliers : bool
        defines if outliers should be identified through MDL
    max_iter : int
        maximum number of iterations for the algorithm
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int

    Returns
    -------
    tuple : (np.ndarray, list, list, list, np.random.RandomState, int, np.ndarray, list, float, float)
        The initial orthonormal rotation matrix,
        The initial dimensionalities of the subpsaces,
        The initial projections,
        The initial cluster centers,
        The random state,
        The number of subspaces (extracted from n_clusters),
        The initial empty numpy array for the labels,
        The initial empty list for the scatter matrices,
        The max_distance value will be equal to the maximum distance between objects within the dataset,
        The precision will be equal to the average minimum feature-wise distance between two objects
    """
    data_dimensionality = X.shape[1]
    # Check if n_clusters is a list
    if not type(n_clusters) is list:
        raise ValueError(
            "Number of clusters must be specified for each subspace_nr and therefore be a list.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains negative values
    if len([x for x in n_clusters if x < 1]) > 0:
        raise ValueError(
            "Number of clusters must not contain negative values or 0.\nYour input:\n" + str(
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
            choices = random_state.choice(subspaces, data_dimensionality - np.sum(m), replace=False)
            for choice in choices:
                m[choice] += 1
    # If m is None but P is defined use P's dimensionality
    elif m is None:
        m = [len(x) for x in P]
    if not type(m) is list or not len(m) is subspaces:
        raise ValueError("A dimensionality list m must be specified for each subspace_nr.\nYour input:\n" + str(m))
    # Calculate projections P
    if P is None:
        possible_projections = list(range(data_dimensionality))
        P = []
        for dimensionality in m:
            choices = random_state.choice(possible_projections, dimensionality, replace=False)
            P.append(choices)
            possible_projections = list(set(possible_projections) - set(choices))
    if not type(P) is list or not len(P) is subspaces:
        raise ValueError("Projection lists must be specified for each subspace_nr.\nYour input:\n" + str(P))
    else:
        # Check if the length of entries in P matches values of m
        used_dimensionalities = []
        for i, dimensionality in enumerate(m):
            used_dimensionalities.extend(P[i])
            if not len(P[i]) == dimensionality:
                raise ValueError(
                    "Values for dimensionality m and length of projection list P do not match.\nDimensionality m:\n" + str(
                        dimensionality) + "\nDimensionality P:\n" + str(len(P[i])))
        # Check if every dimension in considered in P
        if sorted(used_dimensionalities) != list(range(data_dimensionality)):
            raise ValueError("Projections P must include all dimensionalities.\nYour used dimensionalities:\n" + str(
                used_dimensionalities))
    # Define initial cluster centers with kmeans++ for each subspace_nr
    if centers is None:
        centers = [kpp(X, k, row_norms(X, squared=True), random_state=random_state) for k in n_clusters]
    if not type(centers) is list or not len(centers) is subspaces:
        raise ValueError("Cluster centers must be specified for each subspace_nr.\nYour input:\n" + str(centers))
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
            "Max_iter must be an integer greater than 0. Your Max_iter:\n" + str(max_iter))
    # Check outliers value
    if type(outliers) is not bool:
        raise ValueError(
            "outliers must be a boolean. Your input:\n" + str(outliers))
    # Check mdl_for_noisespace value
    if type(mdl_for_noisespace) is not bool:
        raise ValueError(
            "mdl_for_noisespace must be a boolean. Your input:\n" + str(mdl_for_noisespace))
    # Initial labels and scatter matrices
    labels = np.zeros((X.shape[0], subspaces), dtype=np.int32)
    scatter_matrices = [None] * subspaces
    # Check if n_clusters contains more than one noise space
    nr_noise_spaces = len([x for x in n_clusters if x == 1])
    if nr_noise_spaces > 1:
        raise ValueError(
            "Only one subspace_nr can be the noise space (number of clusters = 1).\nYour input:\n" + str(n_clusters))
    # Check if noise space is not the last member in n_clusters
    if 1 in n_clusters and n_clusters[-1] != 1:
        raise ValueError(
            "Noise space (number of clusters = 1) must be the last entry in n_clusters.\nYour input:\n" + str(
                n_clusters))
    return V, m, P, centers, subspaces, labels, scatter_matrices


def _assign_labels(X: np.ndarray, V: np.ndarray, centers_subspace: np.ndarray, P_subspace: np.ndarray) -> np.ndarray:
    """
    Assign each point in each subspace to its nearest cluster center.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    V : np.ndarray
        the orthonormal rotation matrix
    centers_subspace : np.ndarray
        the cluster centers in this subspace
    P_subspace : np.ndarray
        the relevant dimensions (projections) in this subspace

    Returns
    -------
    labels : np.ndarray
        The updated cluster labels in this subspace
    """
    cropped_V = V[:, P_subspace]
    cropped_X = np.matmul(X, cropped_V)
    cropped_centers = np.matmul(centers_subspace, cropped_V)
    # Find nearest center
    labels, _ = pairwise_distances_argmin_min(X=cropped_X, Y=cropped_centers, metric='euclidean',
                                              metric_kwargs={'squared': True})
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32)
    return labels


def _update_centers_and_scatter_matrices(X: np.ndarray, n_clusters_subspace: int, labels_subspace: np.ndarray) -> (
        np.ndarray, np.ndarray):
    """
    Update the cluster centers within this subspace depending on the labels of the data points. Also updates the
    scatter matrix of each cluster by summing up the outer product of the distance between each point and its center.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters_subspace : int
        number of clusters in this subspace
    labels_subspace : np.ndarray
        the cluster labels in this subspace

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The updated cluster centers,
        The updated scatter matrices (one scatter matrix for each cluster)
    """
    # Create empty matrices
    centers = np.zeros((n_clusters_subspace, X.shape[1]))
    scatter_matrices = np.zeros((n_clusters_subspace, X.shape[1], X.shape[1]))
    # Update cluster parameters
    for center_id in range(n_clusters_subspace):
        # Get points in this cluster
        points_in_cluster = labels_subspace == center_id
        if np.sum(points_in_cluster) == 0:
            centers[center_id] = np.nan
            continue
        # Update center
        centers[center_id] = np.mean(X[points_in_cluster], axis=0)
        # Update scatter matrix
        centered_points = X[points_in_cluster] - centers[center_id]
        scatter_matrices[center_id] = np.matmul(centered_points.T, centered_points)
    return centers, scatter_matrices


def _remove_empty_cluster(n_clusters_subspace: int, centers_subspace: np.ndarray, scatter_matrices_subspace: np.ndarray,
                          labels_subspace: np.ndarray, debug: bool) -> (int, np.ndarray, np.ndarray, np.ndarray):
    """
    Check if a cluster got lost after label assignment and center update. Empty clusters will be
    removed for the following rotation. Therefore, all necessary lists will be updated.

    Parameters
    ----------
    n_clusters_subspace : int
        number of clusters in this subspace
    centers_subspace : np.ndarray
        the cluster centers in this subspace
    scatter_matrices_subspace : np.ndarray
        the scatter matrices of this subspace
    labels_subspace : np.ndarray
        the cluster labels in this subspace
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray, np.ndarray)
        The updated number of clusters,
        The updated cluster centers.
        The updated scatter matrices,
        The updated cluster labels
    """
    # Check if any cluster is lost
    if np.any(np.isnan(centers_subspace)):
        # Get ids of lost clusters
        empty_clusters = np.where(np.any(np.isnan(centers_subspace), axis=1))[0]
        if debug:
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


def _update_rotation(X: np.ndarray, V: np.ndarray, first_index: int, second_index: int, n_clusters: list, P: list,
                     scatter_matrices: list, threshold_negative_eigenvalue: float,
                     mdl_for_noisespace: bool, outliers: bool, n_outliers: np.ndarray, max_distance: float,
                     precision: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Update the orthonormal rotation matrix and the subspace projections.
    This happens in a pairwise fashion by considering just two subspaces at a time.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    V : np.ndarray
        the orthonormal rotation matrix
    first_index : int
        index of the first subspace
    second_index : int
        index of the second subspace (in contrast to the first_index this can be the noise space)
    n_clusters : list
        list containing number of clusters for each subspace
    P : list
        list containing projections (ids of corresponding dimensions) for each subspace
    scatter_matrices : list
        the scatter matrices of each subspace
    threshold_negative_eigenvalue : float
        threshold to consider an eigenvalue as negative. Used for the update of the subspace dimensions
    mdl_for_noisespace : bool
        defines if MDL should be used to identify noise space dimensions instead of only considering negative eigenvalues
    outliers : bool
        defines if outliers should be identified through MDL
    n_outliers : np.ndarray
        number of outliers in each subspace
    max_distance : float
        distance used to encode cluster centers and outliers. Only relevant if a MDL strategy is used
    precision : float
        precision used to convert probability densities to actual probabilities. Only relevant if a MDL strategy is used

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray)
        The new projections for the first subspace,
        The new projections for the second subspace,
        The new orthonormal rotation matrix
    """
    # Check if second subspace_nr is the noise space
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
        raise Exception("New V is not orthogonal.\nNew V:\n" + str(V_new))
    # Use number of negative eigenvalues to get new projections
    n_negative_e = len(e[e < 0])
    if is_noise_space:
        if mdl_for_noisespace:
            P_1_new, P_2_new = _compare_possible_splits(X, V_new, first_index, second_index,
                                                        n_negative_e, P_combined, n_clusters,
                                                        scatter_matrices, outliers, n_outliers,
                                                        max_distance, precision)
        else:
            n_negative_e = len(e[e < threshold_negative_eigenvalue])
            P_1_new, P_2_new = _update_projections(P_combined, n_negative_e)
    else:
        P_1_new, P_2_new = _update_projections(P_combined, n_negative_e)
    # Return new dimensionalities, projections and V
    return P_1_new, P_2_new, V_new


def _get_cost_function_of_subspace(cropped_V: np.ndarray, scatter_matrices_subspace: np.ndarray) -> float:
    """
    Calculate the result of the NrKmeans loss function for a certain subspace.
    Depends on the rotation and the scatter matrices. Calculates:
    P^T*V^T*S*V*P

    Parameters
    ----------
    cropped_V : np.ndarray
        cropped orthonormal rotation matrix
    scatter_matrices_subspace : np.ndarray
        the scatter matrices of this subspace

    Returns
    -------
    costs : float
        The NrKmeans loss for this subspace
    """
    scatter_matrix = np.sum(scatter_matrices_subspace, 0)
    costs = np.trace(np.matmul(np.matmul(cropped_V.transpose(), scatter_matrix),
                               cropped_V))
    return costs


def _get_total_cost_function(V: np.ndarray, P: list, scatter_matrices: list) -> float:
    """
    Calculate the sum of the results of the NrKmeans loss function for each subspaces.
    Calls _get_cost_function_of_subspace for each subspace and sums up the results.
    Depends on the rotation, the projections and the scatter matrices. Calculates:
    P_1^T*V^T*S_1*V*P_1 + P_2^T*V^T*S_2*V*P_2 + ...

    Parameters
    ----------
    V : np.ndarray
        the orthonormal rotation matrix
    P : list
        list containing projections (ids of corresponding dimensions) for each subspace
    scatter_matrices : list
        the scatter matrices of each subspace

    Returns
    -------
    costs : float
        The total NrKmeans loss
    """
    costs = np.sum(
        [_get_cost_function_of_subspace(V[:, P[i]], scatter) for i, scatter in enumerate(scatter_matrices)])
    return costs


def _create_full_rotation_matrix(dimensionality: int, P_combined: np.ndarray, V_C: np.ndarray) -> np.ndarray:
    """
    Create full rotation matrix out of the found eigenvectors. Set diagonal to 1 and overwrite columns and rows occurring
    in P_combined (considering the order) with the values from V_C. All other values should be 0.

    Parameters
    ----------
    dimensionality : int
        dimensionality of the full rotation matrix (equals the umber of features in the data set)
    P_combined : np.ndarray
        combined projections of the two subspaces
    V_C : np.ndarray
        the calculated eigenvectors

    Returns
    -------
    V_F : np.ndarray
        the new full-dimensional rotation matrix
    """
    V_F = np.identity(dimensionality)
    V_F[np.ix_(P_combined, P_combined)] = V_C
    return V_F


def _update_projections(P_combined: np.ndarray, n_negative_e: int) -> (np.ndarray, np.ndarray):
    """
    Create the new projections for the two subspaces. First subspace gets as many projections as there are negative
    eigenvalues. Second subspace gets all other projections in reversed order.

    Parameters
    ----------
    P_combined : np.ndarray
        combined projections of the two subspaces
    n_negative_e : int
        number of negative eigenvalues

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The new projections for the first subspace,
        The new projections for the second subspace
    """
    P_1_new = np.array([P_combined[x] for x in range(n_negative_e)], dtype=int)
    P_2_new = np.array([P_combined[x] for x in reversed(range(n_negative_e, len(P_combined)))], dtype=int)
    return P_1_new, P_2_new


def _remove_empty_subspace(n_clusters: list, m: list, P: list, centers: list, labels: np.ndarray,
                           scatter_matrices: list, debug: bool) -> (int, list, list, list, list, np.ndarray, list):
    """
    Check if any empty subspaces occurre after rotating and rearranging the dimensionalities. Empty subspaces will be
    removed for the next iteration. Therefore all necessary lists will be updated.

    Parameters
    ----------
    n_clusters : list
        list containing number of clusters for each subspace
    m : list
        list containing the dimensionalities for each subspace
    P : list
        list containing projections (ids of corresponding dimensions) for each subspace
    centers : list
        list containing the cluster centers for each subspace
    labels : np.ndarray
        the cluster labels of each subspace
    scatter_matrices : list
        the scatter matrices of each subspace
    debug : bool

    Returns
    -------
    tuple : (int, list, list, list, list, np.ndarray, list)
        The number of subspaces,
        The number of clusters per subspace,
        The dimensionality of each subspace,
        The projections of each subspace,
        The cluster centers of each subspace,
        The cluster labels,
        The scatter matrices of each subspace
    """
    if 0 in m:
        np_m = np.array(m)
        empty_spaces = np.where(np_m == 0)[0]
        if debug:
            print("[NrKmeans] ATTENTION: Subspaces were lost! Number of lost subspaces: " + str(
                len(empty_spaces)) + " out of " + str(len(m)))
        n_clusters = [x for i, x in enumerate(n_clusters) if i not in empty_spaces]
        m = [x for i, x in enumerate(m) if i not in empty_spaces]
        P = [x for i, x in enumerate(P) if i not in empty_spaces]
        centers = [x for i, x in enumerate(centers) if i not in empty_spaces]
        labels = np.delete(labels, empty_spaces, axis=1)
        scatter_matrices = [x for i, x in enumerate(scatter_matrices) if i not in empty_spaces]
    return len(n_clusters), n_clusters, m, P, centers, labels, scatter_matrices


def _is_matrix_orthogonal(matrix: np.ndarray) -> bool:
    """
    Check whether a matrix is orthogonal by comparing the multiplication of the matrix and its transpose with
    the identity matrix.
    Uses the _ACCEPTED_NUMERICAL_ERROR as defined in this file to account for numerical inaccuracies.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix

    Returns
    -------
    orthogonal : bool
        True if matrix is orthogonal
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    matrix_product = np.matmul(matrix, matrix.transpose())
    orthogonal = np.allclose(matrix_product, np.identity(matrix.shape[0]), atol=_ACCEPTED_NUMERICAL_ERROR)
    return orthogonal


def _is_matrix_symmetric(matrix: np.ndarray) -> bool:
    """
    Check whether a matrix is symmetric by comparing the matrix with its transpose.
    Uses the _ACCEPTED_NUMERICAL_ERROR as defined in this file to account for numerical inaccuracies.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix

    Returns
    -------
    symmetric : bool
        True if matrix is symmetric
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    symmetric = np.allclose(matrix, matrix.T, atol=_ACCEPTED_NUMERICAL_ERROR)
    return symmetric


def _are_labels_equal(labels_new: np.ndarray, labels_old: np.ndarray) -> bool:
    """
    Check if the old labels and new labels are equal. Therefore check the nmi for each subspace. If all are 1, labels
    have not changed.

    Parameters
    ----------
    labels_new : np.ndarray
        The new cluster labels
    labels_old : np.ndarray
        The old cluster labels

    Returns
    -------
    labels_equal : bool
        True if labels for all subspaces are the same
    """
    if labels_new is None or labels_old is None or labels_new.shape[1] != labels_old.shape[1]:
        return False
    labels_equal = all(
        [nmi(labels_new[:, i], labels_old[:, i], average_method="arithmetic") == 1 for i in range(labels_new.shape[1])])
    return labels_equal


"""
==================== NrKmeans Object ====================
"""


class NrKmeans(BaseEstimator, ClusterMixin):
    """
    The Non-Redundant Kmeans (NrKmeans) algorithm.
    The algorithm will search for the optimal cluster subspaces and assignments
    depending on the input number of clusters and subspaces. The number of subspaces will automatically be traced by the
    length of the input n_clusters array.

    This implementation includes some extensions from 'Automatic Parameter Selection for Non-Redundant Clustering'.

    Parameters
    ----------
    n_clusters : list
        list containing number of clusters for each subspace
    V : np.ndarray
        the initial orthonormal rotation matrix (default: None)
    m : list
        list containing the initial dimensionalities for each subspace (default: None)
    P : list
        list containing the initial projections (ids of corresponding dimensions) for each subspace (default: None)
    cluster_centers : list
        list containing the initial cluster centers for each subspace (default: None)
    mdl_for_noisespace : bool
        defines if MDL should be used to identify noise space dimensions instead of only considering negative eigenvalues (default: False)
    outliers : bool
        defines if outliers should be identified through MDL (default: False)
    max_iter : int
        maximum number of iterations for the algorithm (default: 300)
    n_init : int
        number of times NrKmeans is executed using different seeds. The final result will be the one with lowest costs.
        Costs can be the standard NrKmeans costs or MDL costs (defined by the cost_type parameter) (default: 1)
    cost_type : str
        Can be "default" or "mdl" and defines whether the the standard NrKmeans cost function or MDL costs should be considered to identify the best result.
        Only relevant if n_init is larger than 1 (default: "default")
    threshold_negative_eigenvalue : float
        threshold to consider an eigenvalue as negative. Used for the update of the subspace dimensions (default: -1e-7)
    max_distance : float
        distance used to encode cluster centers and outliers. Only relevant if a MDL strategy is used (default: None)
    precision : float
        precision used to convert probability densities to actual probabilities. Only relevant if a MDL strategy is used (default: None)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    scatter_matrices_ : list
        The final scatter matrices

    References
    ----------
    Mautz, Dominik, et al. "Discovering non-redundant k-means clusterings in optimal subspaces."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.

    and

    Leiber, Collin, et al. "Automatic Parameter Selection for Non-Redundant Clustering."
    Proceedings of the 2022 SIAM International Conference on Data Mining (SDM).
    Society for Industrial and Applied Mathematics, 2022.
    """

    def __init__(self, n_clusters: list, V: np.ndarray = None, m: list = None, P: list = None,
                 cluster_centers: list = None, mdl_for_noisespace: bool = False, outliers: bool = False,
                 max_iter: int = 300, n_init: int = 1, cost_type: str = "default",
                 threshold_negative_eigenvalue: float = -1e-7, max_distance: float = None, precision: float = None,
                 random_state: np.random.RandomState = None, debug: bool = False):
        # Fixed attributes
        self.input_n_clusters = n_clusters.copy()
        self.max_iter = max_iter
        self.n_init = n_init
        self.cost_type = cost_type
        self.threshold_negative_eigenvalue = threshold_negative_eigenvalue
        self.mdl_for_noisespace = mdl_for_noisespace
        self.outliers = outliers
        self.max_distance = max_distance
        self.precision = precision
        self.debug = debug
        self.random_state = check_random_state(random_state)
        # Variables
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        self.V = V
        self.m = m
        self.P = P

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'NrKmeans':
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
        self : NrKmeans
            this instance of the NrKmeans algorithm
        """
        cost_type = self.cost_type.lower()
        assert cost_type in ["default", "mdl"], "cost_type must be 'default' or 'mdl'"
        # precision and max_distance are constant across all executions. Therefore, define those parameters here
        if (self.mdl_for_noisespace or self.outliers) and self.max_distance is None:
            self.max_distance = np.max(pdist(X))
        if self.mdl_for_noisespace and self.precision is None:
            self.precision = _get_precision(X)
        all_random_states = self.random_state.choice(10000, self.n_init, replace=False)
        # Get best result
        best_costs = np.inf
        for i in range(self.n_init):
            local_random_state = check_random_state(all_random_states[i])
            labels, centers, V, m, P, n_clusters, scatter_matrices = _nrkmeans(X, self.n_clusters, self.V, self.m,
                                                                               self.P, self.cluster_centers,
                                                                               self.mdl_for_noisespace,
                                                                               self.outliers, self.max_iter,
                                                                               self.threshold_negative_eigenvalue,
                                                                               self.max_distance, self.precision,
                                                                               local_random_state, self.debug)
            if cost_type == "default":
                costs = _get_total_cost_function(V, P, scatter_matrices)
            else:  # in case of cost_type == "mdl"
                costs, _, _ = _mdl_costs(X, n_clusters, m, P, V, scatter_matrices, labels,
                                         self.outliers, self.max_distance, self.precision)
            if costs < best_costs:
                best_costs = costs
                # Update class variables
                self.labels_ = labels
                self.cluster_centers = centers
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
    #     :return: Array with the labels of the points for each subspace_nr
    #     """
    #     # Check if Nr-Kmeans model was created
    #     if self.labels is None:
    #         raise Exception("The NrKmeans algorithm has not run yet. Use the fit() function first.")
    #     predicted_labels = [None] * len(self.n_clusters)
    #     # Get labels for each subspace_nr
    #     for sub in range(len(self.n_clusters)):
    #         # Predict the labels
    #         predicted_labels[sub] = _assign_labels(X, self.V, self.centers[sub], self.P[sub])
    #         # Check for outliers
    #         if self.outliers:
    #             n_points = X.shape[0]
    #             # Get basic cluster parameters for this subspace_nr
    #             rotation = self.V[:, self.P[sub]]
    #             # Get original costs encoding the points
    #             sum_scatter_matrices = np.sum(self.scatter_matrices[sub], 0)
    #             cluster_costs = mdl._calculate_single_cluster_costs_single_variance(rotation, n_points,
    #                                                                                 sum_scatter_matrices, self.m[sub])
    #             # Get costs for single outlier in this subspace_nr
    #             outlier_coding_cost = mdl.single_outlier_costs(X, rotation)
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
    #                 new_pdf_costs = mdl._calculate_single_cluster_costs_single_variance(rotation, n_points + 1,
    #                                                                                     scatter_matrix_cluster_tmp,
    #                                                                                     self.m[sub])
    #                 # Check if coding costs would be lower
    #                 if outlier_coding_cost + cluster_costs < new_pdf_costs:
    #                     predicted_labels[sub][i] = -1
    #     # Return the predicted labels
    #     return predicted_labels

    def transform_full_space(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input dataset with the orthonormal rotation matrix identified by the fit function.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        rotated_data : np.ndarray
            The rotated dataset
        """
        assert self.labels_ is not None, "The NrKmeans algorithm has not run yet. Use the fit() function first."
        rotated_data = np.matmul(X, self.V)
        return rotated_data

    def transform_subspace(self, X: np.ndarray, subspace_index: int) -> np.ndarray:
        """
        Transform the input dataset with the orthonormal rotation matrix identified by the fit function and
        project it into the specified subspace.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        subspace_index : int
            the index of the specific subspace

        Returns
        -------
        rotated_data : np.ndarray
            The rotated and projected dataset
        """
        assert self.labels_ is not None, "The NrKmeans algorithm has not run yet. Use the fit() function first."
        subspace_V = self.V[:, self.P[subspace_index]]
        rotated_data = np.matmul(X, subspace_V)
        return rotated_data

    def have_subspaces_been_lost(self) -> bool:
        """
        Check whether subspaces have been lost during Nr-Kmeans execution.

        Returns
        -------
        lost : bool
             True if at least one subspace has been lost
        """
        lost = len(self.n_clusters) != len(self.input_n_clusters)
        return lost

    def have_clusters_been_lost(self) -> bool:
        """
        Check whether clusters within any subspace have been lost during Nr-Kmeans execution.
        Will also return true if whole subspaces have been lost (check have_subspaces_been_lost())

        Returns
        -------
        lost : bool
            True if at least one cluster has been lost
        """
        lost = not np.array_equal(self.input_n_clusters, self.n_clusters)
        return lost

    def plot_subspace(self, X: np.ndarray, subspace_index: int, labels: np.ndarray = None, plot_centers: bool = False,
                      gt: np.ndarray = None, equal_axis=False) -> None:
        """
        Plot the specified subspace identified by NrKmeans as scatter matrix plot.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        subspace_index : int
            the index of the specific subspace
        labels : np.ndarray
            the cluster labels used for coloring the plot. If none, the labels identified by the fit() function will be used (default: None)
        plot_centers : bool
            defines whether the cluster centers should be plotted (default: False)
        gt : np.ndarray
            the ground truth labels. In contrast to the labels parameter this will be displayed using different markers instead of colors (default: None)
        equal_axis : bool
            defines whether the axes should be scaled equally
        """
        assert self.labels_ is not None, "The NrKmeans algorithm has not run yet. Use the fit() function first."
        if labels is None:
            labels = self.labels_[:, subspace_index]
        assert X.shape[0] == labels.shape[0], "Number of data objects must match the number of labels."
        plot_scatter_matrix(self.transform_subspace(X, subspace_index), labels,
                            self.transform_subspace(self.cluster_centers[subspace_index], subspace_index) if
                            plot_centers else None, true_labels=gt, equal_axis=equal_axis)

    def calculate_mdl_costs(self, X: np.ndarray) -> (float, float, list):
        """
        Calculate the Mdl Costs of this NrKmeans result.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        tuple : (float, float, list)
            The total costs (global costs + sum of subspace costs),
            The global costs,
            The subspace specific costs (one entry for each subspace)
        """
        assert self.labels_ is not None, "The NrKmeans algorithm has not run yet. Use the fit() function first."
        total_costs, global_costs, all_subspace_costs = _mdl_costs(X, self.n_clusters, self.m, self.P, self.V,
                                                                   self.scatter_matrices_, self.labels_, self.outliers,
                                                                   self.max_distance, self.precision)
        return total_costs, global_costs, all_subspace_costs

    def calculate_cost_function(self) -> float:
        """
        Calculate the result of the NrKmeans loss function. Depends on the rotation and the scatter matrices.
        Calculates for each subspace j:
        P_j^T*V^T*S_j*V*P_j

        Returns
        -------
        costs : float
            The total loss of this NrKmeans object
        """
        assert self.labels_ is not None, "The NrKmeans algorithm has not run yet. Use the fit() function first."
        costs = _get_total_cost_function(self.V, self.P, self.scatter_matrices_)
        return costs


"""
===================== MDL Additions =====================
"""


def _compare_possible_splits(X: np.ndarray, V: np.ndarray, cluster_index: int, noise_index: int, n_negative_e: int,
                             P_combined: np.ndarray, n_clusters: list, scatter_matrices: list,
                             outliers: bool, n_outliers: np.ndarray, max_distance: float,
                             precision: float) -> (np.ndarray, np.ndarray):
    """
    Use MDL to find the best combination of cluster and noise space dimensionality. Try raising number of cluster space
    dimensions until MDL costs increase.
    See 'Automatic Parameter Selection for Non-Redundant Clustering' for more information.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    V : np.ndarray
        the orthonormal rotation matrix
    cluster_index : int
        index of the cluster space
    noise_index : int
        index of the noise space
    n_negative_e : int
        number of negative eigenvalues
    P_combined : np.ndarray
        combined projections of the two subspaces
    n_clusters : list
        list containing number of clusters for each subspace
    scatter_matrices : list
        the scatter matrices of each subspace
    outliers : bool
        defines if outliers should be identified through MDL
    n_outliers : np.ndarray
        number of outliers in each subspace
    max_distance : float
        distance used to encode cluster centers and outliers. Only relevant if a MDL strategy is used
    precision : float
        precision used to convert probability densities to actual probabilities. Only relevant if a MDL strategy is used

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The new projections for the cluster space,
        The new projections for the noise space
    """
    # Find best split of dimensions
    best_costs = np.inf
    best_P_cluster, best_P_noise = _update_projections(P_combined, 0)
    # Try raising number of dimensionalities in the cluster space until costs raise
    for m_cluster in range(1, n_negative_e + 1):
        m_noise = len(P_combined) - m_cluster
        P_cluster, P_noise = _update_projections(P_combined, m_cluster)
        # Get costs for this combination of dimensionalities
        costs = _mdl_m_dependant_subspace_costs(X, V, cluster_index, noise_index, m_cluster, m_noise, P_cluster,
                                                P_noise, scatter_matrices, n_clusters, outliers, n_outliers,
                                                max_distance, precision)
        # If costs are lower, next try. Else break
        if costs < best_costs:
            best_costs = costs
            best_P_cluster = P_cluster.copy()
            best_P_noise = P_noise.copy()
        else:
            break
    return best_P_cluster, best_P_noise


def _mdl_m_dependant_subspace_costs(X: np.ndarray, V: np.ndarray, cluster_index: int, noise_index: int, m_cluster: int,
                                    m_noise: int, P_cluster: np.ndarray, P_noise: np.ndarray,
                                    scatter_matrices: list, n_clusters: list, outliers: bool, n_outliers: np.ndarray,
                                    max_distance: float, precision: float) -> float:
    """
    Get the total costs depending on the subspace dimensions for one cluster space and the noise space.
    Method can be used to determine the best possible number of dimensions to swap from cluster into the noise
    space.
    See 'Automatic Parameter Selection for Non-Redundant Clustering' for more information.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    V : np.ndarray
        the orthonormal rotation matrix
    cluster_index : int
        index of the cluster space
    noise_index : int
        index of the noise space
    m_cluster : int
        dimensionality of the cluster space
    m_noise : int
        dimensionality of the noise space
    P_cluster : np.ndarray
        projections of the cluster space
    P_noise : np.ndarray
        projections of the noise space
    scatter_matrices : list
        the scatter matrices of each subspace
    n_clusters : list
        list containing number of clusters for each subspace
    outliers : bool
        defines if outliers should be identified through MDL
    n_outliers : np.ndarray
        number of outliers in each subspace
    max_distance : float
        distance used to encode cluster centers and outliers. Only relevant if a MDL strategy is used
    precision : float
        precision used to convert probability densities to actual probabilities. Only relevant if a MDL strategy is used

    Returns
    -------
    combined_costs : float
        The combined costs of the cluster and the noise space for this selection of dimensionalities
    """
    n_points = X.shape[0]
    # ==== Costs of cluster space ====
    cropped_V_cluster = V[:, P_cluster]
    # Costs for cluster dimensionality
    cluster_costs = mdl.mdl_costs_integer_value(m_cluster)
    # Costs for centers
    cluster_costs += n_clusters[cluster_index] * _mdl_reference_vector(m_cluster, max_distance, precision)
    # Costs for point encoding
    cluster_costs += mdl.mdl_costs_gmm_single_covariance(m_cluster,
                                                         scatter_matrices[cluster_index],
                                                         n_points - n_outliers[cluster_index], cropped_V_cluster)
    # Costs for outliers
    if outliers:
        cluster_costs += n_outliers[cluster_index] * _mdl_costs_uniform_pdf(m_cluster, max_distance)
    # ==== Costs of noise space ====
    cropped_V_noise = V[:, P_noise]
    # Costs for noise dimensionality
    noise_costs = mdl.mdl_costs_integer_value(m_noise)
    # Costs for centers
    noise_costs += n_clusters[noise_index] * _mdl_reference_vector(m_noise, max_distance, precision)
    # Costs for point encoding
    noise_costs += mdl.mdl_costs_gmm_single_covariance(m_noise,
                                                       scatter_matrices[noise_index],
                                                       n_points - n_outliers[noise_index], cropped_V_noise)
    # Costs for outliers
    if outliers:
        noise_costs += n_outliers[noise_index] * _mdl_costs_uniform_pdf(m_noise, max_distance)
    # Return costs
    combined_costs = cluster_costs + noise_costs
    return combined_costs


def _check_for_outliers(X: np.ndarray, V: np.ndarray, centers_subspace: np.ndarray, labels_subspace: np.ndarray,
                        scatter_matrices_subspace: np.ndarray, m_subspace: int, P_subspace: np.ndarray,
                        max_distance: float) -> (np.ndarray, int):
    """
    Check for each point if it should be interpreted as an outlier in this subspace. Outliers are defined by the cost
    difference when this point is removed from its cluster. If it is cheaper to encode the point separately it is an outlier.
    For each outlier the label will be set to -1. Afterwards, the cluster centers will be updated and the distance
    of this point to its old center will be subtracted from the scatter matrix.
    See 'Automatic Parameter Selection for Non-Redundant Clustering' for more information.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    V : np.ndarray
        the orthonormal rotation matrix
    centers_subspace : np.ndarray
        the cluster centers in this subspace
    labels_subspace : np.ndarray
        the cluster labels in this subspace
    scatter_matrices_subspace : np.ndarray
        the scatter matrices of this subspace
    m_subspace : int
        the dimensionality of this subspace
    P_subspace : np.ndarray
        the relevant dimensions (projections) in this subspace
    max_distance : float
        distance used to encode the outliers

    Returns
    -------
    tuple : (np.ndarray, int)
        The new cluster labels for this subspace,
        The number of outliers in this subspace
    """
    cropped_V = V[:, P_subspace]
    n_points = X.shape[0]
    # Copy labels to update theses based on new outliers
    labels_subspace_copy = labels_subspace.copy()
    # Calculate points distances to respective centers
    cropped_X = np.matmul(X, cropped_V)
    cropped_centers = np.matmul(centers_subspace, cropped_V)
    sum_scatter_matrices = np.sum(scatter_matrices_subspace, 0)
    cropped_scatter_matrix = np.matmul(np.matmul(cropped_V.transpose(), sum_scatter_matrices), cropped_V)
    differences_per_dim = np.power(cropped_X - cropped_centers[labels_subspace], 2)
    differences_sum = np.sum(differences_per_dim, axis=1)
    # Get costs of a single outlier
    outlier_coding_cost = _mdl_costs_uniform_pdf(m_subspace, max_distance)
    outlier_coding_cost += np.log2(n_points) - np.log2(len(centers_subspace))
    # Get trace of cropped scatter matrix
    original_trace = np.trace(cropped_scatter_matrix)
    # Threshold of outlier costs
    outlier_threshold = original_trace * (1 - (n_points - 1) / n_points * np.exp(-1 / (n_points - 1) * (
            2 * np.log(2) * outlier_coding_cost / m_subspace - 1 - np.log(
        2 * np.pi / m_subspace / n_points) - np.log(original_trace))))
    # Get outliers
    is_outlier = differences_sum > outlier_threshold
    # Get number of outliers and change labels
    n_outliers_total = np.sum(is_outlier)
    labels_subspace_copy[is_outlier] = -1
    # Revert changes for clusters that are now empty
    for i in range(len(centers_subspace)):
        if np.sum(labels_subspace_copy == i) == 0:
            original_ids = labels_subspace == i
            n_outliers_total -= np.sum(original_ids)
            labels_subspace_copy[original_ids] = i
    return labels_subspace_copy, n_outliers_total


def _mdl_costs(X: np.ndarray, n_clusters: list, m: list, P: list, V: np.ndarray, scatter_matrices: list,
               labels: np.ndarray, outliers: bool, max_distance: float, precision: float) -> (float, float, list):
    """
    Calculate the total mdl costs of a non-redundant clustering found by NrKmeans.
    Total costs consists of global costs which describe the whole system (e.g. number of subspaces)
    and separate costs for each subspace. This include the exact dimensionalities of the subspaces, number of clusters,
    the centers, cluster assignments, cluster variances and coding costs for each point within a cluster and for each
    outlier.
    See 'Automatic Parameter Selection for Non-Redundant Clustering' for more information.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : list
        list containing number of clusters for each subspace
    m : list
        list containing number of dimensionalities for each subspace
    P : list
        list containing projections (ids of corresponding dimensions) for each subspace
    V : np.ndarray
        the orthonormal rotation matrix
    scatter_matrices : list
        list containing all scatter matrices of the subspaces
    labels : np.ndarray
        the cluster labels of each subspace. -1 equals outlier
    outliers : bool
        defines if outliers should be identified through MDL
    max_distance : float
        distance used to encode cluster centers and outliers
    precision : float
        precision used to convert probability densities to actual probabilities

    Returns
    -------
    tuple : (float, float, list)
        The total costs (global costs + sum of subspace costs),
        The global costs,
        The subspace specific costs (one entry for each subspace)
    """
    n_points = X.shape[0]
    n_outliers = 0
    subspaces = len(m)
    if max_distance is None:
        max_distance = np.max(pdist(X))
    if precision is None:
        precision = _get_precision(X)
    # Calculate costs
    global_costs = 0
    # Costs of matrix V
    # global_costs += mdl.mdl_costs_orthogonal_matrix(n_points, mdl.mdl_costs_float_value(n_points))
    # Costs of number of subspaces
    global_costs += mdl.mdl_costs_integer_value(subspaces)
    # Costs for each subspace_nr
    all_subspace_costs = []
    for subspace_nr in range(subspaces):
        cropped_V = V[:, P[subspace_nr]]
        # Calculate costs
        model_costs = 0
        # Costs for dimensionality
        model_costs += mdl.mdl_costs_integer_value(m[subspace_nr])
        # Number of clusters in subspace_nr
        model_costs += mdl.mdl_costs_integer_value(n_clusters[subspace_nr])
        # Costs for cluster centers
        model_costs += n_clusters[subspace_nr] * \
                       _mdl_reference_vector(m[subspace_nr], max_distance, precision)
        # Coding costs for outliers
        outlier_costs = 0
        if outliers:
            # Encode number of outliers
            n_outliers = len(labels[:, subspace_nr][labels[:, subspace_nr] == -1])
            model_costs += mdl.mdl_costs_integer_value(n_outliers)
            # Encode coding costs of outliers
            outlier_costs += n_outliers * np.log2(n_points)
            outlier_costs += n_outliers * _mdl_costs_uniform_pdf(m[subspace_nr], max_distance)
        # Cluster assignment (is 0 for noise space)
        assignment_costs = (n_points - n_outliers) * mdl.mdl_costs_discrete_probability(
            1 / n_clusters[subspace_nr])
        # Subspace Variance costs
        model_costs += mdl.mdl_costs_bic(n_points)
        # Coding costs for each point
        coding_costs = mdl.mdl_costs_gmm_single_covariance(m[subspace_nr],
                                                           scatter_matrices[subspace_nr],
                                                           n_points - n_outliers, cropped_V)
        coding_costs += n_points * _mdl_costs_precision(m[subspace_nr], precision)
        # Save this subspace_nr costs
        all_subspace_costs.append(model_costs + outlier_costs + assignment_costs + coding_costs)
    # return full and single subspace_nr costs
    total_costs = global_costs + sum(all_subspace_costs)
    return total_costs, global_costs, all_subspace_costs


def _mdl_costs_uniform_pdf(m_subspace: int, max_distance: float) -> float:
    """
    Get the MDL costs of an uniform distribution by using a data range defied by the max_distance parameter.
    This uniform distribution will be used for each of the m_subspace dimensions.

    Parameters
    ----------
    m_subspace : int
        the dimensionality of this subspace
    max_distance :
        distance used for the uniform distribution

    Returns
    -------
    costs_uniform_pdf : float
        The costs for this m_subspace-dimensional uniform distribution
    """
    costs_uniform_pdf = m_subspace * -np.log2(1 / max_distance)
    return costs_uniform_pdf


def _mdl_costs_precision(m_subspace: int, precision: float) -> float:
    """
    Calculate the MDL costs that have to be added to transform a probability density to an actual probability.
    Following applies: -log(prob_dens * precision) = -log(prob_dens) - log(precision).
    This precision dependant costs willadded for each of the m_subspace dimensions.

    Parameters
    ----------
    m_subspace : int
        the dimensionality of this subspace
    precision : float
        precision used to convert probability densities to actual probabilities

    Returns
    -------
    costs_precision : float
        The costs for encoding the precision m_subspace times
    """
    costs_precision = m_subspace * -np.log2(precision)
    return costs_precision


def _mdl_reference_vector(m_subspace: int, max_distance: float, precision: float) -> float:
    """
    Calculate the MDL costs an a reference vector (ie a cluster center).
    Therefore, the m_subspace coordinates will be encoded using a uniform distribution with a data range of max_distance.
    Further, we add the costs of the precision to transform the uniform probability densities to actual probabilities.

    Parameters
    ----------
    m_subspace : int
        the dimensionality of this subspace
    max_distance :
        distance used for the uniform distribution
    precision : float
        precision used to convert probability densities to actual probabilities

    Returns
    -------
    costs_reference_vector : float
        MDL costs to encode a reference vector
    """
    costs_uniform_pdf = _mdl_costs_uniform_pdf(m_subspace, max_distance)
    costs_reference_vector = costs_uniform_pdf + _mdl_costs_precision(m_subspace, precision)
    return costs_reference_vector


def _get_precision(X: np.ndarray) -> float:
    """
    Get the default precision. Will be equal to the average minimum feature-wise distance between two objects

    Parameters
    ----------
    X : np.ndarray
        the given data set

    Returns
    -------
    precision : float
        The calculated precision
    """
    precision_list = []
    for i in range(X.shape[1]):
        dist = pdist(X[:, i].reshape((-1, 1)))
        dist_gt_0 = dist[dist > 0]
        if dist_gt_0.size != 0:
            precision_list.append(np.min(dist_gt_0))
    precision = np.mean(precision_list)
    return precision

"""
@authors:
Collin Leiber
"""

import numpy as np
from clustpy.utils import dip_test, dip_pval, dip_pval_gradient
from clustpy.partition import UniDip
from sklearn.decomposition import PCA
from clustpy.partition.dipext import _angle, _n_starting_vectors_default, _ambiguous_modal_triangle_random
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClusterMixin


def _dip_n_sub(X: np.ndarray, significance: float, threshold: float, step_size: float, momentum: float,
               n_starting_vectors: int, add_tails: bool, outliers: bool, consider_duplicates: bool,
               random_state: np.random.RandomState, debug: bool) -> (int, np.ndarray, np.ndarray):
    """
    Start the actual DipNSub clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
            Threshold to decide if the result of the dip-test is unimodal or multimodal
    threshold : float
        Defines the amount of objects that must be contained in multimodal clusters for the algorithm to continue
    step_size : float
        Step size used for gradient descent
    momentum : float
        Momentum used for gradient descent
    n_starting_vectors : int
        The number of starting vectors for gradient descent
    add_tails : bool
        Defines if TailoredDip should try to add tails to the surrounding clusters
    outliers : bool
        Defines if outliers should be identified as described by UniDip
    consider_duplicates : bool
        If multiple instances on the projection axis share a value, the gradient is ambiguous. If those duplicate values should be considered a random instances will be choses for furhter calculations. Beware: The calculation will not be deterministic anymore
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The final number of clusters,
        The labels as identified by DipNSub,
        The resulting feature space (Number of samples x number of components)
    """
    assert significance >= 0 and significance <= 1, "significance must be a value in the range [0, 1]"
    assert threshold >= 0 and threshold <= 1, "threshold must be a value in the range [0, 1]"
    # Initial values
    n_clusters = 1
    labels = np.zeros(X.shape[0], dtype=np.int32)
    subspace = np.zeros((X.shape[0], 0))
    remaining_X = X
    cluster_sizes = np.array([X.shape[0]])
    # Find multimodal axes
    while remaining_X.shape[1] > 0:
        pvalues, projection, projected_data = _find_min_dippvalue_by_grouped_sgd(remaining_X, labels, n_clusters,
                                                                                 step_size, momentum,
                                                                                 n_starting_vectors,
                                                                                 cluster_sizes, consider_duplicates,
                                                                                 random_state, debug)
        # Check how many objects are contained in multimodal axes on the identified axis
        amount_multimodal_objects = np.sum((pvalues < significance) * cluster_sizes) / X.shape[0]
        if amount_multimodal_objects >= threshold:
            if debug:
                print(
                    "amount_multimodal_objects is {0} >= {1}. Use dimension {2} for clustering. pvalues = {3} / cluster_sizes = {4}".format(
                        amount_multimodal_objects, threshold, subspace.shape[1] + 1, pvalues, cluster_sizes))
            n_clusters_old = n_clusters
            for i in range(n_clusters_old):
                # Operation only sensible if pvalue is below significance level
                if pvalues[i] < significance:
                    points_in_cluster = np.where(labels == i)[0]
                    unidip = UniDip(outliers=outliers, significance=significance, add_tails=add_tails,
                                    pval_strategy="function", random_state=random_state)
                    unidip.fit(projected_data[points_in_cluster])
                    # Update cluster sizes
                    for j in range(unidip.n_clusters_):
                        new_clusters_size = unidip.labels_[unidip.labels_ == j].shape[0]
                        if j == 0:
                            cluster_sizes[i] = new_clusters_size
                        else:
                            cluster_sizes = np.append(cluster_sizes, new_clusters_size)
                    # Update labels
                    labels[points_in_cluster[unidip.labels_ == -1]] = -1
                    labels[points_in_cluster[unidip.labels_ == 0]] = i
                    labels[points_in_cluster[unidip.labels_ > 0]] = unidip.labels_[
                                                                        unidip.labels_ > 0] + n_clusters - 1
                    n_clusters += unidip.n_clusters_ - 1
            if debug:
                print("-> new n_clusters:", n_clusters)
            # Add dimension to subspace
            subspace = np.c_[subspace, projected_data]
            if subspace.shape[0] == X.shape[1]:
                break
            # Prepare next iteration
            orthogonal_space, _ = np.linalg.qr(projection.reshape(-1, 1), mode="complete")
            remaining_X = np.matmul(remaining_X, orthogonal_space[:, 1:])
        else:
            if debug:
                print(
                    "amount_multimodal_objects is {0} < {1}. Abort clustering. pvalues = {2} / cluster_sizes = {3}".format(
                        amount_multimodal_objects, threshold, pvalues, cluster_sizes))
            # Never return an empty subspace
            if subspace.shape[1] == 0:
                subspace = np.c_[subspace, projected_data]
            break
    return n_clusters, labels, subspace


def _find_min_dippvalue_by_grouped_sgd(X: np.ndarray, labels: np.ndarray, n_clusters: int, step_size: float,
                                       momentum: float, n_starting_vectors: int,
                                       cluster_sizes: np.ndarray, consider_duplicates: bool,
                                       random_state: np.random.RandomState, debug: bool) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """
    Find the axes with n_starting_vectors highest weighted dip-p-values and start gradient descent from there.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The current cluster labels
    n_clusters : int
        The current number of clusters
    step_size : float
        Step size used for gradient descent
    momentum : float
        Momentum used for gradient descent
    n_starting_vectors : int
        The number of starting vectors for gradient descent
    cluster_sizes : np.ndarray
        List containing the number of samples in each cluster
    consider_duplicates : bool
        If multiple instances on the projection axis share a value, the gradient is ambiguous. If those duplicate values should be considered a random instances will be choses for furhter calculations. Beware: The calculation will not be deterministic anymore
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray)
        The best identified weighted dip-p-values,
        The corresponing projection axis responsible for the dip-p-values,
        The data projected onto that projection axis
    """
    # Get dip-p-value of each cluster on each axis
    axis_dips = [
        np.array([dip_test(X[labels == j, d], just_dip=True, is_data_sorted=False) for j in range(n_clusters)]) for d in
        range(X.shape[1])]
    axis_pvalues = [np.array(
        [dip_pval(d_inner, cluster_sizes[j], pval_strategy="function") for j, d_inner in enumerate(single_axis_dips)])
        for single_axis_dips in axis_dips]
    if X.shape[1] == 1:
        # Return axis_pvales and trivial projection
        return axis_pvalues[0], np.array([1]), X
    # Calculate weighted sum of p-values (use negative dip-value if all p-values are 0)
    sum_weighted_pvalues_per_axis = np.array([
        np.sum(axis_pvalues[j] * cluster_sizes) if np.sum(axis_pvalues[j]) != 0 else -np.sum(
            axis_dips[j] * cluster_sizes) for j in range(len(axis_pvalues))])
    # Sort axes by weighted sum of p-values
    argsorted_dimensions = np.argsort(sum_weighted_pvalues_per_axis)
    min_sum_weighted_pvalues = sum_weighted_pvalues_per_axis[argsorted_dimensions[0]]
    best_pvalues = axis_pvalues[argsorted_dimensions[0]]
    best_projection = np.zeros(X.shape[1])
    best_projection[argsorted_dimensions[0]] = 1
    best_projected_data = X[:, argsorted_dimensions[0]]
    n_equal_results = 1
    # Start from n_starting_vectors features (max is current total number of features)
    n_starting_vectors = min(n_starting_vectors, X.shape[1])
    # Include features from PCA
    pca = PCA(n_starting_vectors)
    pca.fit(X)
    for i in range(n_starting_vectors):
        for mode in ["axis", "pca"]:
            # Initial projection vector
            if mode == "axis":
                start_projection = np.zeros(X.shape[1])
                start_projection[argsorted_dimensions[i]] = 1
            else:
                start_projection = pca.components_[i]
            pvalues, projection, projected_data, sum_weighted_pvalues = _find_min_dippvalue_by_grouped_sgd_with_start(X,
                                                                                                                      labels,
                                                                                                                      n_clusters,
                                                                                                                      start_projection,
                                                                                                                      step_size,
                                                                                                                      momentum,
                                                                                                                      cluster_sizes,
                                                                                                                      consider_duplicates,
                                                                                                                      random_state,
                                                                                                                      debug)
            # In rare cases the weighted p-values can be equal to current best result -> pick better result randomly
            if sum_weighted_pvalues == min_sum_weighted_pvalues:
                n_equal_results += 1
                random_number = random_state.rand()
                if debug and not np.array_equal(projection, best_projection):
                    # Only used for prints
                    picked_pvalues = (pvalues, best_pvalues) if random_number <= 1 / n_equal_results else (
                        best_pvalues, pvalues)
                    picked_projection = (projection, best_projection) if random_number <= 1 / n_equal_results else (
                        best_projection, projection)
                    print(
                        "--> Current and best costs are both {0}. Therefore, pvalues = {1}, projection = {2} was randomly picked instead of pvalues = {3}, projection = {4}.".format(
                            sum_weighted_pvalues, picked_pvalues[0], picked_projection[0], picked_pvalues[1],
                            picked_projection[1]))
            if sum_weighted_pvalues < min_sum_weighted_pvalues or (
                    sum_weighted_pvalues == min_sum_weighted_pvalues and random_number <= 1 / n_equal_results):
                min_sum_weighted_pvalues = sum_weighted_pvalues
                best_pvalues = pvalues
                best_projection = projection
                best_projected_data = projected_data
    return best_pvalues, best_projection, best_projected_data


def _find_min_dippvalue_by_grouped_sgd_with_start(X: np.ndarray, labels: np.ndarray, n_clusters: int,
                                                  projection: np.ndarray, step_size: float, momentum: float,
                                                  cluster_sizes: np.ndarray, consider_duplicates: bool,
                                                  random_state: np.random.RandomState,
                                                  debug: bool) -> (np.ndarray, np.ndarray, np.ndarray, float):
    """
    Perform gradient descent to find the projection vector with the minimum weighted dip-p-value.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The current cluster labels
    n_clusters : int
        The current number of clusters
    projection : np.ndarray
        The starting projection axis
    step_size : float
        Step size used for gradient descent
    momentum : float
        Momentum used for gradient descent
    cluster_sizes : np.ndarray
        List containing the number of samples in each cluster
    consider_duplicates : bool
        If multiple instances on the projection axis share a value, the gradient is ambiguous. If those duplicate values should be considered a random instances will be choses for furhter calculations. Beware: The calculation will not be deterministic anymore
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, float)
        The best identified weighted dip-p-values,
        The corresponing projection axis responsible for the dip-p-values,
        The data projected onto that projection axis,
        The sum of the weighted dip-p-values
    """
    # Initial values
    total_angle = 0
    best_projection = None
    best_projected_data = None
    best_pvalues = None
    direction = np.zeros(X.shape[1])
    min_sum_weighted_pvalues = np.inf
    n_equal_results = 1
    # Perform SGD
    while True:
        # Ensure unit vector
        projection = projection / np.linalg.norm(projection)
        gradient, dip_values, projected_data = _get_min_dippvalue_using_grouped_gradient(X, labels, n_clusters,
                                                                                         projection, cluster_sizes,
                                                                                         consider_duplicates,
                                                                                         random_state)
        # Calculate p-values
        vecotrize_pvalue = np.vectorize(dip_pval, excluded=["pval_strategy"])
        pvalues = vecotrize_pvalue(dip_values, cluster_sizes, pval_strategy="function")
        sum_weighted_pvalues = np.sum(pvalues * cluster_sizes)
        # Use negative dip-value if all p-values are 0
        if sum_weighted_pvalues == 0:
            sum_weighted_pvalues = -np.sum(dip_values * cluster_sizes)
        # In rare cases the weighted p-values can be equal to current best result -> pick better result randomly
        if sum_weighted_pvalues == min_sum_weighted_pvalues:
            n_equal_results += 1
            random_number = random_state.rand()
            if debug and not np.array_equal(projection, best_projection):
                # Only used for prints
                picked_pvalues = (pvalues, best_pvalues) if random_number <= 1 / n_equal_results else (
                    best_pvalues, pvalues)
                picked_projection = (projection, best_projection) if random_number <= 1 / n_equal_results else (
                    best_projection, projection)
                print(
                    "--> Current and best costs are both {0}. Therefore, pvalues = {1}, projection = {2} was randomly picked instead of pvalues = {3}, projection = {4}.".format(
                        sum_weighted_pvalues, picked_pvalues[0], picked_projection[0], picked_pvalues[1],
                        picked_projection[1]))
        if sum_weighted_pvalues < min_sum_weighted_pvalues or (
                sum_weighted_pvalues == min_sum_weighted_pvalues and random_number <= 1 / n_equal_results):
            min_sum_weighted_pvalues = sum_weighted_pvalues
            best_pvalues = pvalues
            best_projection = projection
            best_projected_data = projected_data
        # Update parameters
        direction = momentum * direction + step_size * gradient
        new_projection = projection + direction
        new_angle = _angle(projection, new_projection)
        total_angle += new_angle
        projection = new_projection
        # We converge if the projection vector barely moves anymore and has no intention (momentum) to do so in the future
        if (new_angle <= 0.1 and np.linalg.norm(direction) < 0.1) or total_angle > 360:
            break
    return best_pvalues, best_projection, best_projected_data, min_sum_weighted_pvalues


def _get_min_dippvalue_using_grouped_gradient(X: np.ndarray, labels: np.ndarray, n_clusters: int,
                                              projection_vector: np.ndarray, cluster_sizes: np.ndarray,
                                              consider_duplicates: bool, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """
    Use current projection_vector to calculate the dip-value and a corresponding modal_triangle.
    The modal_triangle is then used to calculate the gradient of the used projection axis.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The current cluster labels
    n_clusters : int
        The current number of clusters
    projection_vector : np.ndarray
        The current projection axis
    cluster_sizes : np.ndarray
        List containing the number of samples in each cluster
    consider_duplicates : bool
        If multiple instances on the projection axis share a value, the gradient is ambiguous. If those duplicate values should be considered a random instances will be choses for furhter calculations. Beware: The calculation will not be deterministic anymore
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray)
        The gradient,
        List containing the dip-value of each cluster,
        The data set projected onto the projection axis
    """
    # Project data (making a univariate sample)
    projected_data = np.matmul(X, projection_vector)
    # Create result objects
    dip_values = np.zeros(n_clusters)
    gradient = np.zeros(X.shape[1])
    for i in range(n_clusters):
        points_in_cluster = (labels == i)
        if np.sum(points_in_cluster) < 4:
            # Clusters can in theory get very small
            continue
        sorted_indices = np.argsort(projected_data[points_in_cluster])
        sorted_projected_data_in_cluster = projected_data[points_in_cluster][sorted_indices]
        dip_value, _, modal_triangle = dip_test(sorted_projected_data_in_cluster, just_dip=False,
                                                is_data_sorted=True)
        if modal_triangle[0] == -1:
            continue
        if consider_duplicates:
            # If duplicate values should be considered, get random ordering of the objects
            sorted_indices = _ambiguous_modal_triangle_random(sorted_projected_data_in_cluster, sorted_indices,
                                                              modal_triangle, random_state)
        # Calculate the partial derivative for all dimensions regarding this cluster
        gradient_tmp = -dip_pval_gradient(X[points_in_cluster], projected_data[points_in_cluster],
                                          sorted_indices, modal_triangle, dip_value)
        dip_values[i] = dip_value
        # Weight gradient by cluster size
        gradient = gradient + cluster_sizes[i] * gradient_tmp
    gradient = gradient / X.shape[0]
    return gradient, dip_values, projected_data


class DipNSub(BaseEstimator, ClusterMixin):
    """
    Execute the Dip`n`Sub clustering procedure.
    It searches for projection axes in which as many samples as possible are part of multimodal clusters.
    Therefore, it uses the gradient of the p-value of the Dip-test to perform stochastic gradient descent.
    On the identified axes TailoredDip, an extension of UniDip, will be executed to assign cluster labels.
    Note, that clusters will always form a hypercube in the resulting subspace.

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal (default: 0.01)
    threshold : float
        Defines the amount of objects that must be contained in multimodal clusters for the algorithm to continue (default: 0.15)
    step_size : float
        Step size used for gradient descent (default: 0.1)
    momentum : float
        Momentum used for gradient descent (default: 0.95)
    n_starting_vectors : int
        The number of starting vectors for gradient descent. Can be None, in that case it will be equal to np.log(data dimensionality) + 1 (default: None)
    add_tails : bool
        Defines if TailoredDip should try to add tails to the surrounding clusters (default: True)
    outliers : bool
        Defines if outliers should be identified as described by UniDip (default: False)
    consider_duplicates : bool
        If multiple instances on the projection axis share a value, the gradient is ambiguous. If those duplicate values should be considered a random instances will be choses for furhter calculations. Beware: The calculation will not be deterministic anymore (default: False)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    subspace_: np.ndarray
        The resulting subspace

    References
    ----------
    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """

    def __init__(self, significance: float = 0.01, threshold: float = 0.15, step_size: float = 0.1,
                 momentum: float = 0.95, n_starting_vectors: int = None, add_tails=True, outliers=False,
                 consider_duplicates: bool = False, random_state: np.random.RandomState | int = None,
                 debug: bool = False):
        self.significance = significance
        self.threshold = threshold
        self.step_size = step_size
        self.momentum = momentum
        self.n_starting_vectors = n_starting_vectors
        self.add_tails = add_tails
        self.outliers = outliers
        self.consider_duplicates = consider_duplicates
        self.random_state = check_random_state(random_state)
        self.debug = debug

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DipNSub':
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
        self : DipNSub
            this instance of the DipNSub algorithm
        """
        if self.n_starting_vectors is None:
            self.n_starting_vectors = _n_starting_vectors_default(X.shape[1])
        n_clusters, labels, subspace = _dip_n_sub(X, significance=self.significance, threshold=self.threshold,
                                                  step_size=self.step_size, momentum=self.momentum,
                                                  n_starting_vectors=self.n_starting_vectors,
                                                  add_tails=self.add_tails, outliers=self.outliers,
                                                  consider_duplicates=self.consider_duplicates,
                                                  random_state=self.random_state, debug=self.debug)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.subspace_ = subspace
        return self

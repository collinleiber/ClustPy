"""
@authors:
Benjamin Schelling and Sam Maurus (original R implementation),
Collin Leiber
"""

from clustpy.utils import dip_test, dip_gradient
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state


def _dip_ext(X: np.ndarray, n_components: int, do_dip_scaling: bool, step_size: float, momentum: float,
             dip_threshold: float, n_starting_vectors: int, ambiguous_triangle_strategy: str,
             random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, list, np.ndarray):
    """
    Start the actual DipExt dimensionality-reduction procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_components : int
        The number of components to extract. Can be None, in that case dip_threshold wil be used to define the number of components
    do_dip_scaling : bool
        If true, the resulting features space will be scaled by performing a min-max normalization for each feature and multiplying this feautre by its dip-value
    step_size : float
        Step size used for gradient descent
    momentum : float
        Momentum used for gradient descent
    dip_threshold : float
        Defines the number of components if n_components is None. If an identified feature has a dip-value below the maximum dip-value times dip_threshold, DipExt will terminate
    n_starting_vectors : int
        The number of starting vectors for gradient descent
    ambiguous_triangle_strategy : str
        The strategy with which to handle an ambiguous modal triangle. Can be 'ignore', 'random' or 'all'.
        In the case of 'random', a valid triangle is created at random.
        In the case of 'all', for each possible triangle the gradient is calculated and it is checked for which gradient the following result looks most promising - this strategy can increase the runtime noticeably
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only used if ambiguous_triangle_strategy is 'random'

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, list, np.ndarray)
        The resulting feature space (Number of samples x number of components),
        The dip-value of each resulting feature,
        List containing the used projection axes,
        The indices of the sorted original dip-values (decreasing), needed to adjust order of the features when using transform()
    """
    assert n_components is None or n_components < X.shape[
        1], "n_components must be None or smaller than the dimensionality of the data set."
    assert dip_threshold <= 1 and dip_threshold >= 0, "dip_threshold must be within [0, 1]"
    assert type(ambiguous_triangle_strategy) is str, "ambiguous_triangle_strategy must be of type str"
    ambiguous_triangle_strategy = ambiguous_triangle_strategy.lower()
    assert ambiguous_triangle_strategy in ["ignore", "random",
                                           "all"], "ambiguous_triangle_strategy must be 'ignore', 'random' or 'all'"
    # Initial values
    subspace = np.zeros((X.shape[0], 0))
    projection_axes = []
    dip_values = []
    max_dip = 0
    remaining_X = X
    # Find multimodal axes
    while True:
        dip_value, projection, projected_data = _find_max_dip_by_sgd(remaining_X, step_size, momentum,
                                                                     n_starting_vectors, ambiguous_triangle_strategy,
                                                                     random_state)
        if dip_value < max_dip * dip_threshold and n_components is None:
            break
        # Always use the highest dip value
        max_dip = max(dip_value, max_dip)
        dip_values.append(dip_value)
        # Make projection orthogonal
        projection_axes.append(projection)
        subspace = np.c_[subspace, projected_data]
        if subspace.shape[1] == X.shape[1] or subspace.shape[1] == n_components:
            break
        # Prepare next iteration
        orthogonal_space, _ = np.linalg.qr(projection.reshape(-1, 1), mode="complete")
        remaining_X = np.matmul(remaining_X, orthogonal_space[:, 1:])
    # Sort features by dip-value
    argsorted_dip = np.argsort(dip_values)[::-1]
    dip_values = np.array(dip_values)[argsorted_dip]
    subspace = subspace[:, argsorted_dip]
    if do_dip_scaling:
        subspace = _dip_scaling(subspace, dip_values)
    return subspace, dip_values, projection_axes, argsorted_dip


def _find_max_dip_by_sgd(X: np.ndarray, step_size: float, momentum: float, n_starting_vectors: int,
                         ambiguous_triangle_strategy: str, random_state: np.random.RandomState) -> (
        float, np.ndarray, np.ndarray):
    """
    Find the axes with n_starting_vectors highest dip-values and start gradient descent from there.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    step_size : float
        Step size used for gradient descent
    momentum : float
        Momentum used for gradient descent
    n_starting_vectors : int
        The number of starting vectors for gradient descent
    ambiguous_triangle_strategy : str
        The strategy with which to handle an ambiguous modal triangle. Can be 'ignore', 'random' or 'all'.
        In the case of 'random', a valid triangle is created at random.
        In the case of 'all', for each possible triangle the gradient is calculated and it is checked for which gradient the following result looks most promising - this strategy can increase the runtime noticeably
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only used if ambiguous_triangle_strategy is 'random'

    Returns
    -------
    tuple : (float, np.ndarray, np.ndarray)
        The highest dip-value found,
        The corresponding projection axis,
        The data set projected onto this projection axis
    """
    # Get dip-value of each axis
    axis_dips = [dip_test(X[:, i], just_dip=True, is_data_sorted=False) for i in range(X.shape[1])]
    if X.shape[1] == 1:
        return axis_dips[0], np.array([1]), X
    # Sort axes by dip-values
    dips_argsorted = np.argsort(axis_dips)[::-1]
    max_dip = axis_dips[dips_argsorted[0]]
    best_projection = np.zeros(X.shape[1])
    best_projection[dips_argsorted[0]] = 1
    best_projected_data = X[:, dips_argsorted[0]]
    # Start from n_starting_vectors features (max is current total number of features)
    n_starting_vectors = min(n_starting_vectors, X.shape[1])
    for i in range(n_starting_vectors):
        # Initial projection vector
        start_projection = np.zeros(X.shape[1])
        start_projection[dips_argsorted[i]] = 1
        dip_value, projection, projected_data = _find_max_dip_by_sgd_with_start(X, start_projection, step_size,
                                                                                momentum, ambiguous_triangle_strategy,
                                                                                random_state)
        if dip_value > max_dip:
            max_dip = dip_value
            best_projection = projection
            best_projected_data = projected_data
    return max_dip, best_projection, best_projected_data


def _find_max_dip_by_sgd_with_start(X: np.ndarray, projection: np.ndarray, step_size: float, momentum: float,
                                    ambiguous_triangle_strategy: str, random_state: np.random.RandomState) -> (
        float, np.ndarray, np.ndarray):
    """
    Perform gradient descent to find the projection vector with the maximum dip-value.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    projection : np.ndarray
        The initial projection vector
    step_size : float
        Step size used for gradient descent
    momentum : float
        Momentum used for gradient descent
    ambiguous_triangle_strategy : str
        The strategy with which to handle an ambiguous modal triangle. Can be 'ignore', 'random' or 'all'.
        In the case of 'random', a valid triangle is created at random.
        In the case of 'all', for each possible triangle the gradient is calculated and it is checked for which gradient the following result looks most promising - this strategy can increase the runtime noticeably
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only used if ambiguous_triangle_strategy is 'random'

    Returns
    -------
    tuple : (float, np.ndarray, np.ndarray)
        The highest dip-value found,
        The corresponding projection axis,
        The data set projected onto this projection axis
    """
    # Initial values
    total_angle = 0
    best_projection = None
    best_projected_data = None
    direction = np.zeros(X.shape[1])
    max_dip = 0
    # Perform SGD
    while True:
        # Ensure unit vector
        projection = projection / np.linalg.norm(projection)
        gradient, dip_value, projected_data = _get_max_dip_using_gradient(X, projection, ambiguous_triangle_strategy,
                                                                          random_state)
        if dip_value > max_dip:
            max_dip = dip_value
            best_projection = projection
            best_projected_data = projected_data
        # Normally there is only one gradient. But there can be multiple if ambiguous_triangle_strategy is 'all'
        if ambiguous_triangle_strategy == "all":
            tmp_max_dip = 0
            tmp_best_result = None
            for tmp_gradient in gradient:
                tmp_direction = momentum * direction + step_size * tmp_gradient
                tmp_projection = projection + tmp_direction
                tmp_projected_data = np.matmul(X, tmp_projection)
                tmp_dip_value = dip_test(tmp_projected_data, just_dip=True, is_data_sorted=False)
                if tmp_dip_value > tmp_max_dip:
                    tmp_max_dip = tmp_dip_value
                    tmp_best_result = (tmp_direction, tmp_projection)
            direction, new_projection = tmp_best_result
        else:
            # Update parameters
            direction = momentum * direction + step_size * gradient
            new_projection = projection + direction
        # Get new angle and new total angle and use new projection for following iteration
        new_angle = _angle(projection, new_projection)
        total_angle += new_angle
        projection = new_projection
        # We converge if the projection vector barely moves anymore and has no intention (momentum) to do so in the future
        if (new_angle <= 0.1 and np.linalg.norm(direction) < 0.1) or total_angle > 360:
            break
    return max_dip, best_projection, best_projected_data


def _get_max_dip_using_gradient(X: np.ndarray, projection_vector: np.ndarray, ambiguous_triangle_strategy: str,
                                random_state: np.random.RandomState) -> (
        np.ndarray, float, np.ndarray):
    """
    Use current projection_vector to calculate the dip value and a corresponding modal_triangle.
    The modal_triangle is then used to calculate the gradient of the used projection axis.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    projection_vector : np.ndarray
        the current projection vector
    ambiguous_triangle_strategy : str
        The strategy with which to handle an ambiguous modal triangle. Can be 'ignore', 'random' or 'all'.
        In the case of 'random', a valid triangle is created at random.
        In the case of 'all', for each possible triangle the gradient is calculated and it is checked for which gradient the following result looks most promising - this strategy can increase the runtime noticeably
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only used if ambiguous_triangle_strategy is 'random'

    Returns
    -------
    tuple : (np.ndarray, float, np.ndarray)
        The gradient of the dip regarding the projection axis (can contain multiple gradients if ambiguous_triangle_strategy is 'all'),
        The dip-value,
        The data set projected onto the current projection axis
    """
    # Project data (making a univariate sample)
    projected_data = np.matmul(X, projection_vector)
    # Sort data
    sorted_indices = np.argsort(projected_data)
    sorted_projected_data = projected_data[sorted_indices]
    # Calculate dip, capturing the output which we need for touching-triangle calculations
    dip_value, _, modal_triangle = dip_test(sorted_projected_data, just_dip=False, is_data_sorted=True)
    if modal_triangle[0] == -1:
        return np.zeros(X.shape[1]), dip_value, projected_data
    if ambiguous_triangle_strategy == "all":
        # Calculate all possible gradients. Beware: in this case, gradient is a list!
        gradient = _ambiguous_modal_triangle_all(X, projected_data, sorted_projected_data, sorted_indices,
                                                 modal_triangle)
    else:
        if ambiguous_triangle_strategy == "random":
            # If duplicate values should be considered, get random reordering of sorted_indices
            sorted_indices = _ambiguous_modal_triangle_random(sorted_projected_data, sorted_indices, modal_triangle,
                                                              random_state)
        # Calculate the gradient
        gradient = dip_gradient(X, projected_data, sorted_indices, modal_triangle)
    return gradient, dip_value, projected_data


"""
Handle ambiguous modal triangle
"""


def _ambiguous_modal_triangle_random(sorted_projected_data: np.ndarray, sorted_indices: np.ndarray,
                                     modal_triangle: tuple, random_state: np.random.RandomState) -> np.ndarray:
    """
    If on a projection axis multiple values share a coordinate, the modal_triangle is ambiguous.
    In this case the random_state will be used to randomly switch the positions of entries that share the coordinate in sorted_indices.
    This has an influence on the following gradient calculation.

    Parameters
    ----------
    sorted_projected_data : np.ndarray
        The data set projected onto the projection axis and sorted afterwards
    sorted_indices : np.ndarray
        The indices of the sorted univariate data set
    modal_triangle : tuple
        The modal triangle as returned by the dip-test
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    sorted_indices : np.ndarray
        The updated indices of the sorted univariate data set
    """
    triangle_possibilities = _get_ambiguous_modal_triangle_possibilities(sorted_projected_data, modal_triangle)
    # Change order of samples with same value using random ordering
    for i in range(len(modal_triangle)):
        # Get random change
        new_modal_triangle_obj = random_state.choice(triangle_possibilities[i])
        if i < 2 and len(triangle_possibilities[i]) > 1 and sorted_projected_data[modal_triangle[i]] == \
                sorted_projected_data[modal_triangle[i + 1]]:
            # Remove value from list since its triangle coordinate should not be used twice
            triangle_possibilities[i + 1].remove(new_modal_triangle_obj)
        original_index = sorted_indices[modal_triangle[i]]
        sorted_indices[modal_triangle[i]] = sorted_indices[new_modal_triangle_obj]
        sorted_indices[new_modal_triangle_obj] = original_index
    return sorted_indices


def _ambiguous_modal_triangle_all(X: np.ndarray, projected_data: np.ndarray, sorted_projected_data: np.ndarray,
                                  sorted_indices: np.ndarray, modal_triangle: tuple) -> np.ndarray:
    """
    If on a projection axis multiple values share a coordinate, the modal_triangle is ambiguous.
    Therefore, all possible gradients will be calculated and returned.
    This allows the following SGD step to be performed for each gradient and the dip value to be calculated.
    The best value is then used for the further SGD process.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    projected_data : np.ndarray
        The univariate projected data set
    sorted_projected_data : np.ndarray
        The data set projected onto the projection axis and sorted afterwards
    sorted_indices : np.ndarray
        The indices of the sorted univariate data set
    modal_triangle : tuple
        The modal triangle as returned by the dip-test

    Returns
    -------
    gradient : np.ndarray
        Array containing all the calculated gradients
    """
    triangle_possibilities = _get_ambiguous_modal_triangle_possibilities(sorted_projected_data, modal_triangle)
    gradients = []
    # Test all ordering possibilities
    for i0 in triangle_possibilities[0]:
        for i1 in triangle_possibilities[1]:
            for i2 in triangle_possibilities[2]:
                # Skip if we have a coordinate twice
                if (i0 == i1 and len(triangle_possibilities[0]) > 1) or (
                        i1 == i2 and len(triangle_possibilities[1]) > 1):
                    continue
                # Update ordering
                sorted_indices_copy = sorted_indices.copy()
                original_index_0 = sorted_indices_copy[modal_triangle[0]]
                sorted_indices_copy[modal_triangle[0]] = sorted_indices_copy[i0]
                sorted_indices_copy[i0] = original_index_0
                original_index_1 = sorted_indices_copy[modal_triangle[1]]
                sorted_indices_copy[modal_triangle[1]] = sorted_indices_copy[i1]
                sorted_indices_copy[i1] = original_index_1
                original_index_2 = sorted_indices_copy[modal_triangle[2]]
                sorted_indices_copy[modal_triangle[2]] = sorted_indices_copy[i2]
                sorted_indices_copy[i2] = original_index_2
                # Get gradient if we would use that ordering
                gradient_new = dip_gradient(X, projected_data, sorted_indices_copy, modal_triangle)
                gradients.append(gradient_new)
    return np.array(gradients)


def _get_ambiguous_modal_triangle_possibilities(sorted_projected_data: np.ndarray, modal_triangle: np.ndarray) -> list:
    """
    Get all the possibilities for the modal triangle by checking if the surrounding values are equal to the triangle values.

    Parameters
    ----------
    sorted_projected_data : np.ndarray
        The data set projected onto the projection axis and sorted afterwards
    modal_triangle : tuple
        The modal triangle as returned by the dip-test

    Returns
    -------
    triangle_possibilities : list
        List containing a separate list for each triangle component which contains the ids of samples with an equal value
    """
    triangle_possibilities = [[modal_triangle[0]], [modal_triangle[1]], [modal_triangle[2]]]
    # Add indices with same value to triangle
    for j, triangle_point in enumerate(modal_triangle):
        i = 1
        # Check all values below returned triangle position
        while triangle_point - i >= 0 and sorted_projected_data[triangle_point - i] == sorted_projected_data[
            triangle_point]:
            triangle_possibilities[j].insert(0, triangle_point - i)
            i += 1
        # Check all values above returned triangle position
        i = 1
        while triangle_point + i < sorted_projected_data.shape[0] and sorted_projected_data[triangle_point + i] == \
                sorted_projected_data[triangle_point]:
            triangle_possibilities[j].append(triangle_point + i)
            i += 1
    return triangle_possibilities


"""
Utils
"""


def _transform_using_projections(X: np.ndarray, projection_axes: list, do_dip_scaling: bool,
                                 dip_values: np.ndarray, argsorted_dips: np.ndarray) -> np.ndarray:
    """
    Transform a give data set using the projection axes obtained by a previous DipExt execution.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    projection_axes : list
        List containing the axes used for the transformation
    do_dip_scaling : bool
        If true, the resulting features space will be scaled by performing a min-max normalization for each feature and multiplying this feautre by its dip-value
    dip_values : np.ndarray
        The dip-value of each resulting feature
    argsorted_dips : np.ndarray
        The indices of the sorted original dip-values (decreasing), needed to adjust order of the features

    Returns
    -------
    subspace : np.ndarray
        The resulting feature space (Number of samples x number of components),
    """
    subspace = np.zeros((X.shape[0], 0))
    remaining_X = X
    for projection in projection_axes:
        # Project data onto projection axis
        projected_data = np.matmul(remaining_X, projection)
        subspace = np.c_[subspace, projected_data]
        if subspace.shape[1] != len(projection_axes):
            # Prepare next iteration -> Remove transformed feature from data set
            orthogonal_space, _ = np.linalg.qr(projection.reshape(-1, 1), mode="complete")
            remaining_X = np.matmul(remaining_X, orthogonal_space[:, 1:])
    # Adjust order of the features
    subspace = subspace[:, argsorted_dips]
    # Optional: Scale data set using the dip-values
    if do_dip_scaling:
        subspace = _dip_scaling(subspace, dip_values)
    return subspace


def _angle(v: np.ndarray, w: np.ndarray) -> float:
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    v : np.ndarray
        The first vector
    w : np.ndarray
        The second vector

    Returns
    -------
    angle : float
        The calculated angle
    """
    quotient = np.linalg.norm(v, ord=2) * np.linalg.norm(w, ord=2)
    if quotient != 0:
        a = v.dot(w) / quotient
        # Due to numerical errors a can be > 1 or < -1 => force boundaries
        if a > 1:
            a = 1
        if a < -1:
            a = -1
        theta = np.arccos(a)
    else:
        theta = 0
    angle = 180 * theta / np.pi
    return angle


def _n_starting_vectors_default(n_dims: int) -> int:
    """
    Automatically define the number of starting vectors by applying the default strategy as described in the original paper.
    n_starting_vectors will be equal to int(np.log(n_dims)) + 1

    Parameters
    ----------
    n_dims : int
        The current number of features

    Returns
    -------
    n_starting_vectors : int
        The number of starting vectors for gradient descent
    """
    n_starting_vectors = int(np.log(n_dims)) + 1
    return n_starting_vectors


def _dip_scaling(X: np.ndarray, dip_values: np.ndarray) -> np.ndarray:
    """
    Perform dip scaling.
    Normalize each features using min-max normalization and multiply all feature values by their corresponding dip-values.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    dip_values : np.ndarray
        The dip-values for each feature

    Returns
    -------
    X : np.ndarray
        The scaled data set
    """
    X = np.array([dip_values[i] * (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i])) for i in
                  range(X.shape[1])]).T
    return X


def _dip_init(subspace: np.ndarray, n_clusters: int) -> (np.ndarray, np.ndarray):
    """
    Execute the DipInit clustering procedure. Executes KMeans using initial cluster centers.

    Parameters
    ----------
    subspace : np.ndarray
        The subspace as identified by DipExt
    n_clusters : int
        The number of clusters

    Returns
    -------
    tuple: (np.ndarray, np.ndarray)
        The labels as identified by DipInit,
        The cluster centers as identified by DipInit
    """
    how_many = subspace.shape[0] // n_clusters
    # Get the first initialisation by frequency-binning the primary feature
    centers = np.zeros((n_clusters, 1))
    sorted_primary = np.sort(subspace[:, 0])
    for i in range(n_clusters):
        centers[i] = np.mean(sorted_primary[i * how_many:i * how_many + how_many])
    km = KMeans(n_clusters=n_clusters, init=centers, n_init=1)
    km.fit(subspace[:, 0].reshape(-1, 1))
    if subspace.shape[1] > 1:
        # Add features one by one depending on their dip value
        for i in range(1, subspace.shape[1]):
            centers = np.array([np.mean(subspace[km.labels_ == clus, :i + 1], axis=0) for clus in range(n_clusters)])
            km = KMeans(n_clusters=n_clusters, init=centers, n_init=1)
            km.fit(subspace[:, :i + 1])
    return km.labels_, km.cluster_centers_


"""
DipExt object
"""


class DipExt(TransformerMixin, BaseEstimator):
    """
    Execute the DipExt algorithm to reduce the number of features.
    Therefore, it utilizes the gradient of the Dip-test of unimodality to perform gradient descent.
    The output features should show a high degree of modality.

    Parameters
    ----------
    n_components : int
        The number of components to extract. Can be None, in that case dip_threshold wil be used to define the number of components (default: None)
    do_dip_scaling : bool
        If true, the resulting features space will be scaled by performing a min-max normalization for each feature and multiplying this feautre by its dip-value (default: True)
    step_size : float
        Step size used for gradient descent (default: 0.1)
    momentum : float
        Momentum used for gradient descent (default: 0.95)
    dip_threshold : float
        Defines the number of components if n_components is None. If an identified feature has a dip-value below the maximum dip-value times dip_threshold, DipExt will terminate (default: 0.5)
    n_starting_vectors : int
        The number of starting vectors for gradient descent. Can be None, in that case it will be equal to log(data dimensionality) + 1 (default: None)
    ambiguous_triangle_strategy : str
        The strategy with which to handle an ambiguous modal triangle. Can be 'ignore', 'random' or 'all'.
        In the case of 'random', a valid triangle is created at random.
        In the case of 'all', for each possible triangle the gradient is calculated and it is checked for which gradient the following result looks most promising - this strategy can increase the runtime noticeably (default: 'ignore')
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int. Only used if ambiguous_triangle_strategy is 'random' (default: None)

    Attributes
    ----------
    dip_values_ : np.ndarray
        The dip-value of each resulting feature
    projection_axes_ : list
        List containing the axes used for the transformation
    argsorted_dips_ : np.ndarray
        The indices of the sorted original dip-values (decreasing), needed to adjust order of the features when using transform()

    References
    ----------
    Schelling, Benjamin, et al. "Utilizing Structure-rich Features to improve Clustering." (2020).
    The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2020
    """

    def __init__(self, n_components: int = None, do_dip_scaling: bool = True, step_size: float = 0.1,
                 momentum: float = 0.95, dip_threshold: float = 0.5, n_starting_vectors: int = None,
                 ambiguous_triangle_strategy: str = "ignore", random_state: np.random.RandomState | int = None):
        self.n_components = n_components
        self.do_dip_scaling = do_dip_scaling
        self.step_size = step_size
        self.momentum = momentum
        self.dip_threshold = dip_threshold
        self.n_starting_vectors = n_starting_vectors
        self.ambiguous_triangle_strategy = ambiguous_triangle_strategy
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DipExt':
        """
        Retrieve the necessary projection axes to apply DipExt to any given data set.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DipExt
            This instance of the DipExt algorithm
        """
        _ = self.fit_transform(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to the given data set using the projection axes found by DipExt.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        subspace : np.ndarray
            The transformed feature space (Number of samples x number of components)
        """
        assert hasattr(self, "projection_axes_"), "Projection axes have not been obtained. Run fit() first."
        subspace = _transform_using_projections(X, self.projection_axes_, self.do_dip_scaling, self.dip_values_,
                                                self.argsorted_dips_)
        return subspace

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Initiate the actual dimensionality-reduction process on the input data set.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        subspace : np.ndarray
            The transformed feature space (Number of samples x number of components)
        """
        if self.n_starting_vectors is None:
            self.n_starting_vectors = _n_starting_vectors_default(X.shape[1])
        subspace, dip_values, projections, argsorted_dip = _dip_ext(X, self.n_components, self.do_dip_scaling,
                                                                    self.step_size, self.momentum, self.dip_threshold,
                                                                    self.n_starting_vectors,
                                                                    self.ambiguous_triangle_strategy,
                                                                    self.random_state)
        self.n_components = len(dip_values)
        self.dip_values_ = dip_values
        self.projection_axes_ = projections
        self.argsorted_dips_ = argsorted_dip
        return subspace


class DipInit(DipExt, BaseEstimator, ClusterMixin):
    """
    Execute the DipInit clustering procedure.
    Initially, DipExt is executed to identify relevant features.
    Next, KMeans with initial cluster centers is used to identify high-quality cluster labels.
    To get the coordinates of the initial cluster centers DipInit uses the features identified by DipExt one after another.
    In the first iteration the centers will be equally distributed in a one-dimensional space by using the ids of the objects.
    Thereafter, the algorithm adds additional features und uses the current cluster labels to also add another coordinate to the centers.

    Parameters
    ----------
    n_clusters : int
        The number of clusters
    n_components : int
        The number of components to extract. Can be None, in that case dip_threshold wil be used to define the number of components (default: None)
    do_dip_scaling : bool
        If true, the resulting features space will be scaled by performing a min-max normalization for each feature and multiplying this feautre by its dip-value (default: True)
    step_size : float
        Step size used for gradient descent (default: 0.1)
    momentum : float
        Momentum used for gradient descent (default: 0.95)
    dip_threshold : float
        Defines the number of components if n_components is None. If an identified feature has a dip-value below the maximum dip-value times dip_threshold, DipExt will terminate (default: 0.5)
    n_starting_vectors : int
        The number of starting vectors for gradient descent. Can be None, in that case it will be equal to log(data dimensionality) + 1 (default: None)
    ambiguous_triangle_strategy : str
        The strategy with which to handle an ambiguous modal triangle. Can be 'ignore', 'random' or 'all'.
        In the case of 'random', a valid triangle is created at random.
        In the case of 'all', for each possible triangle the gradient is calculated and it is checked for which gradient the following result looks most promising - this strategy can increase the runtime noticeably (default: 'ignore')
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int. Only used if ambiguous_triangle_strategy is 'random' (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers

    References
    ----------
    Schelling, Benjamin, et al. "Utilizing Structure-rich Features to improve Clustering." (2020).
    The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2020
    """

    def __init__(self, n_clusters: int, n_components: int = None, do_dip_scaling: bool = True, step_size: float = 0.1,
                 momentum: float = 0.95, dip_threshold: float = 0.5, n_starting_vectors: int = None,
                 ambiguous_triangle_strategy: str = "ignore", random_state: np.random.RandomState | int = None):
        super().__init__(n_components, do_dip_scaling, step_size, momentum, dip_threshold, n_starting_vectors,
                         ambiguous_triangle_strategy, random_state)
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DipInit':
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
        self : DipInit
            this instance of the DipInit algorithm
        """
        subspace = self.fit_transform(X)
        labels, centers = _dip_init(subspace, self.n_clusters)
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

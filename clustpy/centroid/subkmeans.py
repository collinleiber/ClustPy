"""
@authors:
Collin Leiber
"""

from clustpy.alternative.nrkmeans import NrKmeans, _get_total_cost_function, _mdl_costs
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from clustpy.utils.plots import plot_scatter_matrix
from clustpy.utils.checks import check_parameters
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


def _transform_subkmeans_m_to_nrkmeans_m(m: int, dims: int) -> list:
    """
    Transform the m of SubKmeans to the representation of NrKmeans which includes the noise space.

    Parameters
    ----------
    m : int
        Dimensionality of the subspace of SubKmeans
    dims : int
        Dimensions of the dataset

    Returns
    -------
    nrkmeans_m : list
        The dimensionality of the subspace of SubKmeans and the noise space
    """
    nrkmeans_m = [m, dims - m]
    return nrkmeans_m


def _transform_subkmeans_P_to_nrkmeans_P(m: int, dims: int) -> list:
    """
    Use the m of SubKmeans to create the projections usable by NrKmeans.

    Parameters
    ----------
    m : int
        Dimensionality of the subspace of SubKmeans
    dims : int
        Dimensions of the dataset

    Returns
    -------
    nrkmeans_P : list
        The projections of the subspace of SubKmeans and the noise space
    """
    nrkmeans_P = [np.arange(m), np.arange(m, dims)]
    return nrkmeans_P


def _transform_subkmeans_centers_to_nrkmeans_centers(X: np.ndarray, centers: np.ndarray) -> list:
    """
    Transform the cluster centers of SubKmeans to the representation of NrKmeans which includes the noise space.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    centers : np.ndarray
        The centers of the subspace of SubKmeans

    Returns
    -------
    nrkmeans_centers : list
        The cluster centers of the subspace of SubKmeans and the noise space
    """
    nrkmeans_centers = [centers, np.array([np.mean(X, axis=0)])]
    return nrkmeans_centers


def _transform_subkmeans_scatter_to_nrkmeans_scatters(X: np.ndarray, scatter_matrix: np.ndarray) -> list:
    """
    Transform the scatter matrix of SubKmeans to the representation of NrKmeans which includes the noise space.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    scatter_matrix : np.ndarray
        The scatter matrix of the subspace of SubKmeans

    Returns
    -------
    nrkmeans_scatter_matrices : list
        The scatter matrix of the subspace of SubKmeans and the noise space
    """
    noise_center = np.mean(X, axis=0)
    centered_points = X - noise_center
    noise_scatter_matrix = np.matmul(centered_points.T, centered_points)
    nrkmeans_scatter_matrices = [scatter_matrix, noise_scatter_matrix]
    return nrkmeans_scatter_matrices


class SubKmeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    The Subspace Kmeans (SubKmeans) algorithm.
    The algorithm will simultaneously search for cluster assignments and an optimal subspace regarding those assignments.
    Here, the feature-space will be divided into an clustered space containing the actual clustering result and the noise space which is a special subspace containing a single cluster.
    The rotation and dimensions of the clustered space are defined through an eigenvalue decomposition.

    This implementation includes some extensions from 'Automatic Parameter Selection for Non-Redundant Clustering'.

    Parameters
    ----------
    n_clusters : int
        the number of clusters (deafault: 8)
    V_init : np.ndarray
        the orthonormal rotation matrix (default: None)
    m_init : int
        the initial dimensionality of the clustered space (default: None)
    cluster_centers_init : np.ndarray
        list containing the initial cluster centers (default: None)
    mdl_for_noisespace : bool
        defines if MDL should be used to identify noise space dimensions instead of only considering negative eigenvalues (default: False)
    outliers : bool
        defines if outliers should be identified through MDL (default: False)
    max_iter : int
        maximum number of iterations for the algorithm (default: 300)
    n_init : int
        number of times SubKmeans is executed using different seeds. The final result will be the one with lowest costs.
        Costs can be the standard SubKmeans costs or MDL costs (defined by the cost_type parameter) (default: 1)
    cost_type : str
        Can be "default" or "mdl" and defines whether the the standard SubKmeans cost function or MDL costs should be considered to identify the best result.
        Only relevant if n_init is larger than 1 (default: "default")
    threshold_negative_eigenvalue : float
        threshold to consider an eigenvalue as negative. Used for the update of the subspace dimensions (default: -1e-7)
    max_distance : float
        distance used to encode cluster centers and outliers. Only relevant if a MDL strategy is used (default: None)
    precision : float
        precision used to convert probability densities to actual probabilities. Only relevant if a MDL strategy is used (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    scatter_matrix_ : np.ndarray
        The final scatter matrix
    n_iter_ : list
        The number of iterations used to achieve the result
    n_features_in_ : int
        the number of features used for the fitting

    References
    ----------
    Mautz, Dominik, et al. "Towards an optimal subspace for k-means."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.

    and

    Leiber, Collin, et al. "Automatic Parameter Selection for Non-Redundant Clustering."
    Proceedings of the 2022 SIAM International Conference on Data Mining (SDM).
    Society for Industrial and Applied Mathematics, 2022.
    """

    def __init__(self, n_clusters: int = 8, V_init: np.ndarray = None, m_init: int = None,
                 cluster_centers_init: np.ndarray = None, mdl_for_noisespace: bool = False, outliers: bool = False,
                 max_iter: int = 300, n_init: int = 1, cost_type: str = "default",
                 threshold_negative_eigenvalue: float = -1e-7, max_distance: float = None, precision: float = None,
                 random_state: np.random.RandomState | int = None, debug: bool = False):
        # Fixed attributes
        self.max_iter = max_iter
        self.n_init = n_init
        self.cost_type = cost_type
        self.threshold_negative_eigenvalue = threshold_negative_eigenvalue
        self.mdl_for_noisespace = mdl_for_noisespace
        self.outliers = outliers
        self.max_distance = max_distance
        self.precision = precision
        self.debug = debug
        self.random_state = random_state
        # Variables
        self.n_clusters = n_clusters
        self.cluster_centers_init = cluster_centers_init
        self.V_init = V_init
        self.m_init = m_init

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'SubKmeans':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : SubKmeans
            this instance of the SubKmeans algorithm
        """
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        n_clusters = [self.n_clusters, 1] if self.n_clusters != 1 else [1]
        if self.m_init is not None:
            m_init = _transform_subkmeans_m_to_nrkmeans_m(self.m_init, X.shape[1])
            P_init = _transform_subkmeans_P_to_nrkmeans_P(self.m_init, X.shape[1])
        else:
            m_init = self.m_init
            P_init = None
        if self.cluster_centers_init is not None:
            cluster_centers_init = _transform_subkmeans_centers_to_nrkmeans_centers(X, self.cluster_centers_init)
        else:
            cluster_centers_init = self.cluster_centers_init
        nrkmeans = NrKmeans(n_clusters, V_init=self.V_init, m_init=m_init, P_init=P_init, cluster_centers_init=cluster_centers_init,
                            mdl_for_noisespace=self.mdl_for_noisespace, outliers=self.outliers,
                            max_iter=self.max_iter, n_init=self.n_init,
                            threshold_negative_eigenvalue=self.threshold_negative_eigenvalue,
                            max_distance=self.max_distance,
                            precision=self.precision, random_state=random_state, debug=self.debug)
        nrkmeans.fit(X)
        # Adjust rotation to match SubKmeans properties
        if len(nrkmeans.P_) == 2:
            self.V_ = nrkmeans.V_[:, np.r_[nrkmeans.P_[0], nrkmeans.P_[1]]]
        else:
            self.V_ = nrkmeans.V_[:, np.r_[nrkmeans.P_[0]]]
        if nrkmeans.labels_.ndim == 1:
            self.labels_ = nrkmeans.labels_
        else:
            self.labels_ = nrkmeans.labels_[:, 0]
        self.n_iter_ = nrkmeans.n_iter_
        self.cluster_centers_ = nrkmeans.cluster_centers_[0]
        self.m_ = nrkmeans.m_[0]
        self.scatter_matrix_ = nrkmeans.scatter_matrices_[0]
        self.n_clusters_final_ = nrkmeans.n_clusters_final_[0]
        self.n_features_in_ = X.shape[1]
        return self

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
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        rotated_data = np.matmul(X, self.V_)
        return rotated_data

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input dataset with the orthonormal rotation matrix identified by the fit function and
        project it into the clustered space.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        rotated_data : np.ndarray
            The rotated and projected dataset
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        clustered_space_V = self.V_[:, :self.m_]
        rotated_data = np.matmul(X, clustered_space_V)
        return rotated_data

    def fit_transform(self, X: np.ndarray, y: np.ndarray=None):
        """
        Train the clusterin algorithm on the given data set and return the final embedded version of the data using the obtained subspace.

        Parameters
        ----------
        X: np.ndarray
            The given data set
        y : np.ndarray
            the labels (can usually be ignored)

        Returns
        -------
        X_embed : np.ndarray
            The embedded data set
        """
        self.fit(X, y)
        X_embed = self.transform(X)
        return X_embed

    def plot_clustered_space(self, X: np.ndarray, labels: np.ndarray = None, plot_centers: bool = False,
                             gt: np.ndarray = None, equal_axis=False) -> None:
        """
        Plot the clustered space identified by SubKmeans as scatter matrix plot.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        labels : np.ndarray
            the cluster labels used for coloring the plot. If none, the labels identified by the fit() function will be used (default: None)
        plot_centers : bool
            defines whether the cluster centers should be plotted (default: False)
        gt : np.ndarray
            the ground truth labels. In contrast to the labels parameter this will be displayed using different markers instead of colors (default: None)
        equal_axis : bool
            defines whether the axes should be scaled equally
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        if labels is None:
            labels = self.labels_
        assert X.shape[0] == labels.shape[0], "Number of data objects must match the number of labels."
        plot_scatter_matrix(self.transform(X), labels,
                            self.transform(self.cluster_centers_) if
                            plot_centers else None, true_labels=gt, equal_axis=equal_axis)

    def calculate_mdl_costs(self, X: np.ndarray) -> (float, float, list):
        """
        Calculate the Mdl Costs of this SubKmeans result.

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
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        m = _transform_subkmeans_m_to_nrkmeans_m(self.m_, X.shape[1])
        P = _transform_subkmeans_P_to_nrkmeans_P(self.m_, self.V_.shape[0])
        scatter_matrices = _transform_subkmeans_scatter_to_nrkmeans_scatters(X, self.scatter_matrix_)
        labels = np.c_[self.labels_, np.zeros(self.labels_.shape[0])]
        total_costs, global_costs, all_subspace_costs = _mdl_costs(X, [self.n_clusters, 1], m, P, self.V_,
                                                                   scatter_matrices, labels,
                                                                   self.outliers, self.max_distance, self.precision)
        return total_costs, global_costs, all_subspace_costs

    def calculate_cost_function(self, X: np.ndarray) -> float:
        """
        Calculate the result of the SubKmeans cost function. Depends on the rotation and the scatter matrix.
        Calculates for both subspaces j:
        P_j^T*V^T*S_j*V*P_j

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        costs : float
            The total loss of this SubKmeans object
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        P = _transform_subkmeans_P_to_nrkmeans_P(self.m_, self.V_.shape[0])
        scatter_matrices = _transform_subkmeans_scatter_to_nrkmeans_scatters(X, self.scatter_matrix_)
        costs = _get_total_cost_function(self.V_, P, scatter_matrices)
        return costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of an input dataset. For this method the results from the fit() method will be used.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        predicted_labels : np.ndarray
            the predicted labels of the input data set
        """
        X_transform = self.transform(X)
        centers_transform = self.transform(self.cluster_centers_)
        predicted_labels, _ = pairwise_distances_argmin_min(X=X_transform, Y=centers_transform,
                                                          metric='euclidean',
                                                          metric_kwargs={'squared': True})
        predicted_labels = predicted_labels.astype(np.int32)
        return predicted_labels

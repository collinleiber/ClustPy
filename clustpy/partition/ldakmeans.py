"""
authors:
Collin Leiber
"""

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import numpy as np
from scipy.linalg import eigh
from clustpy.alternative.nrkmeans import _update_centers_and_scatter_matrix
from clustpy.utils.checks import check_parameters
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


def _lda_kmeans(X: np.ndarray, n_clusters: int, n_dims: int, max_iter: int, kmeans_repetitions: int,
                random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, np.ndarray, float, int):
    """
    Start the actual LDA-Kmeans clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        the number of clusters
    n_dims : int
        The number of features in the resulting subspace
    max_iter : int
        the maximum number of iterations
    kmeans_repetitions : int
            Number of repetitions when executing KMeans. For more information see sklearn.cluster.KMeans (default: 10)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, float, int)
        The labels as identified by LDAKmeans,
        The final rotation matrix,
        The cluster centers in the subspace,
        The final error,
        The number of iterations used for clustering
    """
    assert max_iter > 0, "max_iter must be larger than 0"
    if n_dims >= X.shape[1]:
        km = KMeans(n_clusters, n_init=kmeans_repetitions, random_state=random_state)
        km.fit(X)
        return km.labels_, np.identity(X.shape[1]), km.cluster_centers_, km.inertia_, 1
    # Check if labels stay the same (break condition)
    old_labels = None
    # Global parameters
    global_mean = np.mean(X, axis=0)
    centered_points = X - global_mean
    St = np.matmul(centered_points.T, centered_points) / (X.shape[0] - 1)
    # Get initial rotation
    pca = PCA(n_dims)
    pca.fit(X)
    rotation = pca.components_.T
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Update labels
        X_subspace = np.matmul(X, rotation)
        km = KMeans(n_clusters, n_init=kmeans_repetitions, random_state=random_state)
        km.fit(X_subspace)
        # Check if labels have not changed
        if old_labels is not None and nmi(km.labels_, old_labels) == 1:
            break
        else:
            old_labels = km.labels_.copy()
        # Update subspace
        _, scatter = _update_centers_and_scatter_matrix(X, n_clusters, km.labels_)
        Sw = scatter / (X.shape[0] - 1)
        Sb = St - Sw
        try:
            _, eigen_vectors = eigh(Sb, Sw)
            # Take the eigenvectors with largest eigenvalues
            rotation = eigen_vectors[:, ::-1][:, :n_dims]
        except:
            # In case errors occur during eigenvalue decomposition keep algorithm running
            pass
    return km.labels_, rotation, km.cluster_centers_, km.inertia_, iteration + 1


class LDAKmeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    Execute the LDA-Kmeans clustering procedure.
    The initial rotation are normally the (n_clusters-1) components of a PCA.
    Afterward, Kmeans and LDA are executed one after the other until the labels do not change anymore.
    Kmeans always takes place in the rotated subspace.

    Parameters
    ----------
    n_clusters : int
        the number of clusters (default: 8)
    n_dims : int
        The number of features in the resulting subspace. If None this will be equal to n_clusters - 1 (default: None)
    max_iter : int
        the maximum number of iterations (default: 300)
    n_init : int
        number of times LDAKmeans is executed using different seeds. The final result will be the one with lowest costs (default: 1)
    kmeans_repetitions : int
        Number of repetitions when executing KMeans. For more information see sklearn.cluster.KMeans (default: 10)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    rotation_ : np.ndarray
        The final rotation matrix
    cluster_centers_ = np.ndarray
        The cluster centers in the subspace
    error_ : float
        The final error (KMeans error in the subspace)
    n_features_in_ : int
        the number of features used for the fitting

    References
    -------
    Ding, Chris, and Tao Li. "Adaptive dimension reduction using discriminant analysis and k-means clustering."
    Proceedings of the 24th international conference on Machine learning. 2007.
    """

    def __init__(self, n_clusters: int = 8, n_dims: int = None, max_iter: int = 300, n_init: int = 1,
                 kmeans_repetitions: int = 10, random_state: np.random.RandomState | int = None):
        self.n_clusters = n_clusters
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.n_init = n_init
        self.kmeans_repetitions = kmeans_repetitions
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'LDAKmeans':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels are contained in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : LDAKmeans
            this instance of the LDAKmeans algorithm
        """
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        all_random_states = random_state.choice(10000, self.n_init, replace=False)
        n_dims = max(1, self.n_clusters - 1 if self.n_dims is None else self.n_dims)
        # Get best result
        best_costs = np.inf
        for i in range(self.n_init):
            local_random_state = check_random_state(all_random_states[i])
            labels, rotation, centers, error, n_iter = _lda_kmeans(X, self.n_clusters, n_dims, self.max_iter,
                                                           self.kmeans_repetitions,
                                                           local_random_state)
            if error < best_costs:
                best_costs = error
                # Update class variables
                self.labels_ = labels
                self.rotation_ = rotation
                self.cluster_centers_ = centers
                self.error_ = error
                self.n_features_in_ = X.shape[1]
                self.n_iter_ = n_iter
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input dataset with the rotation matrix identified by the fit function.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        rotated_data : np.ndarray
            The rotated data set
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        rotated_data = np.matmul(X, self.rotation_)
        return rotated_data

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
        predicted_labels, _ = pairwise_distances_argmin_min(X=X_transform, Y=self.cluster_centers_,
                                                          metric='euclidean',
                                                          metric_kwargs={'squared': True})
        predicted_labels = predicted_labels.astype(np.int32)
        return predicted_labels

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

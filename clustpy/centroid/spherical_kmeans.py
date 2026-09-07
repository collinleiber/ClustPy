import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import kmeans_plusplus as kpp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import normalize
from clustpy.utils.checks import check_parameters

class SphericalKMeans(BaseEstimator, ClusterMixin):
    """
    The Spherical-k-Means algorithm.
    Instead of using the Euclidean distance, it uses the cosince distance.

    Parameters
    ----------
    n_clusters : int
        the number of clusters (default: 8)
    max_iter : int
        maximum number of iterations for the algorithm (default: 300)
    n_init : int
        The number of times the algorithm is executed using different seeds. Only the result with the lowest inertia will be returned (default: 10)
    tol : float
        tolerance for convergence of the algorithm (default: 1e-5)
    random_state : np.random.RandomState | int | None
        The random state (default: None)

    Attributes
    ----------
    inertia_ : float
        The final inertia
    labels_ : np.ndarray
        The cluster labels
    cluster_centers_ : np.ndarray
        The cluster centers
    n_iter_ : int
        The number of used iterations
    n_features_in_ : int
        the number of features used for the fitting
    """
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, n_init: int = 10, tol: float = 1e-5,
                 random_state: np.random.RandomState | int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

    def calculate_inertia(self, X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the inertia for Spherical-k-Means.
        It is based on the following formulation:
        Within-cluster-sum-of-squares (WCSS) corresponds to 2 * within-cluster-sum-of-cosine-dissimilarities

        Parameters
        ----------
        X : np.ndarray
            The data set
        centers : np.ndarray
            The cluster centers
        labels : np.ndarray
            The cluster labels

        Returns
        -------
        inertia : float
            The inertia with respect to the given clustering parameters
        """
        wcss = ((X-centers[labels])**2).sum()
        inertia = 0.5 * wcss
        return inertia

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'SphericalKMeans':
        """
        Fit Spherical-k-Means to the given data set.

        Parameters
        ----------
        X : np.ndarray
            The data set
        y : np.ndarray | None
            The labels

        Returns
        -------
        self : SphericalKMeans
            This instance of the Spherical-k-Means algorithm
        """
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        assert self.n_clusters < X.shape[0]
        # Convert X to unit size vectors
        X = normalize(X)
        # Start clustering proocess
        self.inertia_ = np.inf
        for _ in range(self.n_init):
            # Initialize clusters
            centers = kpp(X, self.n_clusters, random_state=random_state)[0]
            centers = normalize(centers)
            for iteration in range(self.max_iter):
                cosine_similarity = X @ centers.T
                labels = cosine_similarity.argmax(1)
                # Optimization step
                old_centers = centers
                centers = np.ones((self.n_clusters, X.shape[1]))
                for cluster_id in range(self.n_clusters):
                    cluster_subset = X[labels == cluster_id]
                    if cluster_subset.shape[0] > 0:
                        centers[cluster_id] = np.mean(cluster_subset, axis=0)
                centers = normalize(centers)
                # Check convergence
                if np.linalg.norm(old_centers - centers) < self.tol:
                    break
            inertia_iter = self.calculate_inertia(X, centers, labels)
            # Check if run was better than other iterations
            if inertia_iter < self.inertia_:
                self.inertia_ = inertia_iter
                self.cluster_centers_ = centers
                self.labels_ = labels.astype(np.int32)
                self.n_iter_ = iteration + 1
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given data set using Spherical-k-Means.

        Parameters
        ----------
        X : np.ndarray
            The data set

        Returns
        -------
        self : np.ndarray
            The predicted labels
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        X = normalize(X)
        cosine_similarity = X @ self.cluster_centers_.T
        labels = cosine_similarity.argmax(1)
        return labels

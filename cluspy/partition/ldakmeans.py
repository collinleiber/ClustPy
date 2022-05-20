from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import numpy as np
from scipy.linalg import eigh
from cluspy.alternative.nrkmeans import _update_centers_and_scatter_matrices


def _lda_kmeans(X, n_clusters, max_iter):
    """
    Execute the LDA-Kmeans algorithm. The initial rotation are the (n_clusters-1) components of a PCA.
    Afterward, Kmeans and LDA are executed one after the other until the labels do not change anymore.
    Kmeans always takes place in the rotated subspace.

    Parameters
    ----------
    X: the given data set
    n_clusters
    max_iter

    Returns
    -------
    (labels, final rotation)
    """
    assert n_clusters > 1, "n_clusters must be larger than 1"
    assert max_iter > 0, "max_iter must be larger than 0"
    dims = n_clusters - 1
    if dims >= X.shape[1]:
        km = KMeans(n_clusters)
        km.fit(X)
        return km.labels_, np.identity(X.shape[1])
    # Check if labels stay the same (break condition)
    old_labels = None
    global_mean = np.mean(X, axis=0)
    centered_points = X - global_mean
    St = np.matmul(centered_points.T, centered_points) / (X.shape[0] - 1)
    # Get initial rotation
    pca = PCA(dims)
    pca.fit(X)
    rotation = pca.components_.T
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Update labels
        X_subspace = np.matmul(X, rotation)
        km = KMeans(n_clusters, n_init=1)
        km.fit(X_subspace)
        # Check if labels have not changed
        if old_labels is not None and nmi(km.labels_, old_labels) == 1:
            break
        else:
            old_labels = km.labels_.copy()
        # Update subspace
        _, Sw = _update_centers_and_scatter_matrices(X, n_clusters, km.labels_)
        Sw = np.sum(Sw, axis=0) / (X.shape[0] - 1)
        Sb = St - Sw
        _, eigen_vectors = eigh(Sb, Sw)
        # Take the eigenvectors with largest eigenvalues
        rotation = eigen_vectors[:, ::-1][:, :dims]
    return km.labels_, rotation


class LDAKmeans(BaseEstimator, ClusterMixin):
    """
    LDA K-means algorithm
    """

    def __init__(self, n_clusters, max_iter=300):
        """
        Create an instance of the LDA-K-Means algorithm.

        Parameters
        ----------
        n_clusters : number of clusters
        max_iter : maximum number of iterations (default: 300)

        References
        -------
        Ding, Chris, and Tao Li. "Adaptive dimension reduction using discriminant analysis and k-means clustering."
        Proceedings of the 24th international conference on Machine learning. 2007.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels are contained in the labels_ attribute.

        Parameters
        ----------
        X : the given data set
        y : the labels (default: None - can be ignored)

        Returns
        -------
        returns the clustering object
        """
        labels, rotation = _lda_kmeans(X, self.n_clusters, self.max_iter)
        self.labels_ = labels
        self.rotation_ = rotation
        return self

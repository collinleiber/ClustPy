"""
@authors:
Collin Leiber

Based on the original implementation, available at:
https://github.com/Sibylse/SpecialK/blob/master/SpecialK.ipynb
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
import scipy


def _specialk(X: np.ndarray, significance: float, n_dimensions: int, similarity_matrix: str, n_neighbors: int,
              percentage: float, n_cluster_pairs_to_consider: int, max_n_clusters: int,
              random_state: np.random.RandomState, debug: bool) -> (int, np.ndarray):
    """
    Start the actual SpecialK clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the samples originate from a single distribution or two distributions
    n_dimensions : int
        Dimensionality of the embedding
    similarity_matrix : str
        Defines the similarity matrix to use. Can be one of the following strings or a numpy array / scipy sparse csr matrix.
        If 'NAM', a neighborhood adjacency matrix is used.
        If 'SAM' a symmetrically normalized adjacency matrix is used
    n_neighbors : int
        Number of neighbors for the construction of the similarity matrix. Does not include the object itself
    percentage : float
        The amount of data points that should have at least n_neighbors neighbors.
        Only relevant if use_neighborhood_adj_mat is set to True
    n_cluster_pairs_to_consider : int
        The number of cluster pairs to consider when calculating the probability boundary.
        The cluster pairs responsible for the highest cut. If None, all pairs will be considered.
        Smaller values for n_cluster_pairs_to_consider will decrease the computing time
    max_n_clusters : int
        Maximum number of clusters
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (int, np.ndarray)
        The final number of clusters,
        The labels as identified by DipMeans,
    """
    assert significance >= 0 and significance <= 1, "significance must be a value in the range [0, 1]"
    assert percentage >= 0 and percentage <= 1, "percentage must be a value in the range [0, 1]"
    if type(similarity_matrix) is str and similarity_matrix == 'NAM':
        final_similarity_matrix = _get_neighborhood_adjacency_matrix(X, percentage, n_neighbors)
    elif type(similarity_matrix) is str and similarity_matrix == 'SAM':
        final_similarity_matrix = _get_symmetrically_normalized_adjacency_matrix(X, n_neighbors)
    elif type(similarity_matrix) is np.ndarray or type(similarity_matrix) is scipy.sparse.csr_matrix:
        final_similarity_matrix = similarity_matrix
    else:
        raise ValueError(
            "similarity_matrix must be 'NAM' (Neighborhood Adjacency Matrix), 'SAM' (Symmetrically Normalized Adjacency Matrix) or a numpy array.")
    # Get eigenvalues and eigenvectors
    my_lambda, V = scipy.sparse.linalg.eigsh(final_similarity_matrix, k=n_dimensions, which="LM")
    my_lambda, D = np.absolute(my_lambda), np.absolute(V)
    D = D * np.sqrt(my_lambda)
    # Initial values
    n_clusters = 2
    stop_search = False
    best_labels = np.zeros(X.shape[0])
    while n_clusters <= max_n_clusters:
        if debug:
            print("=== n_clusters={0} ===".format(n_clusters))
        # Execute KMeans
        kmeans = KMeans(n_clusters, random_state=random_state)
        kmeans.fit(D)
        ids_in_each_cluster = [np.where(kmeans.labels_ == c)[0] for c in range(n_clusters)]
        if n_cluster_pairs_to_consider is None:
            # Get all pairs of clusters
            cluster_pairs = [(c1, c2) for c1 in range(n_clusters - 1) for c2 in range(c1 + 1, n_clusters)]
        else:
            # Only consider the pairs of clusters responsible for the maximum cuts
            # Calculate cuts
            one_hot = np.zeros((D.shape[0], n_clusters), dtype=int)
            for c in range(n_clusters):
                one_hot[ids_in_each_cluster[c], c] = 1
            cuts = one_hot.T @ final_similarity_matrix @ one_hot
            # Fill upper triangle matrix with 0s and ignore diagonal
            cuts = np.tril(cuts)
            np.fill_diagonal(cuts, -np.inf)  # operation is inplace
            # Get cluster pairs with maximum cuts
            cluster_pairs = np.column_stack(np.unravel_index(np.argsort(cuts, axis=None), cuts.shape))[::-1]
            cluster_pairs = cluster_pairs[:min(int(n_clusters * (n_clusters - 1) / 2), n_cluster_pairs_to_consider)]
        # Iterate over the cluster pairs
        for c1, c2 in cluster_pairs:
            if debug:
                print("check cluster {0} and {1} (cut={2})".format(c1, c2, cuts[
                    c1, c2] if n_cluster_pairs_to_consider is not None else None))
            ids_in_cluster_1 = ids_in_each_cluster[c1]
            ids_in_cluster_2 = ids_in_each_cluster[c2]
            # Calculate bound
            t_total = _zz_top_bound(D, ids_in_cluster_1, ids_in_cluster_2, debug)
            if debug:
                print("ZZ top:", t_total)
            if t_total > significance:
                # Stop execution -> return n_clusters - 1
                stop_search = True
                break
        if stop_search:
            break
        else:
            # Save current result as best one so far
            best_labels = kmeans.labels_
            n_clusters += 1
    # Return number of clusters and labels
    if debug:
        print("Final n_clusters={0}".format(n_clusters - 1))
    return n_clusters - 1, best_labels


def _zz_top_bound(D: np.ndarray, ids_in_cluster_1: np.ndarray, ids_in_cluster_2: np.ndarray, debug: bool) -> float:
    """
    Calculate the ZZ Top bound

    Parameters
    ----------
    D : np.ndarray
        Approximated similarity matrix
    ids_in_cluster_1 : np.ndarray
        The ids of the objects in cluster 1
    ids_in_cluster_2 : np.ndarray
        The ids of the objects in cluster 2
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    t_total : float
        The calculated bound
    """
    Dj = D[np.r_[ids_in_cluster_1, ids_in_cluster_2],]
    norms = np.linalg.norm(Dj, axis=0)
    norms[norms == 0] = 1
    Z = (Dj - np.mean(Dj, axis=0)) / norms
    sigma2 = np.mean(Z * Z)
    t1 = np.linalg.norm(np.sum(Z[:ids_in_cluster_1.shape[0]], axis=0)) ** 2 / ids_in_cluster_1.shape[0]
    t2 = np.linalg.norm(np.sum(Z[ids_in_cluster_1.shape[0]:], axis=0)) ** 2 / ids_in_cluster_2.shape[0]
    t = max(t1, t2) - sigma2 * Dj.shape[1]
    if debug:
        print("sigma={0} / t={1}".format(sigma2, t))
    t_total = Dj.shape[0] * np.exp(-0.5 * t ** 2 / (Dj.shape[1] * sigma2 + t / 3))
    return t_total


def _get_neighborhood_adjacency_matrix(X: np.ndarray, percentage: float = 0.99,
                                       n_neighbors: int = 10) -> scipy.sparse.csr_matrix:
    """
    Get a neighborhood adjacency matrix, so that p% of the data points have at least n_neighbors neighbors.
    Here, p can be chosen using the 'percentage' parameter.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    percentage : float
        The amount of data points that should have at least n_neighbors neighbors (default: 0.99)
    n_neighbors : int
        The number of neighbors, not including the object itself (default: 10)

    Returns
    -------
    similarity_matrix : scipy.sparse.csr_matrix
        The resulting similarity matrix
    """
    # Get pairwise distances
    dist_matrix = squareform(pdist(X, 'euclidean'))
    # Get kNN distances (+1 because self is not included in n_neighbors)
    knn_distances = np.sort(dist_matrix, axis=1)[:, n_neighbors + 1]
    # Get knn dist so that more than 'percentage' points have 'n_neighbors' neighbors
    knn_dist_sorted = np.sort(knn_distances)
    eps = knn_dist_sorted[int((X.shape[0] - 1) * percentage)]
    # Get neighbor graph
    similarity_matrix = radius_neighbors_graph(X, radius=eps)
    return similarity_matrix


def _get_symmetrically_normalized_adjacency_matrix(X: np.ndarray, n_neighbors: int = 10) -> scipy.sparse.csr_matrix:
    """
    Get a symmetrically normalized adjacency matrix of the kNN graph.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_neighbors : int
        The number of neighbors, not including the object itself (default: 10)

    Returns
    -------
    similarity_matrix : scipy.sparse.csr_matrix
        The resulting similarity matrix
    """
    # Get neighbor graph
    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance", include_self=False, n_jobs=-1)
    W = 0.5 * (W + W.T)
    d = np.sum(W, axis=1)
    # Convert np.matrix to np.array
    d = np.array(d).reshape(-1)
    d = np.power(d, -0.5)
    D = scipy.sparse.diags(d)
    similarity_matrix = D @ W @ D
    return similarity_matrix


class SpecialK(BaseEstimator, ClusterMixin):
    """
    Execute the SpecialK clustering procedure.
    SpecialK is able to autonomously identify a suitable number of clusters for spectral clustering procedures.
    Therefore, it uses probability bounds on the operator norm of centered, symmetric decompositions based on the matrix Bernstein concentration inequality.
    It iteratively increases the number of clusters until the probability of objects originating from two instead of one distribution does not increase further.
    Can be based on an epsilon-neighborhood adjacency matrix or the symmetrically normalized adjacency matrix of the kNN graph.

    Parameters
    ----------
    significance : float
        Threshold to decide if the samples originate from a single distribution or two distributions (default: 0.01)
    n_dimensions : int
        Dimensionality of the embedding (default: 200)
    similarity_matrix : str
        Defines the similarity matrix to use. Can be one of the following strings or a numpy array / scipy sparse csr matrix.
        If 'NAM', a neighborhood adjacency matrix is used.
        If 'SAM' a symmetrically normalized adjacency matrix is used (default: 'NAM')
    n_neighbors : int
        Number of neighbors for the construction of the similarity matrix. Does not include the object itself (default: 10)
    percentage : float
        The amount of data points that should have at least n_neighbors neighbors.
        Only relevant if use_neighborhood_adj_mat is set to True (default: 0.99)
    n_cluster_pairs_to_consider : int
        The number of cluster pairs to consider when calculating the probability boundary.
        The cluster pairs responsible for the highest cut. If None, all pairs will be considered.
        Smaller values for n_cluster_pairs_to_consider will decrease the computing time (default: 10)
    max_n_clusters : int
        Maximum number of clusters. Must be larger than n_clusters_init (default: np.inf)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels

    References
    ----------
    Hess, Sibylle, and Wouter Duivesteijn. "k is the magic number—inferring the number of clusters through nonparametric concentration inequalities."
    Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019, Würzburg, Germany, September 16–20, 2019, Proceedings, Part I. Springer International Publishing, 2020.
    """

    def __init__(self, significance: float = 0.01, n_dimensions: int = 200, similarity_matrix: str = 'NAM',
                 n_neighbors: int = 10, percentage: float = 0.99, n_cluster_pairs_to_consider: int = 10,
                 max_n_clusters: int = np.inf, random_state: np.random.RandomState = None, debug: bool = False):
        self.significance = significance
        self.n_dimensions = n_dimensions
        self.similarity_matrix = similarity_matrix
        self.n_neighbors = n_neighbors
        self.percentage = percentage
        self.n_cluster_pairs_to_consider = n_cluster_pairs_to_consider
        self.max_n_clusters = max_n_clusters
        self.random_state = check_random_state(random_state)
        self.debug = debug

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'SpecialK':
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
        self : SpecialK
            this instance of the SpecialK algorithm
        """
        n_clusters, labels = _specialk(X, self.significance, self.n_dimensions, self.similarity_matrix,
                                       self.n_neighbors, self.percentage, self.n_cluster_pairs_to_consider,
                                       self.max_n_clusters, self.random_state, self.debug)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        return self

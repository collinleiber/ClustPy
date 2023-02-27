"""
@authors:
Collin Leiber
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _multi_density_dbscan(X: np.ndarray, k: int, var: float, min_cluster_size: int) -> (int, np.ndarray, list):
    """
    Start the actual Multiple Density DBSCAN clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    k : int
        the number of neighbors to consider
    var : float
        Defines the factor that the density of a point may deviate from the average cluster density
    min_cluster_size : int
        The minimum cluster size (if a cluster is smaller, all contained points will be labeled as noise)

    Returns
    -------
    tuple : (int, np.ndarray, list)
        The identified number of clusters
        The cluster labels
        The final cluster densities
    """
    assert k <= X.shape[0], "The number of nearest neighbors k can not be larger than the number of data points"
    assert var >= 1, "var must be >= 1"
    assert min_cluster_size > 1, "min_cluster_size must be > 1"
    # Get k nearest neighbors and densities for each point
    nearest_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X)
    densities, knns = nearest_neighbors.kneighbors(X, n_neighbors=k + 1)
    knns = knns[:, 1:]
    densities = np.mean(densities[:, 1:], axis=1)
    # Order densities
    order = np.argsort(densities)
    # Start parameters
    labels = -np.ones(X.shape[0], dtype=np.int32)
    cluster_densities = []
    c_id = 0
    # Iterate over all points
    for p1 in order:
        if labels[p1] != -1:
            # Point is already assigned to a cluster
            continue
        cluster_points, cluster_density = _gather(p1, c_id, densities, knns, labels, var)
        # Check if cluster is large enough
        if len(cluster_points) >= min_cluster_size:
            c_id += 1
            cluster_densities.append(cluster_density)
        else:
            labels[cluster_points] = -1
    n_clusters = c_id
    return n_clusters, labels, cluster_densities


def _gather(p1: int, c_id: int, densities: np.ndarray, knns: np.ndarray, labels: np.ndarray, var: float) -> (
        list, float):
    """
    Expand the current cluster (consisting of a single most dense point).
    Check each added point's neighbors to see if their density is low enough to add them the cluster.

    Parameters
    ----------
    p1 : int
        The id of the starting point of the cluster (most dense point)
    c_id : int
        The cluster id of the current cluster
    densities : np.ndarray
        The densities of all points
    knns : np.ndarray
        The k-nearest neighbors of all points
    labels : np.ndarray
        The current cluster labels
    var : float
        Defines the factor that the density of a point may deviate from the average cluster density

    Returns
    -------
    tuple : (list, float)
        The ids of the points in this cluster
        The density of the cluster
    """
    # Add point to cluster and assign Label
    cluster_points = [p1]
    labels[p1] = c_id
    # Get neighbors of point 1
    neighbors = [kn for kn in knns[p1, :] if labels[kn] == -1]
    neighbors = _sort_neighbors_by_densities(neighbors, densities)
    # Set start density of the cluster
    cluster_density = densities[p1]
    while len(neighbors) > 0:
        p2 = neighbors.pop(0)
        if labels[p2] == -1:
            density_p2 = densities[p2]
            # Is density of point 2 high enough?
            if density_p2 <= var * cluster_density:
                # Add point to cluster and assign Label
                cluster_points.append(p2)
                labels[p2] = c_id
                # Update Cluster density
                cluster_density = (cluster_density * (len(cluster_points) - 1) + density_p2) / len(cluster_points)
                # Add new neighbors
                neighbors = _add_neighbors_to_neighbor_list(densities, labels, neighbors, knns[p2, :])
    return cluster_points, cluster_density


def _sort_neighbors_by_densities(neighbors: list, densities: np.ndarray) -> list:
    """
    Sort the available neighbors by their densities. Sort in ascending order.
    If densities are equal samples will be sorted by their id.

    Parameters
    ----------
    neighbors : list
        the ids of the neighbors
    densities : np.ndarray
        the densities

    Returns
    -------
    neighbors : list
        the ids of the neighbors sorted by densities
    """
    neighbors = sorted(neighbors, key=lambda x: (densities[x], x))
    return neighbors


def _add_neighbors_to_neighbor_list(densities: np.ndarray, labels: np.ndarray, current_neighbors: list,
                                    new_neighbors: np.ndarray) -> list:
    """
    Add the new neighbors to the neighboring list.
    Make sure that they are correctly sorted according to their density.
    Result should be equal to:
    current_neighbors += [kn for kn in new_neighbors if labels[kn] == -1]
    current_neighbors = list(set(current_neighbors))
    current_neighbors = _sort_neighbors_by_densities(neighbors, densities)

    Parameters
    ----------
    densities : np.ndarray
        The densities of all points
    labels : np.ndarray
        The current cluster labels
    current_neighbors : list
        The current list of neighbors of cluster objects
    new_neighbors : list
        The new neighbors that should be added to the neighbor list

    Returns
    -------
    current_neighbors : list
        The updated neighbor list
    """
    # ignore points that are already assigned
    new_neighbors = new_neighbors[labels[new_neighbors] == -1]
    # Sort new neighbors by density
    sorted_new_neighbors = _sort_neighbors_by_densities(new_neighbors, densities)
    # Get densities of current neighbors
    index = 0
    for p in sorted_new_neighbors:
        while index < len(current_neighbors) and (densities[p] > densities[current_neighbors[index]] or (
                densities[p] == densities[current_neighbors[index]] and p > current_neighbors[index])):
            index += 1
        if index == len(current_neighbors) or current_neighbors[index] != p:
            current_neighbors.insert(index, p)  # Add new neighbor, ignore if point already in neighbor list
        index += 1
    return current_neighbors


class MultiDensityDBSCAN(BaseEstimator, ClusterMixin):
    """
    The Multi Density DBSCAN algorithm.
    First, the densities of all data points will be calculated.
    Afterwards, clusters will be expanded starting with the most dense point.
    Density is defined as the average distance to the k-nearest neighbors.

    Parameters
    ----------
    k : int
        The number of nearest neighbors. Does not include the objects itself (default: 15)
    var : float
        Defines the factor that the density of a point may deviate from the average cluster density (default: 2.5)
    min_cluster_size : int
        The minimum cluster size (if a cluster is smaller, all contained points will be labeled as noise) (default: 2)

    Attributes
    ----------
    n_clusters_ : int
        The identified number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_densities_ : list
        The final cluster densities

    References
    ----------
    Ashour, Wesam, and Saad Sunoallah. "Multi density DBSCAN."
    International Conference on Intelligent Data Engineering and Automated Learning. Springer, Berlin, Heidelberg, 2011.
    """

    def __init__(self, k: int = 15, var: float = 2.5, min_cluster_size: int = 2):
        self.k = k
        self.var = var
        self.min_cluster_size = min_cluster_size

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MultiDensityDBSCAN':
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
        self : MultiDensityDBSCAN
            this instance of the Multi Density DBSCAN algorithm
        """
        n_clusters, labels, cluster_densities = _multi_density_dbscan(X, self.k, self.var, self.min_cluster_size)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_densities_ = cluster_densities
        return self

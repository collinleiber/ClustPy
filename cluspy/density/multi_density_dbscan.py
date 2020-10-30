"""
Ashour, Wesam, and Saad Sunoallah. "Multi density DBSCAN."
International Conference on Intelligent Data Engineering and
Automated Learning. Springer, Berlin, Heidelberg, 2011.
"""

from scipy.spatial.distance import squareform, pdist
import numpy as np


def _multi_density_dbscan(X, k, var, min_cluster_size, always_sort_densities):
    assert var >= 1, "var must be >= 1"
    assert min_cluster_size > 1, "min_cluster_size must be > 1"
    # Calculate distance matrix
    dist_matrix = squareform(pdist(X))
    # Get k nearest neighbors ids for each point
    knns = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]  # First is the same object
    # Calculate densities (mean knn dist)
    densities = np.array([np.mean(dist_matrix[i, knns[i, :]]) for i in range(X.shape[0])])
    # Order densities
    order = np.argsort(densities)
    # Start parameters
    labels = np.full(X.shape[0], -1)
    cluster_densities = []
    c_id = 0
    # Iterate over all points
    for p1 in order:
        if labels[p1] != -1:
            # Point is already assigned to a cluster
            continue
        cluster_points, cluster_density = _gather(p1, c_id, densities, knns, labels, var, always_sort_densities)
        # Check if cluster is large enough
        if len(cluster_points) >= min_cluster_size:
            c_id += 1
            cluster_densities.append(cluster_density)
        else:
            labels[cluster_points] = -1
    n_clusters = c_id
    return n_clusters, labels, cluster_densities


def _gather(p1, c_id, densities, knns, labels, var, always_sort_densities):
    # Add point to cluster and assign Label
    cluster_points = [p1]
    labels[p1] = c_id
    # Get neighbors of point 1
    neighbors = [kn for kn in knns[p1, :] if labels[kn] == -1]
    if always_sort_densities:
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
                neighbors += [kn for kn in knns[p2, :] if labels[kn] == -1]
                if always_sort_densities:
                    neighbors = _sort_neighbors_by_densities(neighbors, densities)
    return cluster_points, cluster_density


def _sort_neighbors_by_densities(neighbors, densities):
    neighbors = sorted(neighbors, key=lambda x: densities[x])
    return neighbors


class MultiDensityDBSCAN():

    def __init__(self, k, var=2.5, min_cluster_size=2, always_sort_densities=False):
        self.k = k
        self.var = var
        self.min_cluster_size = min_cluster_size
        self.always_sort_densities = always_sort_densities

    def fit(self, X):
        n_clusters, labels, cluster_densities = _multi_density_dbscan(X, self.k, self.var, self.min_cluster_size,
                                                                      self.always_sort_densities)
        self.n_clusters = n_clusters
        self.labels = labels
        self.cluster_densities = cluster_densities

"""
Kalogeratos, Argyris, and Aristidis Likas. "Dip-means: an
incremental clustering method for estimating the number of
clusters." Advances in neural information processing systems.
2012.

@authors Collin Leiber
"""

from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist, squareform
from cluspy.utils import dip, dip_pval, PVAL_BY_TABLE, PVAL_BY_BOOT, dip_boot_samples


def _dipmeans(X, pval_threshold, split_viewers_threshold, pval_strategy, n_boots, n_new_centers, max_n_clusters):
    # Calculate distance matrix
    data_dist_matrix = squareform(pdist(X, 'euclidean'))
    # Initialize parameters
    n_clusters = 0
    centers = np.mean(X, axis=0).reshape(1, -1)
    labels = np.zeros(X.shape[0])
    while n_clusters < max_n_clusters:
        n_clusters += 1
        # Default score is 0 for all clusters
        cluster_scores = np.zeros(n_clusters)
        ids_in_each_cluster = []
        for c in range(n_clusters):
            ids_in_cluster = np.where(labels == c)[0]
            ids_in_each_cluster.append(ids_in_cluster)
            # Get pairwise distances of points in cluster
            cluster_dist_matrix = data_dist_matrix[np.ix_(ids_in_cluster, ids_in_cluster)]
            # Calculate dip values for the distances of each point
            cluster_dips = np.array([dip(cluster_dist_matrix[p, :], just_dip=True, is_data_sorted=False) for p in
                                     range(ids_in_cluster.shape[0])])
            # Calculate p-values
            if pval_strategy == PVAL_BY_BOOT:
                boot_dips = dip_boot_samples(ids_in_cluster.shape[0], n_boots)
                cluster_pvals = np.array([np.mean(point_dip <= boot_dips) for point_dip in cluster_dips])
            else:
                cluster_pvals = np.array([dip_pval(point_dip, ids_in_cluster.shape[0], pval_strategy=pval_strategy,
                                                   n_boots=n_boots) for point_dip in cluster_dips])
            # Get split viewers (points with dip of distances <= threshold)
            split_viewers = cluster_dips[cluster_pvals <= pval_threshold]
            # Check if percentage share of split viewers in cluster is larger than threshold
            if split_viewers.shape[0] / ids_in_cluster.shape[0] >= split_viewers_threshold:
                # Calculate cluster score
                cluster_scores[c] = np.mean(split_viewers)
        # Get cluster with maximum score
        cluster_id_to_split = np.argmax(cluster_scores)
        # Check if any cluster has to be split
        if cluster_scores[cluster_id_to_split] > 0:
            # Split cluster using bisecting kmeans
            km = _execute_bisecting_kmeans(X, ids_in_each_cluster, cluster_id_to_split, centers,
                                           n_new_centers)
            labels = km.labels_
            centers = km.cluster_centers_
        else:
            break
    return n_clusters, centers, labels


def _execute_bisecting_kmeans(X, ids_in_each_cluster, cluster_id_to_split, centers, n_new_centers):
    # Prepare cluster for splitting
    old_center = centers[cluster_id_to_split, :]
    reduced_centers = np.delete(centers, cluster_id_to_split, axis=0)
    ids_in_cluster = ids_in_each_cluster[cluster_id_to_split]
    # Try to find kmeans result with smallest squared distances
    best_kmeans = None
    min_squared_dist = np.inf
    for i in range(n_new_centers):
        # Get random point in cluster as new center
        random_center = X[np.random.choice(ids_in_cluster), :].reshape(1, -1)
        # Calculate second new center as: new2 = old - (new1 - old)
        adjusted_center = (old_center - (random_center - old_center)).reshape(1, -1)
        # Run kmeans with new centers
        tmp_centers = np.r_[reduced_centers, random_center, adjusted_center]
        km = KMeans(n_clusters=tmp_centers.shape[0], init=tmp_centers, n_init=1)
        km.fit(X)
        # Check squared distances to find best kmeans result
        if km.inertia_ < min_squared_dist:
            min_squared_dist = km.inertia_
            best_kmeans = km
    return best_kmeans


class DipMeans():

    def __init__(self, pval_threshold=0, split_viewers_threshold=0.01, pval_strategy=PVAL_BY_TABLE, n_boots=2000,
                 n_new_centers=10, max_n_clusters=np.inf):
        self.pval_threshold = pval_threshold
        self.split_viewers_threshold = split_viewers_threshold
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.n_new_centers = n_new_centers
        self.max_n_clusters = max_n_clusters

    def fit(self, X):
        n_clusters, centers, labels = _dipmeans(X, self.pval_threshold, self.split_viewers_threshold,
                                                self.pval_strategy, self.n_boots, self.n_new_centers,
                                                self.max_n_clusters)
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.labels_ = labels

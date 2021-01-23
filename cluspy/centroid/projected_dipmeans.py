"""
Chamalis, Theofilos, and Aristidis Likas. "The projected dip-
means clustering algorithm." Proceedings of the 10th Hellenic
Conference on Artificial Intelligence. 2018.

@authors Collin Leiber
"""

import numpy as np
from cluspy.centroid.dipmeans import _execute_bisecting_kmeans
from sklearn.decomposition import PCA
from cluspy.utils import dip, dip_pval, PVAL_BY_TABLE


def _proj_dipmeans(X, pval_threshold, n_random_projections, pval_strategy, n_boots, n_new_centers, max_n_clusters):
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
            # Get projections
            projections = _get_projected_data(X[ids_in_cluster], n_random_projections)
            # Calculate dip values for the distances of each point
            cluster_dips = np.array([dip(projections[:, p], just_dip=True, is_data_sorted=False) for p in
                                     range(projections.shape[1])])
            # Calculate p-values of maximum dip
            pval = dip_pval(np.max(cluster_dips), ids_in_cluster.shape[0], pval_strategy=pval_strategy, n_boots=n_boots)
            # Calculate cluster score
            cluster_scores[c] = pval
        # Get cluster with maximum score
        cluster_id_to_split = np.argmin(cluster_scores)
        # Check if any cluster has to be split
        if cluster_scores[cluster_id_to_split] <= pval_threshold:
            # Split cluster using bisecting kmeans
            km = _execute_bisecting_kmeans(X, ids_in_each_cluster, cluster_id_to_split, centers,
                                           n_new_centers)
            labels = km.labels_
            centers = km.cluster_centers_
        else:
            break
    return n_clusters, centers, labels


def _get_projected_data(X, n_random_projections):
    # Execute PCA
    pca = PCA()
    pca_X = pca.fit_transform(X) if X.shape[0] > 1 else np.empty((X.shape[0], 0))
    # Add random projections
    random_projections = np.zeros((X.shape[0], n_random_projections))
    for i in range(n_random_projections):
        # Add random vector
        projection_vector = np.random.rand(X.shape[1])
        projected_X = np.dot(X, projection_vector)
        random_projections[:, i] = projected_X
    projections = np.c_[X, pca_X, random_projections]
    return projections


class ProjectedDipMeans():

    def __init__(self, pval_threshold=0, n_random_projections=10, pval_strategy=PVAL_BY_TABLE, n_boots=2000,
                 n_new_centers=10, max_n_clusters=np.inf):
        self.pval_threshold = pval_threshold
        self.n_random_projections = n_random_projections
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.n_new_centers = n_new_centers
        self.max_n_clusters = max_n_clusters

    def fit(self, X):
        n_clusters, centers, labels = _proj_dipmeans(X, self.pval_threshold, self.n_random_projections,
                                                     self.pval_strategy, self.n_boots, self.n_new_centers,
                                                     self.max_n_clusters)
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.labels_ = labels

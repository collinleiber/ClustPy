"""
Mautz, Dominik, et al. "Towards an optimal subspace for
k-means." Proceedings of the 23rd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining. 2017.
"""

from cluspy.alternative.nrkmeans import NrKmeans
import numpy as np
from cluspy.alternative.nrkmeans import _mdl_costs

def _get_n_clusters(X, max_n_clusters, add_noise_space, repetitions, mdl_for_noisespace,
                    outliers, max_iter, random_state):
    n_clusters = 2
    min_costs = np.inf
    best_subkmeans = None
    while n_clusters <= max_n_clusters:
        better_found = False
        for i in range(repetitions):
            subkmeans = SubKmeans(n_clusters, add_noise_space, mdl_for_noisespace, outliers, max_iter, random_state)
            subkmeans.fit(X)
            costs = _mdl_costs(X, subkmeans)[0]
            if costs < min_costs:
                better_found = True
                min_costs = costs
                best_subkmeans = subkmeans
        if better_found:
            n_clusters += 1
        else:
            break
    return best_subkmeans

class SubKmeans():

    def __init__(self, n_clusters, add_noise_space=True, mdl_for_noisespace=False, outliers=False,
                 max_iter=300, random_state=None):
        self.n_clusters = [n_clusters, 1] if add_noise_space else [n_clusters]
        self.mdl_for_noisespace = mdl_for_noisespace
        self.outliers = outliers
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        nrkmeans = NrKmeans(self.n_clusters, mdl_for_noisespace=self.mdl_for_noisespace, outliers=self.outliers,
                            max_iter=self.max_iter, random_state=self.random_state)
        nrkmeans.fit(X)
        self.labels_ = nrkmeans.labels_
        self.cluster_centers_ = nrkmeans.cluster_centers_
        self.V = nrkmeans.V
        self.P = nrkmeans.P
        self.m = nrkmeans.m
        self.scatter_matrices_ = nrkmeans.scatter_matrices_
        self.n_clusters = nrkmeans.n_clusters

class MDLSubKmeans():

    def __init__(self, max_n_clusters=np.inf, add_noise_space=True, repetitions=10, mdl_for_noisespace=True,
                 outliers=False, max_iter=300, random_state=None):
        self.max_n_clusters = max_n_clusters
        self.add_noise_space = add_noise_space
        self.repetitions = repetitions
        self.mdl_for_noisespace = mdl_for_noisespace
        self.outliers = outliers
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        subkmeans = _get_n_clusters(X, self.max_n_clusters, self.add_noise_space, self.repetitions,
                                    self.mdl_for_noisespace, self.outliers, self.max_iter, self.random_state)
        self.labels_ = subkmeans.labels_
        self.cluster_centers_ = subkmeans.cluster_centers_
        self.n_clusters = subkmeans.n_clusters
        self.V = subkmeans.V
        self.P = subkmeans.P
        self.m = subkmeans.m
        self.scatter_matrices_ = subkmeans.scatter_matrices_
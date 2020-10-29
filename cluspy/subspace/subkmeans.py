"""
Mautz, Dominik, et al. "Towards an optimal subspace for
k-means." Proceedings of the 23rd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining. 2017.
"""

from cluspy.alternative.nrkmeans import NrKmeans


class SubKmeans():

    def __init__(self, n_clusters, add_noise_space=True, mdl_for_noisespace=True, outliers=False,
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
        self.labels = nrkmeans.labels
        self.centers = nrkmeans.centers
        self.V = nrkmeans.V
        self.P = nrkmeans.P
        self.m = nrkmeans.m
        self.scatter_matrices = nrkmeans.scatter_matrices
        self.n_clusters = nrkmeans.n_clusters

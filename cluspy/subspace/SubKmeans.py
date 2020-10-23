"""
Mautz, Dominik, et al. "Towards an optimal subspace for
k-means." Proceedings of the 23rd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining. 2017.
"""

from cluspy.nonredundant.nrkmeans_vanilla import NrKmeans


class SubKmeans():

    def __init__(self, n_clusters, add_noise_space=True):
        self.n_clusters = [n_clusters, 1] if add_noise_space else [n_clusters]

    def fit(self, X):
        nrkmeans = NrKmeans(self.n_clusters)
        nrkmeans.fit(X)
        self.labels = nrkmeans.labels[0]  # TODO: has to be changed with new version of NrKmeans
        self.centers = nrkmeans.centers
        self.V = nrkmeans.V
        self.P = nrkmeans.P
        self.m = nrkmeans.m
        self.scatter_matrices = nrkmeans.scatter_matrices
        self.n_clusters = nrkmeans.n_clusters[0]

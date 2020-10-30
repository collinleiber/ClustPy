"""
Hamerly, Greg, and Charles Elkan. "Learning the k in k-means."
Advances in neural information processing systems. 2004.
"""

from pyclustering.cluster.gmeans import gmeans
from cluspy._pyclustering_wrapper import adjust_lists, adjust_labels


class GMeans():

    def __init__(self, init_n_clusters=1, tolerance=0.025, kmeans_repetitions=3):
        self.init_n_clusters = init_n_clusters
        self.tolerance = tolerance
        self.kmeans_repetitions = kmeans_repetitions

    def fit(self, X):
        gmeans_obj = gmeans(X, k_init=self.init_n_clusters, tolerance=self.tolerance, repeat=self.kmeans_repetitions)
        gmeans_obj.process()
        self.labels = adjust_labels(X.shape[0], gmeans_obj.get_clusters())
        self.centers = adjust_lists(gmeans_obj.get_centers())
        self.n_clusters = self.centers.shape[0]

"""
Pelleg, Dan, and Andrew W. Moore. "X-means: Extending
k-means with efficient estimation of the number of clusters."
Icml. Vol. 1. 2000.
"""

from pyclustering.cluster.xmeans import xmeans
from cluspy._wrapper_methods import pyclustering_adjust_labels
import numpy as np

class XMeans():

    def __init__(self, max_n_clusters=100, initial_centers=None, tolerance=0.025, kmeans_repetitions=3):
        self.max_n_clusters = max_n_clusters
        if initial_centers is not None and type(initial_centers) is not list:
            initial_centers = initial_centers.tolist()
        self.initial_centers = initial_centers
        self.tolerance = tolerance
        self.kmeans_repetitions = kmeans_repetitions

    def fit(self, X):
        xmeans_obj = xmeans(X, kmax=self.max_n_clusters, initial_centers=self.initial_centers, tolerance=self.tolerance,
                            repeat=self.kmeans_repetitions)
        xmeans_obj.process()
        self.labels = pyclustering_adjust_labels(X.shape[0], xmeans_obj.get_clusters())
        self.centers = np.array((xmeans_obj.get_centers()))
        self.n_clusters = self.centers.shape[0]

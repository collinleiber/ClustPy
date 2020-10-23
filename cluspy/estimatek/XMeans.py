"""
Pelleg, Dan, and Andrew W. Moore. "X-means: Extending
k-means with efficient estimation of the number of clusters."
Icml. Vol. 1. 2000.
"""

from pyclustering.cluster.xmeans import xmeans
import numpy as np


class XMeans():

    def __init__(self, max_n_clusters=np.inf):
        self.max_n_clusters = max_n_clusters

    def fit(self, X):
        xmeans_obj = xmeans(X, kmax=self.max_n_clusters)
        xmeans_obj.process()
        labels = np.zeros(X.shape[0])
        for i, l in enumerate(xmeans_obj.get_clusters()):
            labels[l] = i
        self.labels = labels
        self.centers = np.array(xmeans_obj.get_centers())
        self.n_clusters = self.centers.shape[0]

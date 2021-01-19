"""
Hamerly, Greg, and Charles Elkan. "Learning the k in k-means."
Advances in neural information processing systems. 2004.
"""

from pyclustering.cluster.fcm import fcm
from cluspy.utils._wrapper_methods import pyclustering_adjust_labels
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
import numpy as np
try:
    # Old sklearn versions
    from sklearn.cluster._kmeans import _k_init as kpp
except:
    # New sklearn versions
    from sklearn.cluster._kmeans import _kmeans_plusplus as kpp


class FuzzyCMeans():

    def __init__(self, n_cluster, initial_centers=None, tolerance=0.025, itermax=200, m=2):
        self.n_clusters = n_cluster
        self.initial_centers = initial_centers
        self.tolerance = tolerance
        self.itermax = itermax
        self.m = m

    def fit(self, X):
        if self.initial_centers is None:
            self.initial_centers = kpp(X, self.n_clusters, row_norms(X, squared=True),
                                       random_state=check_random_state(None))
        fuzzycmeans_obj = fcm(X, initial_centers=self.initial_centers, tolerance=self.tolerance,
                              itermax=self.itermax, m=self.m)
        fuzzycmeans_obj.process()
        self.labels_ = pyclustering_adjust_labels(X.shape[0], fuzzycmeans_obj.get_clusters())
        self.cluster_centers_ = np.array((fuzzycmeans_obj.get_centers()))
        self.membership_ = np.array((fuzzycmeans_obj.get_membership()))

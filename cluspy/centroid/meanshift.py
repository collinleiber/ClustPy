"""
Comaniciu, Dorin, and Peter Meer. "Mean shift: A robust
approach toward feature space analysis." IEEE Transactions on
pattern analysis and machine intelligence 24.5 (2002): 603-619.
"""

from sklearn.cluster import MeanShift as skMeanShift
from cluspy._wrapper_methods import sklearn_get_n_clusters

class MeanShift():

    def __init__(self, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None,
                 max_iter=300):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def fit(self, X):
        mean_shift = skMeanShift(self.bandwidth, self.seeds, self.bin_seeding, self.min_bin_freq, self.cluster_all,
                               self.n_jobs, self.max_iter)
        mean_shift.fit(X)
        self.labels = mean_shift.labels_
        self.centers = mean_shift.cluster_centers_
        self.n_clusters = sklearn_get_n_clusters(self.labels)

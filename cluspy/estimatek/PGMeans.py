"""
Feng, Yu, and Greg Hamerly. "PG-means: learning the number of
clusters in data." Advances in neural information processing
systems. 2007.
"""

import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import ks_2samp
import math

def _pgmeans(X, confidence, n_projections, n_samples, n_new_centers, max_n_clusters):
    n_clusters = 1
    input_centers = [np.mean(X, axis=0).reshape(1, -1)]
    highs = np.max(X, axis=0)
    lows = np.min(X, axis=0)
    while n_clusters <= max_n_clusters:
        best_gmm = None
        best_log_likelihood = -np.inf
        # Try different center possibilities
        for center in input_centers:
            gmm = GMM(n_components=n_clusters, n_init=1, means_init=center)
            gmm.fit(X)
            if gmm.lower_bound_ > best_log_likelihood:
                best_log_likelihood = gmm.lower_bound_
                best_gmm = gmm
        gmm_matches = True
        for _ in range(n_projections):
            # Get random projection
            projection_vector = np.random.rand(X.shape[1], 1)
            # Project data
            projected_X = np.dot(X, projection_vector).reshape(-1,)
            # Sample from model and project samples
            samples, _ = best_gmm.sample(n_samples)
            projected_samples = np.dot(samples, projection_vector).reshape(-1,)
            # Execute Kolmogorov-Smirnov test
            _, p_value = ks_2samp(projected_X, projected_samples)
            # Is hypothesis being rejected?
            if p_value < confidence:
                gmm_matches = False
                break
        if gmm_matches:
            break
        else:
            # add new center
            n_clusters += 1
            new_centers = np.random.uniform(lows, highs, (n_new_centers, X.shape[1]))
            input_centers = []
            for new_c in new_centers:
                input_centers.append(np.r_[best_gmm.means_, [new_c]])
    centers = best_gmm.means_
    labels = best_gmm.predict(X)
    return n_clusters, centers, labels


class PGMeans():
    def __init__(self, confidence=0.01, n_projections=None, n_samples=None, n_new_centers=10, max_n_clusters=np.inf):
        self.confidence = confidence
        if n_projections is None:
            self.n_projections = math.ceil(-2.6198 * np.log(confidence))
        else:
            self.n_projections = n_projections
        if n_samples is None:
            self.n_samples = math.ceil(3 / self.confidence)
        else:
            self.n_samples = n_samples
        self.n_new_centers = n_new_centers
        self.max_n_clusters = max_n_clusters

    def fit(self, X):
        n_clusters, centers, labels = _pgmeans(X, self.confidence, self.n_projections, self.n_samples, self.n_new_centers,
                                               self.max_n_clusters)
        self.n_clusters = n_clusters
        self.centers = centers
        self.labels = labels

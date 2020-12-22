"""
Feng, Yu, and Greg Hamerly. "PG-means: learning the number of
clusters in data." Advances in neural information processing
systems. 2007.

@authors Collin Leiber
"""

import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
import math


def _pgmeans(X, confidence, n_projections, n_samples, n_new_centers, random_centers, max_n_clusters):
    assert random_centers >= 0 and random_centers <= 1, "random_centers must be a value in the range [0, 1]"
    # Maximum and minimum values per feature. Needed for center sampling
    highs = np.max(X, axis=0)
    lows = np.min(X, axis=0)
    # Start parameters
    n_clusters = 0
    input_centers = [np.mean(X, axis=0).reshape(1, -1)]
    while n_clusters < max_n_clusters:
        n_clusters += 1
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
            projected_X = np.dot(X, projection_vector).reshape(-1, )
            # Project model - Alternative: Sample directly from model and project samples (should be slower)
            proj_gmm = _project_model(best_gmm, projection_vector, n_clusters)
            projected_samples, _ = proj_gmm.sample(n_samples)
            projected_samples = projected_samples.reshape(-1, )
            # Execute Kolmogorov-Smirnov test
            _, p_value = ks_2samp(projected_X, projected_samples)
            # Is hypothesis being rejected?
            if p_value < confidence:
                gmm_matches = False
                break
        if gmm_matches:
            break
        else:
            # Add new centers
            n_random_centers = math.floor(n_new_centers * random_centers)
            # Add centers farthest away from other centroids
            if n_random_centers < n_new_centers:
                new_centers = _add_non_random_centers(X, best_gmm.means_, n_new_centers - n_random_centers)
            # Add random centers
            if n_random_centers > 0:
                new_centers = np.r_[new_centers, np.random.uniform(lows, highs, (n_random_centers, X.shape[1]))]
            # Get centers for next iteration - combine gmm centers and new centers
            input_centers = []
            for new_c in new_centers:
                input_centers.append(np.r_[best_gmm.means_, [new_c]])
    centers = best_gmm.means_
    labels = best_gmm.predict(X)
    return n_clusters, centers, labels


def _project_model(gmm, projection_vector, n_clusters):
    # Project the model parameters
    proj_cov = np.array(
        [np.matmul(projection_vector.T, np.matmul(cov, projection_vector)) for cov in gmm.covariances_])
    proj_mean = np.matmul(gmm.means_, projection_vector)
    # Create new 1d GMM
    proj_gmm = GMM(n_components=n_clusters)
    proj_gmm.covariances_ = proj_cov
    proj_gmm.means_ = proj_mean
    proj_gmm.weights_ = gmm.weights_
    return proj_gmm


def _add_non_random_centers(X, centers, n_non_random_centers):
    new_centers = np.zeros((n_non_random_centers, X.shape[1]))
    # Get distance to nearest center for each point
    distances = np.min(cdist(X, centers), axis=1)
    for i in range(n_non_random_centers):
        # Get point farthest away from its nearest center
        new_c = X[np.argmax(distances), :]
        # Update minimum distances to centers
        distances_to_new_c = cdist(X, [new_c]).reshape(-1, )
        distances = np.minimum(distances, distances_to_new_c)
        # Add center
        new_centers[i, :] = new_c
    return new_centers


class PGMeans():
    
    def __init__(self, confidence=0.01, n_projections=None, n_samples=None, n_new_centers=10, random_centers=0.5,
                 max_n_clusters=np.inf):
        self.confidence = confidence
        if n_projections is None:
            n_projections = math.ceil(-2.6198 * np.log(confidence))
        self.n_projections = n_projections
        if n_samples is None:
            n_samples = math.ceil(3 / self.confidence)
        self.n_samples = n_samples
        self.n_new_centers = n_new_centers
        self.max_n_clusters = max_n_clusters
        self.random_centers = random_centers

    def fit(self, X):
        n_clusters, centers, labels = _pgmeans(X, self.confidence, self.n_projections, self.n_samples,
                                               self.n_new_centers, self.random_centers, self.max_n_clusters)
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.labels_ = labels

import numpy as np
import cluspy.MDLCosts as mdl
from cluspy.subspace.SubKmeans import SubKmeans


def _get_mdl_costs(X, subkmeans):
    data_dimensionality = X.shape[1]
    n_points = X.shape[0]
    # Costs of matrix V
    mdl_costs = mdl.mdl_costs_orthogonal_matrix(n_points, data_dimensionality)
    # Costs for dimensionality
    mdl_costs += mdl.mdl_costs_integer_value(subkmeans.m[0])
    # Number of clusters in subspace
    mdl_costs += mdl.mdl_costs_integer_value(subkmeans.n_clusters)
    # Costs for cluster centers
    mdl_costs += subkmeans.n_clusters * subkmeans.m[0] * mdl.mdl_costs_float_value(n_points)
    # Subspace Variance costs
    mdl_costs += mdl.mdl_costs_float_value(n_points)
    # Cluster assignment
    mdl_costs += n_points * mdl.mdl_costs_uniform_distribution(subkmeans.n_clusters)
    # Coding costs for each point
    mdl_costs += mdl.mdl_costs_gmm_single_covariance(subkmeans.V[:, subkmeans.P[0]], subkmeans.m[0],
                                                     subkmeans.scatter_matrices[0], n_points)
    # For noise space
    if len(subkmeans.m) > 1:
        mdl_costs += subkmeans.m[1] * mdl.mdl_costs_float_value(n_points)
        mdl_costs += mdl.mdl_costs_float_value(n_points)
        mdl_costs += mdl.mdl_costs_gmm_single_covariance(subkmeans.V[:, subkmeans.P[1]], subkmeans.m[1],
                                                         subkmeans.scatter_matrices[1], n_points)
    # return full and single subspace costs
    return mdl_costs


def _get_n_clusters(X, max_n_clusters, add_noise_space, repetitions):
    n_clusters = 2
    min_costs = np.inf
    best_subkmeans = None
    while n_clusters <= max_n_clusters:
        tmp_min_costs = np.inf
        tmp_best_subkmeans = None
        for i in range(repetitions):
            subkmeans = SubKmeans(n_clusters, add_noise_space)
            subkmeans.fit(X)
            costs = _get_mdl_costs(X, subkmeans)
            if costs < tmp_min_costs:
                tmp_min_costs = costs
                tmp_best_subkmeans = subkmeans
        if tmp_min_costs < min_costs:
            min_costs = tmp_min_costs
            best_subkmeans = tmp_best_subkmeans
            n_clusters += 1
        else:
            break
    return best_subkmeans


class MDLSubKmeans():

    def __init__(self, max_n_clusters=np.inf, add_noise_space=True, repetitions=10):
        self.max_n_clusters = max_n_clusters
        self.add_noise_space = add_noise_space
        self.repetitions = repetitions

    def fit(self, X):
        subkmeans = _get_n_clusters(X, self.max_n_clusters, self.add_noise_space, self.repetitions)
        self.labels = subkmeans.labels
        self.centers = subkmeans.centers
        self.n_clusters = subkmeans.n_clusters
        self.V = subkmeans.V
        self.P = subkmeans.P
        self.m = subkmeans.m

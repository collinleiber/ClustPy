# Implementation of DCSI by
# - Author: Jana Gauss - Github user `JanaGauss`
# - Source: https://github.com/JanaGauss/dcsi/
# - License: -

# Paper: DCSI -- An improved measure of cluster separability based on separation and connectedness
# Authors: Jana Gauss, Fabian Scheipl, and Moritz Herrmann
# Link: https://arxiv.org/abs/2310.12806

# Our modifications:
#    (1) translated from R to python


import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import LabelEncoder
from clustpy.metrics._metrics_utils import _check_length_data_and_labels, handle_noise


def dcsi_score(X, labels, min_points=5, noise_strategy="filter"):
    """
    Calculate DCSI-index for cluster validation, as defined in [1]

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)
    min_points : int
        min_points value to use.
    noise_strategy : str
        Strategy for handling noise (see clustpy.metrics.handle_noise). Must be one of:
        - "one_cluster"     : Assign all noise points to a single new cluster.
        - "singletons"      : Assign each noise point to its own cluster.
        - "filter"          : Remove all noise points.
        - "nearest_cluster" : Assign each noise point to nearest cluster.

    Returns
    -------
    dcsi : float,
        The resulting DCSI validity index.

    References:
    -----------
    .. [1] DCSI -- An improved measure of cluster separability based on separation and connectedness
           by Jana Gauss, Fabian Scheipl, and Moritz Herrmann
           see https://arxiv.org/abs/2310.12806
    """

    assert noise_strategy != "keep", "DCSI score is not defined for noise points."
    labels, X, _ = handle_noise(labels, strategy=noise_strategy, X=X)
    labels = LabelEncoder().fit_transform(labels)
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(
        type(labels)
    )

    dist = squareform(pdist(X)) ** 2
    cluster_labels = np.unique(labels)
    n_clusters = len(cluster_labels)
    dcsi = 0
    MST = {}
    CORE_PTS = {}
    core_labels = []
    for i in range(0, n_clusters):
        # indices of objects in cluster i
        objects_cl = np.where(labels == cluster_labels[i])[0]
        # distance in the cluster
        dist_i = dist[np.ix_(objects_cl, objects_cl)]
        epsilon = calculate_epsilon(dist_i, 2 * min_points)
        CORE_PTS[cluster_labels[i]] = core_points(dist_i, epsilon, min_points)
        # the official implementation only looks at core points (line 249 official git for i in unique(labels_core))
        if len(CORE_PTS[cluster_labels[i]]) == 0:
            continue
        core_labels.append(cluster_labels[i])
        dist_i = dist_i[np.ix_(CORE_PTS[cluster_labels[i]], CORE_PTS[cluster_labels[i]])]
        MST[cluster_labels[i]] = minimal_spanning_tree(dist_i)

    for i in range(0, n_clusters - 1):
        if i in core_labels:
            for j in range(i + 1, n_clusters):
                if j in core_labels:
                    part = pairwise_dcsi(MST, CORE_PTS, X, labels, cluster_labels[i], cluster_labels[j])

                    dcsi = dcsi + part
    dcsi = (2 / (n_clusters * (n_clusters - 1))) * dcsi

    return dcsi


def calculate_epsilon(dist_i, k):
    distances = []
    for i in range(0, dist_i.shape[0]):
        dists = np.unique(dist_i[i])
        if k >= len(dists):
            distances.append(dists[-1])
        else:
            distances.append(dists[k])
    epsilon = np.median(distances)
    return epsilon


def core_points(dist, epsilon, min_pts):
    neighborhoods = []
    for i in range(len(dist)):
        row = []
        for j in range(len(dist)):
            if i != j:
                if dist[i, j] <= epsilon:
                    row.append(dist[i, j])
        neighborhoods.append(row)
    core_pts = [i for i in range(len(neighborhoods)) if len(neighborhoods[i]) > min_pts - 1]
    return core_pts


def pairwise_dcsi(MST, CORE_PTS, data, partition, i, j):
    sep_dcsi = pairwise_separation(CORE_PTS, data, partition, i, j)
    conn_dcsi = pairwise_connectedness(MST, i, j)
    q = sep_dcsi / conn_dcsi
    return q / (1 + q)


def pairwise_separation(CORE_POINTS, data, labels, i, j):
    # distances between core points in between C_i and C_j
    # subset to include internal nodes of cluster i only
    subset_i = data[labels == i, :]
    core_pts_i = CORE_POINTS[i]
    subset_i = subset_i[core_pts_i]
    # subset to include internal nodes of cluster j only
    subset_j = data[labels == j, :]
    core_pts_j = CORE_POINTS[j]
    subset_j = subset_j[core_pts_j]
    sep_dcsi_list = cdist(subset_i, subset_j, metric="euclidean") ** 2
    sep_dcsi = np.min(sep_dcsi_list)
    return sep_dcsi


def pairwise_connectedness(MST, i, j):
    conn_dcsi = max(cluster_conn(MST, i), cluster_conn(MST, j))
    return conn_dcsi


def cluster_conn(MST, i):
    """
    Conn_dcsi(C_i) = max d(x_i, x_j), (x_i, x_j) in V

    :param MST:
    :param i:
    :return:
    """
    # maximum edge weight of MST
    conn_dcsi = np.max(MST[i])
    return conn_dcsi


def minimal_spanning_tree(dist_i):
    # transform to array
    dist = np.array(dist_i)
    # calculate minimal spanning tree and extract adjacency matrix
    # this calculates Kruskal
    mst = minimum_spanning_tree(dist).toarray()
    # mst is upper triangular matrix, make it symmetric
    mst_temp = mst + mst.T
    return mst_temp

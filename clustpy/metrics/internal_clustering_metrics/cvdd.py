"""
@authors:
Lena Krieger and Pascal Weber
"""

# Paper: An Internal Validity Index Based on Density-Involved Distance
# Authors: Lianyu Hu and Caiming Zhong
# Link: https://ieeexplore.ieee.org/document/8672850


import numpy as np
from scipy.spatial.distance import squareform, pdist
from clustpy.utils.dctree import DCTree
from sklearn.preprocessing import LabelEncoder
from clustpy.metrics._metrics_utils import _check_length_data_and_labels, handle_noise


def density_estimation(d, KNNG, k):
    ## Equation 1
    Den = []
    # for each point in the dataset
    for i in range(len(d)):
        den_i = [d[i, neighbor] for neighbor in KNNG[i]]
        den_i = sum(den_i)
        den_i = den_i / k
        Den.append(den_i)
    return Den

def outlier_factor(Den):
    ## Equation 2
    fDen = [den_i / max(Den) for den_i in Den]
    return fDen

def mutual_density_factor(Rel):
    ## Equation 5
    n = len(Rel)
    fRel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            temp = Rel[i, j] + Rel[j, i] - 2
            temp = np.abs(temp)
            temp = temp * -1
            fRel[i, j] = 1 - np.exp(temp)
    return fRel

def relative_density(Den):
    n = len(Den)
    Rel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Rel[i, j] = Den[i] / Den[j]
    return Rel

def _nD(Den):
    n = len(Den)
    nD = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            nD[i, j] = Den[i] + Den[j]
    return nD

def connectivity_distance(drD):
    ## Equation 8
    conD = DCTree(drD, min_points=1, precomputed=True).dc_distances()
    return conD

def dr_distance(d, RelD):
    n = len(d)
    drD = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            drD[i, j] = d[i, j] + RelD[i, j]
    return drD

def rel_D(fRel, nD):
    n = len(fRel)
    relD = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            relD[i, j] = fRel[i, j] * nD[i, j]
    return relD

def density_involved_dist(d, k):
    ## Equation 9
    n = len(d)
    # indices of k nearest neighbors
    KNNG = KNearestNeighborGraph(d, k)
    # list of mean k nearest neighbor distance for each point
    Den = density_estimation(d, KNNG, k)
    # Normalize towards maximum Den_i
    fDen = outlier_factor(Den)
    # this is where the exception might be thrown
    Rel = relative_density(Den)

    fRel = mutual_density_factor(Rel)
    # nd is a reference
    nD = _nD(Den)
    relD = rel_D(fRel, nD)
    drD = dr_distance(d, relD)
    conD = connectivity_distance(drD)
    DD = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            DD[i, j] = np.sqrt(fDen[i] * fDen[j]) * conD[i, j]
    return DD

def path_distance(d):
    ## Equation 7
    pD = DCTree(d, min_points=1, precomputed=True).dc_distances()
    return pD

def KNearestNeighborGraph(d, k):
    N = len(d)
    KNNG = [[] for _ in range(N)]
    dist = d.copy()
    dist[np.eye(N) == 1] = np.inf
    for i in range(N):
        dists = dist[i, :]
        indices = np.argsort(dists)
        KNNG[i] = list(indices[:k])
    return KNNG

def CVDD(sep, com):
    ## Equation 13
    return np.sum(sep) / np.sum(com)

def cvdd_score(
    X,
    labels,
    num_of_neighbors=7,
    noise_strategy="keep",
):
    """
    Calculate CVDD-index for cluster validation, as defined in [1]

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)
    noise_strategy : str
        Strategy for handling noise. Must be one of:
        - "keep"               : Keep all noise points as they are (default).
        - "as_one_cluster"     : Assign all noise points to a single new cluster.
        - "as_singletons"      : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points.
        - "to_nearest_cluster" : Assign each noise point to nearest cluster.

    Returns
    -------
    cvdd : float,
        The resulting CVDD validity index.

    References:
    -----------
    .. [1] An Internal Validity Index Based on Density-Involved Distance
           by Lianyu Hu and Caiming Zhong
           see https://ieeexplore.ieee.org/document/8672850
    """
    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)
    labels[labels != -1] = LabelEncoder().fit_transform(labels[labels != -1])
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(type(labels))

    unique_labels = np.unique(labels)
    num_cluster = len(unique_labels)
    d = squareform(pdist(X, metric="minkowski", p=2))
    try:
        DD = density_involved_dist(d, num_of_neighbors)
    except ZeroDivisionError:
        return 0

    sep = np.zeros(num_cluster)
    com = np.zeros(num_cluster)
    i = 0
    for label in unique_labels:
        if label == -1:
            continue
        a = (labels == label).nonzero()[0]
        b = (labels != label).nonzero()[0]
        n = len(a)
        if n > 0:
            # compute the separation sep[i]
            DD1 = np.take(DD, a, axis=0)
            DD2 = np.take(DD1, b, axis=1)
            sep[i] = np.min(DD2)

            d1 = np.take(d, a, axis=0)
            d2 = np.take(d1, a, axis=1)
            pD_i = path_distance(d2)
            mean_i = np.mean(pD_i)

            d_2 = abs(pD_i - mean_i) ** 2
            var = d_2.sum() / (n - 1)
            std_i = var ** 0.5

            com[i] = (1 / n) * std_i * mean_i
            # print('Sep and com from i {}'.format(i))
            # print(sep[i])
            # print(com[i])
            i = i + 1

    score = CVDD(sep, com)
    return score

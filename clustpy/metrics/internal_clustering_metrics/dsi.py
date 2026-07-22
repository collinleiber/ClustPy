# Implementation of DSI by
# - Author: Shuyue Guan - Github user `ShuyueG`
# - Source: https://github.com/ShuyueG/CVI_using_DSI/blob/main/cluster_DSI_example.py
# - License: GPL-3.0 licence (https://github.com/ShuyueG/CVI_using_DSI/blob/main/LICENSE)

# Paper: An Internal Cluster Validity Index Using a Distance-based Separability Measure
# Authors: Shuyue Guan and Murray Loew
# Link: https://ieeexplore.ieee.org/document/9288314


import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from clustpy.metrics._metrics_utils import _check_length_data_and_labels, handle_noise


def dists(data, dist_func=distance.euclidean):  # compute ICD
    num = data.shape[0]
    data = data.reshape((num, -1))
    dist = []
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            dist.append(dist_func(data[i], data[j]))
    return np.array(dist)


def dist_btw(a, b, dist_func=distance.euclidean):  # compute BCD
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    dist = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            dist.append(dist_func(a[i], b[j]))
    return np.array(dist)


def dsi_score(X, labels, noise_strategy="filter"):  # KS test on ICD and BCD
    """
    Calculate DSI-index for cluster validation, as defined in [1]

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)
    noise_strategy : str
        Strategy for handling noise. Must be one of:
        - "as_one_cluster"     : Assign all noise points to a single new cluster.
        - "as_singletons"      : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points (default).
        - "to_nearest_cluster" : Assign each noise point to nearest cluster.

    Returns
    -------
    dsi : float,
        The resulting DSI validity index.

    References:
    -----------
    .. [1] An Internal Cluster Validity Index Using a Distance-based Separability Measure
           by Shuyue Guan and Murray Loew
           see https://ieeexplore.ieee.org/document/9288314
    """
    assert noise_strategy != "keep", "DSI score is not defined for noise points."
    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)
    labels = LabelEncoder().fit_transform(labels)
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(type(labels))

    classes = np.unique(labels)
    SUM = 0
    for c in classes:
        pos = X[np.squeeze(labels == c)]
        neg = X[np.squeeze(labels != c)]

        dist_pos = dists(pos)
        distbtw = dist_btw(pos, neg)
        D, _ = ks_2samp(dist_pos, distbtw)  # KS test
        SUM += D
    SUM = SUM / classes.shape[0]  # normed: b/c ks_2samp ranges [0,1]
    return SUM

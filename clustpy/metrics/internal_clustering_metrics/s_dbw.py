# Implementation of S_Dbw by
# - Author: Alexander Lashkov - Github user `alashkov83`
# - Source: https://github.com/alashkov83/S_Dbw/blob/master/s_dbw/s_dbw.py
# - License: MIT License (https://github.com/alashkov83/S_Dbw/blob/master/LICENSE)

# Paper: A density-based cluster validity approach using multi-representatives
# Link: https://doi.org/10.1016/j.patrec.2007.12.011
# Authors: Maria Halkidi and Michalis Vazirgiannis

# Our modifications:
#    (1) 503- 513 to move labeling from to zero, e.g., 1,3,5 -> 0,1,2


import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from clustpy.metrics._metrics_utils import _check_length_data_and_labels, handle_noise


def calc_nearest_points(X, labels, unique_labels, centroids, metric):
    """
    Calculation of coordinates of clusters points closest to their geometric centers

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    unique_labels : array-like (n_clusters,),
        Unique labels of clusters (k > 0)
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    metric : str,
        The distance metric, can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski',
        'yule'. Default is 'euclidean'.

    Returns
    -------
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    """
    centroids_p = dict()
    for i in unique_labels:
        dist = cdist(X[labels == i], np.array(centroids[i], ndmin=2), metric=metric)
        centroids_p[i] = X[labels == i][np.argmin(dist)]
    return centroids_p


def calc_centroids(X, unique_labels, labels, centr):
    """
    Calculation of coordinates of centroids

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    unique_labels : array-like (n_clusters,),
        Unique labels of clusters (k > 0)
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    centr : str,
        Cluster center calculation method (mean (default) or median)

    Returns
    -------
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    """
    centers = dict()
    for i in unique_labels:
        if centr == "mean":
            centers[i] = np.mean(X[labels == i], axis=0)
        elif centr == "median":
            centers[i] = np.median(X[labels == i], axis=0)
    return centers


def centroid_distance(unique_labels, centroids, metric):
    """
    Calculation of distances between cluster centers given by: (Dmax/Dmin) * sum{forall i in 1:|C|} 1 /( sum{forall j in 1:|C|} ||vi - vj|| )

    Parameters
    ----------
    unique_labels : array-like (n_clusters,),
        Unique labels of clusters (k > 0)
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    metric : str,
        The distance metric, can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski',
        'yule'. Default is 'euclidean'.

    Returns
    -------
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    """
    sum_dist = 0
    max_dist = 0
    min_dist = 1e6
    for i in unique_labels:
        module_dist = 0
        for j in unique_labels:
            dist = cdist(np.array(centroids[j], ndmin=2), np.array(centroids[i], ndmin=2), metric=metric)[0, 0]
            module_dist += dist
            if dist > max_dist:
                max_dist = dist
            if min_dist > dist > 0:
                min_dist = dist
        sum_dist += 1 / module_dist
    distance = (max_dist / min_dist) * sum_dist
    return distance


def calc_density(X, centroids, labels, stdev, clusters_list, method, density_dict=None, lambd=0.7):
    """
    Compute the density of one or two cluster(depend on cluster_list)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    stdev : float,
        Average standard deviation of clusters (for Halkidi method [1])
    clusters_list : list,
        List contains one or two No. of cluster
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]
    density_dict : dict(),
        List contains density of each cluster for calculate muij [3]
    lambd : float,
        Lambda coefficient is a positive constant between 0 and 1 (default: 0.7, see [3]

    Returns
    -------
    score : float
        Density of one or two cluster

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, "Clustering validity assessment: Finding the optimal partitioning
        of a data set," in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD'2003, Seoul, Korea, April 30–May 2,
        2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """

    density = 0
    center_p1 = centroids[clusters_list[0]]
    if len(clusters_list) == 2:
        center_p2 = centroids[clusters_list[1]]
        if method == 'Kim' or method == 'Tong':
            sigmai = np.std(X[labels == clusters_list[0]], axis=0)
            sigmaj = np.std(X[labels == clusters_list[1]], axis=0)
            sigmaij = (sigmai + sigmaj) / 2
            ni = X[labels == clusters_list[0]].shape[0]
            nj = X[labels == clusters_list[1]].shape[0]
            nij = ni + nj
            if method == 'Tong':
                center_v = lambd * (center_p1 * nj + center_p2 * ni) / nij + \
                           (1 - lambd) * ((center_p1 * density_dict[clusters_list[0]] +
                                           center_p2 * density_dict[clusters_list[1]]) /
                                          (density_dict[clusters_list[0]] + density_dict[clusters_list[1]]))
            else:
                center_v = (center_p1 + center_p2) / 2
        else:
            center_v = (center_p1 + center_p2) / 2
    else:
        center_v = center_p1
        if method == 'Kim' or method == 'Tong':
            sigmaij = np.std(X[labels == clusters_list[0]], axis=0)
            nij = X[labels == clusters_list[0]].shape[0]
    if method == 'Halkidi':
        for i in clusters_list:
            temp = X[labels == i]
            for j in temp:
                if np.linalg.norm(j - center_v) <= stdev:
                    density += 1
    elif method == 'Kim' or method == 'Tong':
        CI = 1.96 * sigmaij / math.sqrt(nij)
        for i in clusters_list:
            temp = X[labels == i]
            for j in temp:
                if np.all(np.abs(j - center_v) <= CI):
                    density += 1

    return density


def Dens_bw(X, centroids, labels, unique_labels, method='Halkidi'):
    """
    Compute Inter-cluster Density (ID) - It evaluates the average density in the region among clusters in relation
    with the density of the clusters. The goal is the density among clusters to be significant low in comparison with
    the density in the considered clusters [1].

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    centroids : dict-like,
        Key: cluster index, Value: n_features-dimensional data point
    unique_labels : array-like (n_clusters,)
        Unique labels of clusters (k > 0)
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]

    Returns
    -------
    score : float
        Inter-cluster Density

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, "Clustering validity assessment: Finding the optimal partitioning
        of a data set," in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD'2003, Seoul, Korea, April 30–May 2,
        2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """
    k = np.size(unique_labels)
    density_dict = dict()
    result = 0
    stdev = 0
    if method == 'Halkidi':
        for i in unique_labels:
            std_matrix_i = np.std(X[labels == i], axis=0)
            stdev += math.sqrt(np.dot(std_matrix_i.T, std_matrix_i))
        stdev = math.sqrt(stdev) / k
    for i in unique_labels:
        density_dict[i] = calc_density(X, centroids, labels, stdev, [i], method)

    count_zeros = 0
    for cluster in density_dict.keys():
        if density_dict[cluster] == 0:
            count_zeros += 1

        if count_zeros > 1:
            raise ValueError('The density for two or more clusters to equal zero.')

    for i in unique_labels:
        for j in unique_labels:
            if i == j:
                continue
            result += calc_density(X, centroids, labels, stdev, [i, j], method, density_dict) / max(density_dict[i],
                                                                                                    density_dict[j])

    return result / (k * (k - 1))


def Scat(X, unique_labels, labels, method):
    """
    Calculate intra-cluster variance (Average scattering for clusters).
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    unique_labels : array-like (n_clusters,)
        Unique labels of clusters (k > 0)
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]

    Returns
    -------
    score : float
        Average scattering for clusters

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, "Clustering validity assessment: Finding the optimal partitioning
        of a data set," in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD'2003, Seoul, Korea, April 30–May 2,
     2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """
    k = np.size(unique_labels)
    theta_s = np.std(X, axis=0)
    theta_s_2norm = math.sqrt(np.dot(theta_s.T, theta_s))
    sum_theta_2norm = 0
    if method == 'Halkidi':
        for i in unique_labels:
            theta_i = np.std(X[labels == i], axis=0)
            sum_theta_2norm += math.sqrt(np.dot(theta_i.T, theta_i))
        result = sum_theta_2norm / (theta_s_2norm * k)
    else:
        n = len(labels)
        for i in unique_labels:
            ni = X[labels == i].shape[0]
            theta_i = np.std(X[labels == i], axis=0)
            sum_theta_2norm += ((n - ni) / n) * math.sqrt(np.dot(theta_i.T, theta_i))
        result = sum_theta_2norm / (theta_s_2norm * k)
        if method == 'Tong':
            result = sum_theta_2norm / (theta_s_2norm * (k - 1))
    return result


def s_dbw_score(
    X,
    labels,
    centers_id=None,
    method="Halkidi",
    noise_strategy="filter",
    centr="mean",
    nearest_centr=True,
    metric="euclidean",
):
    """
    Compute the S_Dbw validity index
    S_Dbw validity index is defined by equation:
    S_Dbw = scatt + dens
    where scatt - means average scattering for clusters and dens - inter-cluster density.
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample (-1 - for noise).
    centers_id : array-like, shape (n_samples,)
        The center_id of each cluster's center. If None - cluster's center calculate automatically.
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]
    noise_strategy : str
        Strategy for handling noise. Must be one of:
        - "as_one_cluster"     : Assign all noise points to a single new cluster (default).
        - "as_singletons"      : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points.
        - "to_nearest_cluster" : Assign each noise point to nearest cluster.
    centr : str,
        cluster center calculation method (mean (default) or median)
    nearest_centr : bool,
        The centroid corresponds to the cluster point closest to the geometric center (default: True).
    metric : str,
        The distance metric, can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski',
        'yule'. Default is 'euclidean'.

    Returns
    -------
    score : float
        The resulting S_DBw score.

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, "Clustering validity assessment: Finding the optimal partitioning
        of a data set," in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD'2003, Seoul, Korea, April 30–May 2,
        2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """
    assert noise_strategy != "keep", "S_DBw score is not defined for noise points."
    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)
    labels = LabelEncoder().fit_transform(labels)
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(type(labels))

    unique_labels = np.unique(labels)
    if centers_id:
        centroids = dict()
        for index, label in enumerate(unique_labels):
            centroids[label] = X[centers_id[index]]
    else:
        centroids = calc_centroids(X, unique_labels, labels, centr)
        if nearest_centr:
            centroids = calc_nearest_points(X, labels, unique_labels, centroids, metric)

    sdbw = Dens_bw(X, centroids, labels, unique_labels, method) + Scat(X, unique_labels, labels, method)
    return sdbw


def sd_score(X, labels, k=1.0, centers_id=None, noise_strategy="filter", centr='mean', nearest_centr=True, metric='euclidean'):
    """
    Compute the SD validity index
    SD validity index is defined by equation:
    SD = k*scatt + distance
    where scatt - means average scattering for clusters and distance - distances between cluster centers,
    k = distances(Cmax), where Cmax - maximum number of clusters.
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample (-1 - for noise).
    k: float
        The weighting coefficient equal to distances(Cmax). It is necessary for evaluating solutions
         with vary number of clusters because distances(C) depends on number of clusters.
    centers_id : array-like, shape (n_samples,)
        The center_id of each cluster's center. If None - cluster's center calculate automatically.
    noise_strategy : str
        Strategy for handling noise. Must be one of:
        - "as_one_cluster"     : Assign all noise points to a single new cluster.
        - "as_singletons"      : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points (default).
        - "to_nearest_cluster" : Assign each noise point to nearest cluster.
    centr : str,
        cluster center calculation method (mean (default) or median)
    nearest_centr : bool,
        The centroid corresponds to the cluster point closest to the geometric center (default: True).
    metric : str,
        The distance metric, can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski',
        'yule'. Default is 'euclidean'.

    Returns
    -------
    score : float
        The resulting SD score.

    References:
    -----------
    .. [4] Halkidi, Maria & Vazirgiannis, Michalis & Batistakis, Yannis. (2000).
    Quality Scheme Assessment in the Clustering Process. LNCS (LNAI). 1910. 265-276.
    """

    assert noise_strategy != "keep", "DSI score is not defined for noise points."
    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)
    labels = LabelEncoder().fit_transform(labels)
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(type(labels))

    unique_labels = np.unique(labels)
    if centers_id:
        centroids = dict()
        for index, label in enumerate(unique_labels):
            centroids[label] = X[centers_id[index]]
    else:
        centroids = calc_centroids(X, unique_labels, labels, centr)
        if nearest_centr:
            centroids = calc_nearest_points(X, labels, unique_labels, centroids, metric)
    sd = k * Scat(X, unique_labels, labels, 'Halkidi') + centroid_distance(unique_labels, centroids, metric)
    return sd

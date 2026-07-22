"""
@authors:
Lena Krieger and Pascal Weber
"""

# Paper: A Novel Cluster Validity Index Based on Local Cores
# Authors: Dongdong Cheng, Qingsheng Zhu, Jinlong Huang, Quanwang Wu, and Lijun Yang
# Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8424512


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from clustpy.metrics._metrics_utils import _check_length_data_and_labels, handle_noise

def nan_searching(X, dist):
    # NaN seraching Algorithm 1
    # number of datapoints
    N = X.shape[0]
    dist_ = dist.copy()
    dist_[np.eye(N) == 1] = np.inf
    # the radius for neighbors
    r = 0
    # flag necessary for while true
    flag = 0
    nb = np.zeros(N)
    # number of reverse neighbors
    nb_r = [np.zeros(N) for i in range(N)]
    # r nearest neighbors
    NNr = [[] for i in range(N)]
    # reverse neighbors
    RNNr = [[] for i in range(N)]
    # local neighbor
    LN = [[] for i in range(N)]
    # count r denotes the number of elements not having a reverse neighbor
    count = []
    # while stop condition not met
    while flag == 0:
        # for every point
        for i in range(N):
            # take the distances from i to every other point
            dists = dist_[i, :]
            # indices sorted by distance (kth nearest neighbor is at indices[k])
            indices = np.argsort(dists)
            # get position of rth nearest neighbor
            y = indices[r]
            # number of neighbors for y increases
            nb[y] = nb[y] + 1
            # nearest neighbors from i includes y
            NNr[i].append(y)
            # reverse neighbors from y should include i
            RNNr[y].append(i)

        nb_r[r] = nb
        mu = np.max(nb_r[r])
        # append number of elements not having a reverse neighbor at radius r
        count.append(list(nb_r[r]).count(0))
        # stop condition either every element has a reverse neighbor or the number of reverse neighbors does not
        # change anymore
        if count[-1] == 0 or (len(count) > 1 and count[-1] == count[-2]):
            flag = 1
        # increase radius
        r = r + 1
    # reduce lambda since it is increase once more after stopping condition is met
    lamd = r - 1
    # for every datapoint
    for i in range(N):
        # local neighbor of i are neighbors added until r= lambda
        LN[i] = NNr[i]
    return lamd, LN, NNr, mu


def local_density(dist, i, LN):
    ## Equation 3
    # NNr nearest neighbors
    # rho_i = mu/(sum_j(dist(i,j))) with j being the mu-th nearest neighbors
    mu = len(LN[i])
    dists = [dist[i, j] for j in LN[i]]
    return mu / sum(dists)


def LORE(LN, rho, X, dist):
    #### Algorithm 2
    # representative of each point
    # number of datapoints
    N = len(X)
    rep = [[] for i in range(N)]
    local_cores = []
    # for each point in the dataset
    for i in range(N):
        # find point y with maximum density in LN i
        maxdens = 0
        max_index = 0
        # print('LN i {}'.format(i))
        # print(LN[i])
        # for each point j in Local neighbors of i
        for j in LN[i]:
            # print('rho j {}'.format(j))
            # print(rho[j])
            # if maxdens is smaller than local dens at j
            if maxdens < rho[j]:
                # set new maxdens
                maxdens = rho[j]
                # save index
                max_index = j
        # for each point p in Local neighbors of i
        for p in LN[i]:
            # if p does not have a representative
            if len(rep[p]) == 0:
                # representative is local neighbor of i with max density
                rep[p] = [max_index]
            # if p has a representative and the represenative is not y
            elif len(rep[p]) != 0 and rep[p][0] != max_index:
                # RCR (Representative competition rule)
                # representative is the one closer to p
                if dist[p][max_index] < dist[p][rep[p][0]]:
                    rep[p] = [max_index]
            # for each datapoint
            for z in range(N):
                # if the representative is p
                if len(rep[z]) != 0 and rep[z][0] == p:
                    # RTR (Representative Transfer Rule)
                    # if the representative from z is p and the representative from p is y
                    # then the representative from z is y ( z->p, p->y --> z->y)
                    rep[z] = [max_index]

    # print(LN[316])
    # for each datapoint
    for i in range(N):
        # if the representative of i ist i itself
        if len(rep[i]) != 0 and rep[i][0] == i:
            # the i is local core
            local_cores.append(i)
    return local_cores, rep


def lccv_score(X, labels, noise_strategy="keep"):
    """
    Calculate LCCV-index for cluster validation, as defined in [1]

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
    lccv : float,
        The resulting LCCV validity index.

    References:
    -----------
    .. [1] A Novel Cluster Validity Index Based on Local Cores
           by Dongdong Cheng, Qingsheng Zhu, Jinlong Huang, Quanwang Wu, and Lijun Yang
           see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8424512
    """
    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)
    labels[labels != -1] = LabelEncoder().fit_transform(labels[labels != -1])
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(type(labels))

    # euclidean pairwise distance
    dist = cdist(X, X)
    # number of datapoints
    N = len(dist)
    # radius, local neighbors
    lamd, LN, NNr, mu = nan_searching(X, dist)
    # local density - value for each datapoint
    rho = [local_density(dist, i, LN) for i in range(N)]
    # define Local cores and representatives with LORE
    local_cores, rep = LORE(LN, rho, X, dist)
    # rep count necessary to define which node is representing how many nodes
    rep_count = [
        x
        for xs in rep
        for x in xs
    ]
    ########## directed saturated neighbor graph Definition 4
    # if j is one of the nearest neighbors of i then edge i->j
    conn = np.ones((N, N))
    ## set edge weight to infinity
    conn = conn * np.inf
    ## for every point in the dataset
    for i in range(N):
        # if j is one of the nearest neighbors we set the weight to euclidean distance between i and j
        for j in LN[i]:
            conn[i, j] = dist[i, j]

    # graph-based distance  (Equation 4)
    graph = csr_matrix(conn)
    # shortest paths define dist_matrix
    dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=True, return_predecessors=True)
    uniques = np.unique(dist_matrix)
    if np.isinf(uniques[-1]):
        max = np.unique(dist_matrix)[-2]
        dist_matrix[dist_matrix > max] = max

    # local cores dict mapping clusterlabel to corresponding local cores
    local_cores_in = {key: [] for key in np.unique(labels)}
    # for each local core
    for i in local_cores:
        # append list including local cores at cluster label
        local_cores_in[labels[i]].append(i)

    # lccv sum is needed to collect local-core wise results
    lccv_sum = 0
    # print(local_cores_in)
    # for each local core
    for i in local_cores:
        # get cluster for i (A)
        label = labels[i]
        # local cores that are noise are only included in b_i
        if label != -1:
            # get all local cores belonging to the same cluster
            local_core_in_A = local_cores_in[label]
            # get number of local cores in cluster A
            n_l_A = len(local_core_in_A)
            # if there is only one local core in the cluster
            if n_l_A < 2:
                lccv_sum += 0
            else:
                # distances between i and every local core also belonging to the same cluster
                dists = [dist_matrix[i, j] for j in local_core_in_A]
                # a is defined in section D LCCV Index
                a_i = (1 / (n_l_A - 1)) * np.sum(dists)
                # distances between i and local points form other clusters cluster wise
                cluster_wise_dists = []
                # for each label that is not A
                for l in np.unique(labels):
                    if l != label:
                        if len(local_cores_in[l]) < 1:
                            continue
                        # include local core averaged distance to i
                        dists = [dist_matrix[i, j] for j in local_cores_in[l]]
                        cluster_wise_dists.append((1 / len(local_cores_in[l])) * sum(dists))
                # get minimum local core averaged distance to i over all other clusters
                b_i = min(cluster_wise_dists)
                # set into silhouette coefficient like equation
                lccv = sil_eq(a_i, b_i)
                # number of points being represented by i
                n_i = rep_count.count(i)
                # add score for this local core
                lccv_sum = lccv_sum + (lccv * n_i)
    # reduce local core wise scores to one score for the clustering
    lccv_c = (1 / N) * lccv_sum
    return lccv_c


def sil_eq(a, b):
    return (b - a) / np.max([a, b])

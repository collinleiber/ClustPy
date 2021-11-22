from sklearn.datasets import make_blobs
from scipy.stats import special_ortho_group
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import check_random_state


def create_subspace_data(n_samples=1000, n_clusters=3, subspace_features=[2, 2], n_outliers=[0, 0], std=1,
                         box=(-10, 10), random_state=None):
    assert type(n_clusters) is int, "n_clusters must be of type int"
    X, L = create_nr_data(n_samples=n_samples, n_clusters=[n_clusters, 1], subspace_features=subspace_features,
                          n_outliers=n_outliers, std=std, box=box, random_state=random_state)
    return X, L[:,0]


def create_nr_data(n_samples=1000, n_clusters=[3, 3, 1], subspace_features=[2, 2, 2], n_outliers=[0, 0, 0], std=1,
                   box=(-10, 10), random_state=None):
    random_state = check_random_state(random_state)
    # Transform n_clusters to list
    if type(n_clusters) is not list:
        n_clusters = [n_clusters]
    # Transform cluster_features to list
    if type(subspace_features) is not list:
        subspace_features = [subspace_features] * len(n_clusters)
    assert len(n_clusters) == len(
        subspace_features), "inconsistent number of subspaces between n_clusters and subspace_features"
    # Transform n_outliers to list
    if type(n_outliers) is not list:
        n_outliers = [n_outliers] * len(n_clusters)
    assert len(n_clusters) == len(n_outliers), "inconsistent number of subspaces between n_clusters and n_outliers"
    # Transform std to list
    if type(std) is not list:
        std = [std] * len(n_clusters)
    assert len(n_clusters) == len(std), "inconsistent number of subspaces between n_clusters and std"
    # Transform box to list
    if type(box) is not list:
        box = [box] * len(n_clusters)
    assert len(n_clusters) == len(box), "inconsistent number of subspaces between n_clusters and box"
    # Create empty dataset
    X, L = np.empty((n_samples, 0)), np.empty((n_samples, 0))
    for i in range(len(n_clusters)):
        # Create Clusters
        X_tmp, L_tmp = make_blobs(n_samples - n_outliers[i], subspace_features[i], centers=n_clusters[i],
                                  cluster_std=std[i], center_box=box[i], random_state=random_state)
        # Create outliers
        if n_outliers[i] != 0:
            X_out = random_state.random((n_outliers[i], subspace_features[i]))
            out_box = (box[i][0] - 4 * std[i], box[i][1] + 4 * std[i])
            X_out = X_out * (out_box[1] - out_box[0]) + out_box[0]
            L_out = np.array([-1] * n_outliers[i])
            X_tmp = np.r_[X_tmp, X_out]
            L_tmp = np.r_[L_tmp, L_out]
            # Shuffle data so outliers wont be always in the last positions
            X_tmp, L_tmp = shuffle(X_tmp, L_tmp, random_state=random_state)
        # Add subspace to dataset
        X = np.c_[X, X_tmp]
        L = np.c_[L, L_tmp]
    # Rotate space
    V = special_ortho_group.rvs(dim=sum(subspace_features))
    X_transformed = np.matmul(X, V)
    return X_transformed, L

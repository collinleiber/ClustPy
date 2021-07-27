from sklearn.datasets import make_blobs
from scipy.stats import special_ortho_group
import numpy as np

def create_subspace_data(n_samples=1000, n_clusters=3, cluster_features=2, total_features = 5):
    assert total_features >= cluster_features, "total_features can not be smaller than cluster_features"
    V = special_ortho_group.rvs(dim=total_features)
    X1, L = make_blobs(n_samples, cluster_features, n_clusters)
    X2, _ = make_blobs(n_samples, (total_features - cluster_features), 1)
    X = np.c_[X1, X2]
    X_transformed = np.matmul(X, V)
    return X_transformed, L

def create_nr_data(n_samples=1000, n_clusters=[3,3], cluster_features=[2,2], total_features=5, std=1, box=(-10, 10)):
    assert total_features >= sum(cluster_features), "total_features can not be smaller than sum of cluster_features"
    assert len(n_clusters) == len(cluster_features), "inconsistent number of subspaces"
    V = special_ortho_group.rvs(dim=total_features)
    X1, L = np.empty((n_samples, 0)), np.empty((n_samples, 0))
    for i in range(len(n_clusters)):
        X_tmp, L_tmp = make_blobs(n_samples, cluster_features[i], centers=n_clusters[i], cluster_std=std, center_box=box)
        X1 = np.c_[X1, X_tmp]
        L = np.c_[L, L_tmp]
    X2, _ = make_blobs(n_samples, (total_features - sum(cluster_features)), centers=1)
    X = np.c_[X1, X2]
    X_transformed = np.matmul(X, V)
    return X_transformed, L

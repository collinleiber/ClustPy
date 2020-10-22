from sklearn.datasets import make_blobs
from scipy.stats import ortho_group
import numpy as np

def create_subspace_data(n_samples=1000, n_clusters=3, cluster_features=2, total_features = 5):
    V = ortho_group.rvs(dim=total_features)
    X1, L = make_blobs(n_samples, cluster_features, n_clusters)
    X2, _ = make_blobs(n_samples, (total_features - cluster_features), 1)
    X = np.c_[X1, X2]
    X_transformed = np.matmul(X, V)
    return X_transformed, L

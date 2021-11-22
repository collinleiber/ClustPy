import numpy as np
try:
    # Sklearn version >= 0.24.X
    from sklearn.cluster import kmeans_plusplus as kpp
except:
    try:
        # Old sklearn versions
        from sklearn.cluster._kmeans import _kmeans_plusplus as kpp
    except:
        # Very old sklearn versions
        from sklearn.cluster._kmeans import _k_init as kpp

def pyclustering_adjust_labels(n_points, pyclus_labels):
    labels = np.zeros(n_points)
    for i, l in enumerate(pyclus_labels):
        # Change label of all points within list l
        labels[l] = i
    return labels

def _get_n_clusters_from_algo(algo_obj):
    if hasattr(algo_obj, "n_clusters"):
        n_clusters = algo_obj.n_clusters
    elif hasattr(algo_obj, "n_clusters_"):
        n_clusters = algo_obj.n_clusters_
    else:
        n_clusters = np.unique(algo_obj.labels_).shape[0]
    return n_clusters

def _kmeans_plus_plus(X, n_clusters, x_squared_norms, random_state=None, n_local_trials=None):
    result = kpp(X, n_clusters, x_squared_norms, random_state, n_local_trials)
    if type(result) is tuple:
        result = result[0]
    return result
import numpy as np


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
        n_clusters = np.unique(algo_obj.labels).shape[0]
    return n_clusters

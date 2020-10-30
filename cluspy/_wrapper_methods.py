import numpy as np


def pyclustering_adjust_labels(n_points, pyclus_labels):
    labels = np.zeros(n_points)
    for i, l in enumerate(pyclus_labels):
        # Change label of all points within list l
        labels[l] = i
    return labels

def sklearn_get_n_clusters(labels):
    return np.unique(labels).shape[0]
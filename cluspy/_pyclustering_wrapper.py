import numpy as np


def adjust_labels(n_points, pyclus_labels):
    labels = np.zeros(n_points)
    for i, l in enumerate(pyclus_labels):
        # Change label of all points within list l
        labels[l] = i
    return labels


def adjust_lists(pyclus_centers):
    return np.array(pyclus_centers)

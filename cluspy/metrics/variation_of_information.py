"""
Meilă, Marina. "Comparing clusterings by the variation of
information." Learning theory and kernel machines. Springer,
Berlin, Heidelberg, 2003. 173-187.
"""

import numpy as np


def _check_number_of_points(ground_truth, prediction):
    if prediction.shape[0] != ground_truth.shape[0]:
        raise Exception(
            "Number of objects of the prediction and ground truth are not equal.\nNumber of prediction objects: " + str(
                len(prediction.shape[0])) + "\nNumber of ground truth objects: " + str(ground_truth.shape[0]))


# Source https://en.wikipedia.org/wiki/Variation_of_information
def variation_of_information(ground_truth, prediction):
    _check_number_of_points(ground_truth, prediction)
    n = len(ground_truth)
    cluster_ids_gt = np.unique(ground_truth)
    cluster_ids_pred = np.unique(prediction)
    result = 0.0
    for id_gt in cluster_ids_gt:
        points_in_cluster_gt = np.argwhere(ground_truth == id_gt)[:, 0]
        p = len(points_in_cluster_gt) / n
        for id_pred in cluster_ids_pred:
            points_in_cluster_pred = np.argwhere(prediction == id_pred)[:, 0]
            q = len(points_in_cluster_pred) / n
            r = len([point for point in points_in_cluster_gt if point in points_in_cluster_pred]) / n
            if r != 0:
                result += r * (np.log(r / p) + np.log(r / q))
    return -1 * result

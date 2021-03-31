import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def _check_number_of_points(labels_true, labels_pred):
    if labels_pred.shape[0] != labels_true.shape[0]:
        raise Exception(
            "Number of objects of the prediction and ground truth are not equal.\nNumber of prediction objects: " + str(
                labels_pred.shape[0]) + "\nNumber of ground truth objects: " + str(labels_true.shape[0]))

"""
Meilă, Marina. "Comparing clusterings by the variation of
information." Learning theory and kernel machines. Springer,
Berlin, Heidelberg, 2003. 173-187.
"""
# Source https://en.wikipedia.org/wiki/Variation_of_information
def variation_of_information(labels_true, labels_pred):
    """
    Variation of Information
    :param labels_true:
    :param labels_pred:
    :return:
    """

    _check_number_of_points(labels_true, labels_pred)
    n = len(labels_true)
    cluster_ids_true = np.unique(labels_true)
    cluster_ids_pred = np.unique(labels_pred)
    result = 0.0
    for id_true in cluster_ids_true:
        points_in_cluster_gt = np.argwhere(labels_true == id_true)[:, 0]
        p = len(points_in_cluster_gt) / n
        for id_pred in cluster_ids_pred:
            points_in_cluster_pred = np.argwhere(labels_pred == id_pred)[:, 0]
            q = len(points_in_cluster_pred) / n
            r = len([point for point in points_in_cluster_gt if point in points_in_cluster_pred]) / n
            if r != 0:
                result += r * (np.log(r / p) + np.log(r / q))
    return -1 * result

"""
Yang, Yi, et al. "Image clustering using local discriminant
models and global integration." IEEE Transactions on Image
Processing 19.10 (2010): 2761-2773.
"""
# Source https://github.com/XifengGuo/DEC-keras/blob/master/metrics.py
def unsupervised_clustering_accuracy(labels_true, labels_pred):
    """
    Unsupervised Clustering Accuracy
    # Return
        accuracy, in [0,1]
    """
    _check_number_of_points(labels_true, labels_pred)
    D = max(labels_pred.max(), labels_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(labels_pred.size):
        w[labels_pred[i], labels_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size
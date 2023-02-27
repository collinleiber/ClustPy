import numpy as np
from scipy.optimize import linear_sum_assignment


def _check_number_of_points(labels_true: np.ndarray, labels_pred: np.ndarray) -> bool:
    """
    Check if the length of the ground truth labels and the prediction labels match.
    If they do not match throw an exception.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    boolean : bool
        True if execution was successful
    """
    if labels_pred.shape[0] != labels_true.shape[0]:
        raise Exception(
            "Number of objects of the prediction and ground truth are not equal.\nNumber of prediction objects: " + str(
                labels_pred.shape[0]) + "\nNumber of ground truth objects: " + str(labels_true.shape[0]))
    return True


def variation_of_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the variation of information between the ground truth labels and the predicted labels.
    Implemented as defined in https://en.wikipedia.org/wiki/Variation_of_information

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    vi : float
        The variation of information

    References
    -------
    Meilă, Marina. "Comparing clusterings by the variation of information."
    Learning theory and kernel machines. Springer, Berlin, Heidelberg, 2003. 173-187.
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
    vi = -1 * result
    return vi


def unsupervised_clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Evaluate the quality of predicted labels by comparing it to the ground truth labels using the
    clustering accuracy.
    Returns a value between 1.0 (perfect match) and 0.0 (arbitrary result).
    Since the id of a cluster is not fixed in a clustering setting, the clustering accuracy evaluates each possible
    combination with the ground truth labels.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    acc : float
        The accuracy between the two input label sets.

    References
    -------
    Yang, Yi, et al. "Image clustering using local discriminant models and global integration."
    IEEE Transactions on Image Processing 19.10 (2010): 2761-2773.
    """
    _check_number_of_points(labels_true, labels_pred)
    max_label = int(max(labels_pred.max(), labels_true.max()) + 1)
    match_matrix = np.zeros((max_label, max_label), dtype=np.int64)
    for i in range(labels_true.shape[0]):
        match_matrix[int(labels_true[i]), int(labels_pred[i])] -= 1
    indices = linear_sum_assignment(match_matrix)
    acc = -np.sum(match_matrix[indices]) / labels_pred.size
    return acc

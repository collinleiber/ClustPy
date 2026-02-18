import numpy as np
from sklearn.metrics.cluster._supervised import check_clusterings
from sklearn.utils import check_X_y


def _check_labels_arrays(labels_true: np.ndarray, labels_pred: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Check that the ground truth labels and the prediction labels are compatible.
    If they do not match throw an exception.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The ground truth labels,
        The predicted labels
    """
    labels_true = labels_true.astype(int)
    labels_pred = labels_pred.astype(int)
    if labels_true.ndim == 1 and labels_pred.ndim == 1:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    else:
        if labels_true.ndim > 1:
            if labels_pred.ndim > 1:
                labels_pred_ref = labels_pred[:, 0]
            else:
                labels_pred_ref = labels_pred
            labels_true = labels_true.copy()
            for i in range(labels_true.shape[1]):
                labels_true_i, _ = check_clusterings(labels_true[:, i], labels_pred_ref)
                labels_true[:, i] = labels_true_i
        if labels_pred.ndim > 1:
            if labels_true.ndim > 1:
                labels_true_ref = labels_true[:, 0]
            else:
                labels_true_ref = labels_true
            labels_pred = labels_pred.copy()
            for i in range(labels_pred.shape[1]):
                labels_pred_i, _ = check_clusterings(labels_pred[:, i], labels_true_ref)
                labels_pred[:, i] = labels_pred_i
    return labels_true, labels_pred


def _check_length_data_and_labels(X: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Check that the data and the prediction labels are compatible.
    If they do not match throw an exception.

    Parameters
    ----------
    X : np.ndarray
        The data set
    labels : np.ndarray
        The labels as predicted by a clustering algorithm

     Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The data set,
        The predicted labels
    """
    X, labels = check_X_y(X, labels)
    labels = labels.astype(int)
    n_pred_clusters = len(np.unique(labels))
    if X.shape[0] != X.shape[0]:
        raise ValueError(
            "Number of data objects and predicted labels are not equal.\nNumber of data objects: " + str(
                X.shape[0]) + "\nNumber of predicted labels: " + str(labels.shape[0]))
    if n_pred_clusters == 1 or n_pred_clusters == X.shape[0]:
        raise ValueError("The number of different labels must be within [2, n_samples -1]")
    return X, labels
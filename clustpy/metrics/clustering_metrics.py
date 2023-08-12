import numpy as np
from scipy.optimize import linear_sum_assignment
from clustpy.metrics.confusion_matrix import ConfusionMatrix
from scipy.special import comb
from sklearn.metrics import normalized_mutual_info_score as nmi


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
    Returns a minimum value of 0.0 which corresponds to a perfect match.
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


def information_theoretic_external_cluster_validity_measure(labels_true: np.ndarray, labels_pred: np.ndarray,
                                                            scale: bool = True) -> float:
    """
    Evaluate the quality of predicted labels by comparing it to the ground truth labels using the
    Information-Theoretic External Cluster-Validity Measure. Often simply called DOM.
    A lower value indicates a better clustering result.
    If the result is scaled, this method will return a value between 1.0 (perfect match) and 0.0 (arbitrary result).
    An advantage of this metric is that it also works with a differing number of clusters.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm
    scale : bool
        Scale the result to (0, 1], where 1 indicates a perfect match and 0 indicates an arbitrary result (default: True)

    Returns
    -------
    dom : float
        The validity between the two input label sets.

    References
    -------
    Byron E. Dom. 2002. "An information-theoretic external cluster-validity measure."
    In Proceedings of the Eighteenth conference on Uncertainty in artificial intelligence (UAI'02).
    """
    _check_number_of_points(labels_true, labels_pred)
    # Build confusion matrix
    cm = ConfusionMatrix(labels_true, labels_pred)
    n_points = labels_true.shape[0]
    n_classes = cm.confusion_matrix.shape[0]
    # Get number of objects per predicted label
    hks = np.sum(cm.confusion_matrix, axis=0)
    # Calculate Q_0
    cm_tmp = cm.confusion_matrix.copy()  # Needed if some cells are 0 so log can be calculated
    cm_tmp[cm_tmp == 0] = 1  # will later be multiplied by 0, so this does not change the final result
    empirical_conditional_entropy = cm.confusion_matrix / n_points * np.log(cm_tmp / hks)
    empirical_conditional_entropy = - np.sum(
        empirical_conditional_entropy)  # [~np.isnan(empirical_conditional_entropy)])
    sum_binom_coefficient = np.sum([np.log(comb(hk + n_classes - 1, n_classes - 1)) for hk in hks])
    Q_0 = empirical_conditional_entropy + sum_binom_coefficient / n_points
    if scale:
        # --- Scale Q_0 to (0, 1] ---
        # Get number of objects per ground truth label
        hcs = np.sum(cm.confusion_matrix, axis=1)
        # Calculate Q_2
        min_Q_0 = np.sum([np.log(comb(hc + n_classes - 1, n_classes - 1)) for hc in hcs]) / n_points
        entropy_H_C = -np.sum([hc / n_points * np.log(hc / n_points) for hc in hcs])
        max_Q_0 = entropy_H_C + np.log(n_classes)
        assert Q_0 >= min_Q_0 and Q_0 <= max_Q_0, "Q_0 must be between min_Q_0 and max_Q_0"
        # Scale Q_0 to receive final value
        Q_0 = (max_Q_0 - Q_0) / (max_Q_0 - min_Q_0)
    return Q_0


def fair_normalized_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray):
    """
    Evaluate the quality of predicted labels by comparing it to the ground truth labels using the
    fair normalized mutual information score. Often simply called FNMI.
    A value of 1 indicates a perfect clustering result, a value of 0 indicates totally random result.
    The FNMI punishes if the number of predicted clusters diverges from the ground truth number of clusters.
    Therefore, it uses the normalized mutual information from sklearn and scales the value by using the predicted and ground truth number of clusters.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    fnmi : float
        The score between the two input label sets.

    References
    -------
    Amelio, Alessia, and Clara Pizzuti. "Is normalized mutual information a fair measure for comparing community detection methods?."
    Proceedings of the 2015 IEEE/ACM international conference on advances in social networks analysis and mining 2015. 2015.
    """
    # Get the normalized mutual information
    my_nmi = nmi(labels_true, labels_pred)
    # Get number of clusters
    n_clusters_true = len(np.unique(labels_true))
    n_clusters_pred = len(np.unique(labels_pred))
    # Calculate FNMI
    factor = np.exp(-abs(n_clusters_true - n_clusters_pred) / n_clusters_true)
    fnmi = factor * my_nmi
    return fnmi

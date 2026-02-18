import numpy as np
from scipy.optimize import linear_sum_assignment
from clustpy.metrics.confusion_matrix import ConfusionMatrix
from scipy.special import comb
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.metrics._metrics_utils import _check_labels_arrays


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
    confusion_matrix = ConfusionMatrix(labels_true, labels_pred).confusion_matrix
    n = len(labels_true)
    p = confusion_matrix.sum(1).reshape((-1, 1)) / n
    q = confusion_matrix.sum(0).reshape((1, -1)) / n
    r = confusion_matrix / n
    # Consider zero entries
    mask = (r == 0)
    r[mask] = 1
    result = r * (np.log(r / p) + np.log(r / q))
    result[mask] = 0
    vi = -result.sum()
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
    confusion_matrix = ConfusionMatrix(labels_true, labels_pred, "square").confusion_matrix
    indices = linear_sum_assignment(-confusion_matrix)
    acc = np.sum(confusion_matrix[indices]) / len(labels_true)
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
    # Build confusion matrix
    confusion_matrix = ConfusionMatrix(labels_true, labels_pred).confusion_matrix
    n_points = labels_true.shape[0]
    n_classes = confusion_matrix.shape[0]
    # Get number of objects per predicted label
    hks = np.sum(confusion_matrix, axis=0)
    # Calculate Q_0
    cm_tmp = confusion_matrix.copy()  # Needed if some cells are 0 so log can be calculated
    cm_tmp[cm_tmp == 0] = 1  # will later be multiplied by 0, so this does not change the final result
    empirical_conditional_entropy = confusion_matrix / n_points * np.log(cm_tmp / hks)
    empirical_conditional_entropy = - np.sum(
        empirical_conditional_entropy)  # [~np.isnan(empirical_conditional_entropy)])
    sum_binom_coefficient = np.sum([np.log(comb(hk + n_classes - 1, n_classes - 1)) for hk in hks])
    Q_0 = empirical_conditional_entropy + sum_binom_coefficient / n_points
    if scale:
        # --- Scale Q_0 to (0, 1] ---
        # Get number of objects per ground truth label
        hcs = np.sum(confusion_matrix, axis=1)
        # Calculate Q_2
        min_Q_0 = np.sum([np.log(comb(hc + n_classes - 1, n_classes - 1)) for hc in hcs]) / n_points
        entropy_H_C = -np.sum([hc / n_points * np.log(hc / n_points) for hc in hcs])
        max_Q_0 = entropy_H_C + np.log(n_classes)
        assert Q_0 >= min_Q_0 and Q_0 <= max_Q_0, "Q_0 must be between min_Q_0 and max_Q_0"
        # Scale Q_0 to receive final value
        Q_0 = (max_Q_0 - Q_0) / (max_Q_0 - min_Q_0)
    return Q_0


def fair_normalized_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Evaluate the quality of predicted labels by comparing to the ground truth labels using the
    fair normalized mutual information score. Often simply called FNMI.
    A value of 1 indicates a perfect clustering result, a value of 0 indicates a totally random result.
    The FNMI punishes results where the number of predicted clusters diverges from the ground truth number of clusters.
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
    labels_true, labels_pred = _check_labels_arrays(labels_true, labels_pred)
    # Get the normalized mutual information
    my_nmi = nmi(labels_true, labels_pred)
    # Get number of clusters
    n_clusters_true = len(np.unique(labels_true))
    n_clusters_pred = len(np.unique(labels_pred))
    # Calculate FNMI
    factor = np.exp(-abs(n_clusters_true - n_clusters_pred) / n_clusters_true)
    fnmi = factor * my_nmi
    return fnmi


def purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Evaluate the quality of predicted labels by comparing it to the ground truth labels using the
    clustering purity.
    Returns a value between 1.0 (perfect match) and 0.0 (arbitrary result).
    Note that the purity is usually very high when the number of predicted clusters is much larger than the ground truth number of clusters.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    purity : float
        The purity between the two input label sets.

    References
    -------
    Manning, Christopher D. An introduction to information retrieval. 2009.
    """
    conf_matrix = ConfusionMatrix(labels_true, labels_pred).confusion_matrix
    best_matches = np.max(conf_matrix, axis=0)
    purity = np.sum(best_matches) / labels_true.shape[0]
    return purity

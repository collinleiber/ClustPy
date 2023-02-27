from clustpy.metrics.clustering_metrics import _check_number_of_points
import numpy as np

"""
Internal functions
"""


def _jaccard_score(n_tp: int, n_fp: int, n_fn: int) -> float:
    """
    Calculate the jaccard score.
    Jaccard score = n_tp / (n_tp + n_fp + n_fn).

    Parameters
    ----------
    n_tp : int
        The number of true positives
    n_fp : int
        The number of false positives
    n_fn : int
        The number of false negatives

    Returns
    -------
    score : float
        The jaccard score

    References
    ----------
    Jaccard, Paul. "Lois de distribution florale dans la zone alpine."
    Bull Soc Vaudoise Sci Nat 38 (1902): 69-130.
    """
    score = 0 if (n_tp + n_fp + n_fn) == 0 else n_tp / (n_tp + n_fp + n_fn)
    return score


def _rand_score(n_tp: int, n_fp: int, n_fn: int, n_tn: int) -> float:
    """
    Calculate the rand score.
    Rand score = (n_tp + n_tn) / (n_tp + n_fp + n_fn + n_tn).

    Parameters
    ----------
    n_tp : int
        The number of true positives
    n_fp : int
        The number of false positives
    n_fn : int
        The number of false negatives
    n_tn : int
        The number of true negatives

    Returns
    -------
    score : float
        The rand score

    References
    ----------
    Rand, William M. "Objective criteria for the evaluation of clustering methods."
    Journal of the American Statistical association 66.336 (1971): 846-850.
    """
    score = 0 if (n_tp + n_fp + n_fn + n_tn) == 0 else (n_tp + n_tn) / (n_tp + n_fp + n_fn + n_tn)
    return score


def _precision_score(n_tp: int, n_fp: int) -> float:
    """
    Calculate the precision.
    Precision score = n_tp / (n_tp + n_fp).

    Parameters
    ----------
    n_tp : int
        The number of true positives
    n_fp : int
        The number of false positives

    Returns
    -------
    score : float
        The precision score

    References
    ----------
    Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
    American Documentation (pre-1986) 6.2 (1955): 93.
    """
    score = 0 if (n_tp + n_fp) == 0 else n_tp / (n_tp + n_fp)
    return score


def _recall_score(n_tp: int, n_fn: int) -> float:
    """
    Calculate the recall.
    Recall score = n_tp / (n_tp + n_fn).

    Parameters
    ----------
    n_tp : int
        The number of true positives
    n_fn : int
        The number of false negatives

    Returns
    -------
    score : float
        The recall score

    References
    ----------
    Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
    American Documentation (pre-1986) 6.2 (1955): 93.
    """
    score = 0 if (n_tp + n_fn) == 0 else n_tp / (n_tp + n_fn)
    return score


def _f1_score(n_tp: int, n_fp: int, n_fn: int) -> float:
    """
    Calculate the f1 score.
    F1 score = 2 * precision * recall / (precision + recall).

    Parameters
    ----------
    n_tp : int
        The number of true positives
    n_fp : int
        The number of false positives
    n_fn : int
        The number of false negatives

    Returns
    -------
    score : float
        The f1 score

    References
    ----------
    Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth-Heinemann.
    """
    precision = _precision_score(n_tp, n_fp)
    recall = _recall_score(n_tp, n_fn)
    score = 0 if (precision == 0 and recall == 0) else 2 * precision * recall / (precision + recall)
    return score


"""
External functions
"""


def pc_jaccard_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the jaccard score.
    Jaccard score = n_tp / (n_tp + n_fp + n_fn).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    score : float
        The jaccard score

    References
    ----------
    Jaccard, Paul. "Lois de distribution florale dans la zone alpine."
    Bull Soc Vaudoise Sci Nat 38 (1902): 69-130.

    and

    Pfitzner, Darius, Richard Leibbrandt, and David Powers. "Characterization and evaluation of similarity measures for pairs of clusterings."
    Knowledge and Information Systems 19 (2009): 361-394.
    """
    n_tp, n_fp, n_fn, _ = _get_pair_counting_categories(labels_true, labels_pred)
    score = _jaccard_score(n_tp, n_fp, n_fn)
    return score


def pc_rand_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the rand score.
    Rand score = (n_tp + n_tn) / (n_tp + n_fp + n_fn + n_tn).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    score : float
        The rand score

    References
    ----------
    Rand, William M. "Objective criteria for the evaluation of clustering methods."
    Journal of the American Statistical association 66.336 (1971): 846-850.

    and

    Pfitzner, Darius, Richard Leibbrandt, and David Powers. "Characterization and evaluation of similarity measures for pairs of clusterings."
    Knowledge and Information Systems 19 (2009): 361-394.
    """
    n_tp, n_fp, n_fn, n_tn = _get_pair_counting_categories(labels_true, labels_pred)
    score = _rand_score(n_tp, n_fp, n_fn, n_tn)
    return score


def pc_precision_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the precision.
    Precision score = n_tp / (n_tp + n_fp).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    score : float
        The precision score

    References
    ----------
    Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
    American Documentation (pre-1986) 6.2 (1955): 93.

    and

    Pfitzner, Darius, Richard Leibbrandt, and David Powers. "Characterization and evaluation of similarity measures for pairs of clusterings."
    Knowledge and Information Systems 19 (2009): 361-394.
    """
    n_tp, n_fp, _, _ = _get_pair_counting_categories(labels_true, labels_pred)
    score = _precision_score(n_tp, n_fp)
    return score


def pc_recall_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the recall.
    Recall score = n_tp / (n_tp + n_fn).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    score : float
        The recall score

    References
    ----------
    Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
    American Documentation (pre-1986) 6.2 (1955): 93.

    and

    Pfitzner, Darius, Richard Leibbrandt, and David Powers. "Characterization and evaluation of similarity measures for pairs of clusterings."
    Knowledge and Information Systems 19 (2009): 361-394.
    """
    n_tp, _, n_fn, _ = _get_pair_counting_categories(labels_true, labels_pred)
    score = _recall_score(n_tp, n_fn)
    return score


def pc_f1_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the f1 score.
    F1 score = 2 * precision * recall / (precision + recall).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    score : float
        The f1 score

    References
    ----------
    Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth-Heinemann.

    and

    Pfitzner, Darius, Richard Leibbrandt, and David Powers. "Characterization and evaluation of similarity measures for pairs of clusterings."
    Knowledge and Information Systems 19 (2009): 361-394.
    """
    n_tp, n_fp, n_fn, _ = _get_pair_counting_categories(labels_true, labels_pred)
    score = _f1_score(n_tp, n_fp, n_fn)
    return score


def _get_pair_counting_categories(labels_true: np.ndarray, labels_pred: np.ndarray) -> (int, int, int, int):
    """
    Get the number of 'true positives', 'false positives', 'false negatives' and 'true negatives' to calculate pair-counting scores.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    tuple : (int, int, int, int)
        The number of true positives,
        The number of false positives,
        The number of false negatives,
        The number of true negatives
    """
    _check_number_of_points(labels_true, labels_pred)
    if labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise Exception("labels_true and labels_pred labels should just contain a single column.")
    n_tp = 0
    n_fp = 0
    n_fn = 0
    n_tn = 0
    for i in range(labels_pred.shape[0] - 1):
        n_tp += np.sum((labels_pred[i] == labels_pred[i + 1:]) & (labels_true[i] == labels_true[i + 1:]))
        n_fp += np.sum((labels_pred[i] == labels_pred[i + 1:]) & (labels_true[i] != labels_true[i + 1:]))
        n_fn += np.sum((labels_pred[i] != labels_pred[i + 1:]) & (labels_true[i] == labels_true[i + 1:]))
        n_tn += np.sum((labels_pred[i] != labels_pred[i + 1:]) & (labels_true[i] != labels_true[i + 1:]))
    return n_tp, n_fp, n_fn, n_tn


"""
Pair Counting Object
"""


class PairCountingScores():
    """
    Obtain all parameters that are necessary to calculate the pair-counting scores 'jaccard', 'rand', 'precision', 'recall' and 'f1'.
    These parameters are the number of 'true positives', 'false positives', 'false negatives' and 'true negatives'.
    The resulting object can call all pair-counting score methods.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Attributes
    ----------
    n_tp : int
        The number of true positives,
    n_fp : int
        The number of false positives,
    n_fn : int
        The number of false negatives,
    n_tn : int
        The number of true negatives

    References
    ----------
    Pfitzner, Darius, Richard Leibbrandt, and David Powers. "Characterization and evaluation of similarity measures for pairs of clusterings."
    Knowledge and Information Systems 19 (2009): 361-394.
    """

    def __init__(self, labels_true: np.ndarray, labels_pred: np.ndarray):
        _check_number_of_points(labels_true, labels_pred)
        n_tp, n_fp, n_fn, n_tn = _get_pair_counting_categories(labels_true, labels_pred)
        self.n_tp = n_tp
        self.n_fp = n_fp
        self.n_fn = n_fn
        self.n_tn = n_tn

    def jaccard(self) -> float:
        """
        Calculate the jaccard score.
        Jaccard score = n_tp / (n_tp + n_fp + n_fn).

        Returns
        -------
        score : float
            The jaccard score

        References
        ----------
        Jaccard, Paul. "Lois de distribution florale dans la zone alpine."
        Bull Soc Vaudoise Sci Nat 38 (1902): 69-130.
        """
        score = _jaccard_score(self.n_tp, self.n_fp, self.n_fn)
        return score

    def rand(self) -> float:
        """
        Calculate the rand score.
        Rand score = (n_tp + n_tn) / (n_tp + n_fp + n_fn + n_tn).

        Returns
        -------
        score : float
            The rand score

        References
        ----------
        Rand, William M. "Objective criteria for the evaluation of clustering methods."
        Journal of the American Statistical association 66.336 (1971): 846-850.
        """
        score = _rand_score(self.n_tp, self.n_fp, self.n_fn, self.n_tn)
        return score

    def precision(self) -> float:
        """
        Calculate the precision.
        Precision score = n_tp / (n_tp + n_fp).

        Returns
        -------
        score : float
            The precision score

        References
        ----------
        Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
        American Documentation (pre-1986) 6.2 (1955): 93.
        """
        score = _precision_score(self.n_tp, self.n_fp)
        return score

    def recall(self) -> float:
        """
        Calculate the recall.
        Recall score = n_tp / (n_tp + n_fn).

        Returns
        -------
        score : float
            The recall score

        References
        ----------
        Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
        American Documentation (pre-1986) 6.2 (1955): 93.
        """
        score = _recall_score(self.n_tp, self.n_fn)
        return score

    def f1(self) -> float:
        """
        Calculate the f1 score.
        F1 score = 2 * precision * recall / (precision + recall).

        Returns
        -------
        score : float
            The f1 score

        References
        ----------
        Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth-Heinemann.
        """
        score = _f1_score(self.n_tp, self.n_fp, self.n_fn)
        return score

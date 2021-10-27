import numpy as np
from itertools import permutations
from cluspy.metrics.clustering_metrics import _check_number_of_points, variation_of_information as vi
from cluspy.metrics.pair_counting_scores import PairCountingScores, _pc_f1_score, _pc_recall_score, _pc_precision_score, \
    _pc_rand_score, _pc_jaccard_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from cluspy.metrics.confusion_matrix import ConfusionMatrix, _plot_confusion_matrix

_CONFUSION_AGGREGATIONS = ["max", "min", "permut-max", "permut-min", "avg"]

"""
HELPERS
"""


def remove_noise_spaces_from_labels(labels):
    no_noise_spaces = [True] * labels.shape[1]
    for c in range(labels.shape[1]):
        unique_labels = np.unique(labels[:, c])
        # Consider outliers
        len_unique_labels = len(unique_labels) if np.all(unique_labels >= 0) else len(unique_labels) - 1
        if len_unique_labels == 1:
            no_noise_spaces[c] = False
    labels_new = labels[:, no_noise_spaces]
    return labels_new


"""
Multiple Labelings Pair Counting Scores
"""


def multiple_labelings_pc_jaccard_score(labels_true, labels_pred, remove_noise_spaces=True):
    n_tp, n_fp, n_fn, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                           remove_noise_spaces)
    return _pc_jaccard_score(n_tp, n_fp, n_fn)


def multiple_labelings_pc_rand_score(labels_true, labels_pred, remove_noise_spaces=True):
    n_tp, n_fp, n_fn, n_tn = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                              remove_noise_spaces)
    return _pc_rand_score(n_tp, n_fp, n_fn, n_tn)


def multiple_labelings_pc_precision_score(labels_true, labels_pred, remove_noise_spaces=True):
    n_tp, n_fp, _, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred, remove_noise_spaces)
    return _pc_precision_score(n_tp, n_fp)


def multiple_labelings_pc_recall_score(labels_true, labels_pred, remove_noise_spaces=True):
    n_tp, _, n_fn, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred, remove_noise_spaces)
    return _pc_recall_score(n_tp, n_fn)


def multiple_labelings_pc_f1_score(labels_true, labels_pred, remove_noise_spaces=True):
    n_tp, n_fp, n_fn, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                           remove_noise_spaces)
    return _pc_f1_score(n_tp, n_fp, n_fn)


def _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred, remove_noise_spaces):
    _check_number_of_points(labels_true, labels_pred)
    if labels_true.ndim == 1:
        labels_true = labels_true.reshape((-1, 1))
    if labels_pred.ndim == 1:
        labels_pred = labels_pred.reshape((-1, 1))
    if remove_noise_spaces:
        labels_true = remove_noise_spaces_from_labels(labels_true)
        labels_pred = remove_noise_spaces_from_labels(labels_pred)
    n_tp = 0
    n_fp = 0
    n_fn = 0
    n_tn = 0
    for i in range(labels_pred.shape[0] - 1):
        for j in range(i + 1, labels_pred.shape[0]):
            if _anywhere_same_cluster(labels_pred, i, j):
                if _anywhere_same_cluster(labels_true, i, j):
                    n_tp += 1
                else:
                    n_fp += 1
            else:
                if _anywhere_same_cluster(labels_true, i, j):
                    n_fn += 1
                else:
                    n_tn += 1
    return n_tp, n_fp, n_fn, n_tn


def _anywhere_same_cluster(labels, i, j):
    for s in range(labels.shape[1]):
        if labels[i, s] == labels[j, s]:
            return True
    return False


class MultipleLabelingsPairCountingScores(PairCountingScores):

    def __init__(self, labels_true, labels_pred, remove_noise_spaces=True):
        n_tp, n_fp, n_fn, n_tn = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                                  remove_noise_spaces)
        self.n_tp = n_tp
        self.n_fp = n_fp
        self.n_fn = n_fn
        self.n_tn = n_tn


"""
Multiple Labelings Confusion Matrix
"""


class MultipleLabelingsConfusionMatrix(ConfusionMatrix):

    def __init__(self, labels_true, labels_pred, metric=nmi, remove_noise_spaces=True, metric_params={},
                 auto_rearrange=False):
        _check_number_of_points(labels_true, labels_pred)
        if labels_true.ndim == 1:
            labels_true = labels_true.reshape((-1, 1))
        if labels_pred.ndim == 1:
            labels_pred = labels_pred.reshape((-1, 1))
        if remove_noise_spaces:
            labels_true = remove_noise_spaces_from_labels(labels_true)
            labels_pred = remove_noise_spaces_from_labels(labels_pred)
        if labels_true.shape[1] == 0 or labels_pred.shape[1] == 0:
            raise Exception("labels_true or labels_pred matrix contains zero columns.")
        assert type(metric_params) is dict, "metric_params must be a dict"
        assert callable(metric), "metric must be a method"
        confusion_matrix = np.zeros((labels_true.shape[1], labels_pred.shape[1]))
        for i in range(labels_true.shape[1]):
            for j in range(labels_pred.shape[1]):
                confusion_matrix[i, j] = metric(labels_true[:, i], labels_pred[:, j], **metric_params)
        self.confusion_matrix = confusion_matrix
        if auto_rearrange:
            self.rearrange()

    def plot(self, show_text=True, figsize=(10, 10), cmap="YlGn", textcolor="black", vmin=0, vmax=1):
        _plot_confusion_matrix(self.confusion_matrix, show_text, figsize, cmap, textcolor, vmin=vmin, vmax=vmax)

    def aggregate(self, aggregation):
        if aggregation not in _CONFUSION_AGGREGATIONS:
            raise Exception("Your input '", aggregation, "' is not supported. Possibilities are: ",
                            _CONFUSION_AGGREGATIONS)
        # Permutation strategy
        if aggregation == "permut-max" or aggregation == "permut-min":
            best_score = -np.inf if aggregation == "permut-max" else np.inf
            max_sub = max(self.confusion_matrix.shape)
            min_sub = min(self.confusion_matrix.shape)
            for permut in permutations(range(max_sub)):
                score_sum = 0
                for m in range(min_sub):
                    if self.confusion_matrix.shape[0] >= self.confusion_matrix.shape[1]:
                        i = permut[m]
                        j = m
                    else:
                        i = m
                        j = permut[m]
                    score_sum += self.confusion_matrix[i, j]
                if aggregation == "permut-max" and score_sum > best_score:
                    best_score = score_sum
                if aggregation == "permut-min" and score_sum < best_score:
                    best_score = score_sum
            best_score /= self.confusion_matrix.shape[0]
        # Maximum score strategy
        elif aggregation == "max":
            best_score = np.sum(np.max(self.confusion_matrix, axis=1))
            best_score /= self.confusion_matrix.shape[0]
        # Minimum score strategy
        elif aggregation == "min":
            best_score = np.sum(np.min(self.confusion_matrix, axis=1))
            best_score /= self.confusion_matrix.shape[0]
        # Average score strategy
        elif aggregation == "avg":
            best_score = np.avg(self.confusion_matrix)
        return best_score


"""
Multiple Labelings Scoring
"""


def calculate_average_redundancy(labelings, metric=vi, remove_noise_spaces=True, confusion_matrix_obj=None,
                                 metric_params={}):
    assert confusion_matrix_obj is None or type(confusion_matrix_obj) is MultipleLabelingsConfusionMatrix, \
        "confusion_matrix must be None or of type MultipleLabelingsConfusionMatrix"
    assert labelings.ndim > 1, "Labelings has only a single column. Redundancy can not be calculated"
    if remove_noise_spaces:
        labelings = remove_noise_spaces_from_labels(labelings)
    # Calculate average confusion matrix scores
    if confusion_matrix_obj is None:
        confusion_matrix_obj = MultipleLabelingsConfusionMatrix(labelings, labelings, metric, metric_params)
    confusion_matrix = confusion_matrix_obj.confusion_matrix
    if confusion_matrix.shape != (labelings.shape[1], labelings.shape[1]):
        raise Exception(
            "Shape of the confusion matrix is wrong! Must be (|labelings| x |labelings|). In this case: ",
            (labelings.shape[1], labelings.shape[1]))
    # Return score (ignore identities)
    score = np.sum(confusion_matrix)
    score /= (confusion_matrix.shape[0] * (confusion_matrix.shape[0] - 1))
    return score


def is_multi_labelings_n_clusters_correct(labels_true, labels_pred, check_subset=True, remove_noise_spaces=True):
    _check_number_of_points(labels_true, labels_pred)
    if labels_true.ndim == 1:
        labels_true = labels_true.reshape((-1, 1))
    if labels_pred.ndim == 1:
        labels_pred = labels_pred.reshape((-1, 1))
    if remove_noise_spaces:
        labels_true = remove_noise_spaces_from_labels(labels_true)
        labels_pred = remove_noise_spaces_from_labels(labels_pred)
    # Is noise space desired?
    if labels_true.shape[1] > labels_pred.shape[1] or (
            not check_subset and labels_pred.shape[1] > labels_true.shape[1]):
        return False
    unique_labels_true = [np.unique(labels_true[:, i]) for i in range(labels_true.shape[1])]
    unique_labels_true = np.sort([len(u[u >= 0]) for u in unique_labels_true])
    unique_labels_pred = [np.unique(labels_pred[:, i]) for i in range(labels_pred.shape[1])]
    unique_labels_pred = np.sort([len(u[u >= 0]) for u in unique_labels_pred])
    if check_subset:
        is_equal = all([gt in unique_labels_pred for gt in unique_labels_true])
    else:
        is_equal = np.array_equal(unique_labels_true, unique_labels_pred)
    return is_equal

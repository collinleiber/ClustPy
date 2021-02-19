"""
Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual
support." 2012 IEEE 28th International Conference on Data
Engineering. IEEE, 2012.
"""
from cluspy.metrics.clustering_metrics import _check_number_of_points

"""
Internal functions
"""


def _pc_jaccard_score(n_tp, n_fp, n_fn):
    return 0 if (n_tp + n_fp + n_fn) == 0 else n_tp / (n_tp + n_fp + n_fn)


def _pc_rand_score(n_tp, n_fp, n_fn, n_tn):
    return 0 if (n_tp + n_fp + n_fn + n_tn) == 0 else (n_tp + n_tn) / (n_tp + n_fp + n_fn + n_tn)


def _pc_precision_score(n_tp, n_fp):
    return 0 if (n_tp + n_fp) == 0 else n_tp / (n_tp + n_fp)


def _pc_recall_score(n_tp, n_fn):
    return 0 if (n_tp + n_fn) == 0 else n_tp / (n_tp + n_fn)


def _pc_f1_score(n_tp, n_fp, n_fn):
    precision = _pc_precision_score(n_tp, n_fp)
    recall = _pc_recall_score(n_tp, n_fn)
    return 0 if (precision == 0 and recall == 0) else 2 * precision * recall / (precision + recall)


"""
External functions
"""


def pc_jaccard_score(labels_true, labels_pred):
    n_tp, n_fp, n_fn, _ = _get_pair_counting_categories(labels_true, labels_pred)
    return _pc_jaccard_score(n_tp, n_fp, n_fn)


def pc_rand_score(labels_true, labels_pred):
    n_tp, n_fp, n_fn, n_tn = _get_pair_counting_categories(labels_true, labels_pred)
    return _pc_rand_score(n_tp, n_fp, n_fn, n_tn)


def pc_precision_score(labels_true, labels_pred):
    n_tp, n_fp, _, _ = _get_pair_counting_categories(labels_true, labels_pred)
    return _pc_precision_score(n_tp, n_fp)


def pc_recall_score(labels_true, labels_pred):
    n_tp, _, n_fn, _ = _get_pair_counting_categories(labels_true, labels_pred)
    return _pc_recall_score(n_tp, n_fn)


def pc_f1_score(labels_true, labels_pred):
    n_tp, n_fp, n_fn, _ = _get_pair_counting_categories(labels_true, labels_pred)
    return _pc_f1_score(n_tp, n_fp, n_fn)


def _get_pair_counting_categories(labels_true, labels_pred):
    _check_number_of_points(labels_true, labels_pred)
    if labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise Exception("labels_true and labels_pred labels should just contain a single column.")
    n_tp = 0
    n_fp = 0
    n_fn = 0
    n_tn = 0
    for i in range(labels_pred.shape[0] - 1):
        for j in range(i + 1, labels_pred.shape[0]):
            if labels_pred[i] == labels_pred[j]:
                if labels_true[i] == labels_true[j]:
                    n_tp += 1
                else:
                    n_fp += 1
            else:
                if labels_true[i] == labels_true[j]:
                    n_fn += 1
                else:
                    n_tn += 1
    return n_tp, n_fp, n_fn, n_tn

"""
Pair Counting Object
"""

class PairCountingScores():
    def __init__(self, labels_true, labels_pred):
        _check_number_of_points(labels_true, labels_pred)
        n_tp, n_fp, n_fn, n_tn = _get_pair_counting_categories(labels_true, labels_pred)
        self.n_tp = n_tp
        self.n_fp = n_fp
        self.n_fn = n_fn
        self.n_tn = n_tn

    def jaccard(self):
        return _pc_jaccard_score(self.n_tp, self.n_fp, self.n_fn)

    def rand(self):
        return _pc_rand_score(self.n_fp, self.n_fn, self.n_fn, self.n_tn)

    def precision(self):
        return _pc_precision_score(self.n_tp, self.n_fp)

    def recall(self):
        return _pc_recall_score(self.n_tp, self.n_fn)

    def f1(self):
        return _pc_f1_score(self.n_tp, self.n_fp, self.n_fn)

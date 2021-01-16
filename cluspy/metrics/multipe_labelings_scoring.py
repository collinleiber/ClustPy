import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari, mutual_info_score as mi
from itertools import permutations
from .pair_counting_scores import PairCountingScore
from .clustering_metrics import variation_of_information, _check_number_of_points

_METRICS = ["nmi", "ami", "ari", "f1", "rand", "jaccard", "precision", "recall", "mi", "vi"]
_NR_METRICS = ["nr-f1", "nr-rand", "nr-jaccard", "nr-precision", "nr-recall"]
_AGGREGATIONS = ["max", "min", "permut-max", "permut-min", "avg"]

"""
CHECKS
"""


def _check_input(input, possibilities):
    if input not in possibilities:
        raise Exception("Your input '", input, "' is not supported. Possibilities are: ", possibilities)


"""
HELPERS
"""


def _identify_non_noise_spaces(labels):
    no_noise_spaces = [True] * labels.shape[1]
    for c in range(labels.shape[1]):
        unique_labels = np.unique(labels[:, c])
        # Consider outliers
        len_unique_labels = len(unique_labels) if np.all(unique_labels >= 0) else len(unique_labels) - 1
        if len_unique_labels == 1:
            no_noise_spaces[c] = False
    return no_noise_spaces


def _get_score_from_confusion_matrix(confusion_matrix, strategy):
    _check_input(strategy, _AGGREGATIONS)
    # Permutation strategy
    if strategy == "permut-max" or strategy == "permut-min":
        best_score = -np.inf if strategy == "permut-max" else np.inf
        max_sub = max(confusion_matrix.shape)
        min_sub = min(confusion_matrix.shape)
        for permut in permutations(range(max_sub)):
            score_sum = 0
            for m in range(min_sub):
                if confusion_matrix.shape[0] >= confusion_matrix.shape[1]:
                    i = permut[m]
                    j = m
                else:
                    i = m
                    j = permut[m]
                score_sum += confusion_matrix[i, j]
            if strategy == "permut-max" and score_sum > best_score:
                best_score = score_sum
            if strategy == "permut-min" and score_sum < best_score:
                best_score = score_sum
        best_score /= confusion_matrix.shape[0]
    # Maximum score strategy
    elif strategy == "max":
        best_score = 0
        for row in confusion_matrix:
            best_score += np.max(row)
        best_score /= confusion_matrix.shape[0]
    # Minimum score strategy
    elif strategy == "min":
        best_score = 0
        for row in confusion_matrix:
            best_score += np.min(row)
        best_score /= confusion_matrix.shape[0]
    # Average score strategy
    elif strategy == "avg":
        best_score = np.avg(confusion_matrix)
    return best_score


"""
Multiple Labelings Scoring Class
"""


class MultipleLabelingsScoring():
    def __init__(self, labels_true, labels_pred):
        _check_number_of_points(labels_true, labels_pred)
        self.labels_true = labels_true
        self.labels_pred = labels_pred
        self.labels_true_no_ns = None
        self.labels_pred_no_ns = None
        self.categories = None
        self.categories_multiple_labelings = None

    def calculate_multiple_labelings_score(self, metric, aggregation="permut-max",
                                           remove_noise_spaces=True, confusion_matrix=None):
        """

        Pair-counting measurements "Evaluation of Clusterings – Metrics and VisualSupport"

        Compare ground truth of a dataset with the labels form a multiple labelings clustering algorithm. Checks for each subspace the chosen scoring
        function and calculates the average over all subspaces. A perfect result will always be 1, the worst 0.
        :param labels_true: dataset with the true labels for each subspace
        :param metric: the metric can be "nmi", "ami", "rand", "f1" or "pc-f1" (default: "nmi")
        :return: scring result. Between 0 <= score <= 1
        """
        _check_input(metric, _METRICS + _NR_METRICS)
        _check_input(aggregation, _AGGREGATIONS)
        # Is noise space desired?
        labels_true, labels_pred = self._get_labelings(remove_noise_spaces)
        # Calculate multiple labelings scores
        if metric.startswith("nr-"):
            categories_multiple_labelings = self._get_pair_counting_categories_multiple_labelings(labels_true,
                                                                                                  labels_pred)
            if metric == "nr-rand":
                return categories_multiple_labelings.rand()
            elif metric == "nr-f1":
                return categories_multiple_labelings.f1()
            elif metric == "nr-jaccard":
                return categories_multiple_labelings.jaccard()
            elif metric == "nr-precision":
                return categories_multiple_labelings.precision()
            elif metric == "nr-recall":
                return categories_multiple_labelings.recall()
        # Calculate average confusion matrix scores
        if confusion_matrix is None:
            confusion_matrix = self.get_scoring_confusion_matrix(metric, remove_noise_spaces)
        if confusion_matrix.shape != (labels_true.shape[1], labels_pred.shape[1]):
            raise Exception(
                "Shape of the confusion matrix is wrong! Must be (|ground truth labelings| x |prediction labelings|). In this case: ",
                (labels_true.shape[1], labels_pred.shape[1]))
        # Return best found score
        if confusion_matrix.shape[0] == 0 or confusion_matrix.shape[1] == 0:
            return 0
        best_score = _get_score_from_confusion_matrix(confusion_matrix, aggregation)
        return best_score

    def get_scoring_confusion_matrix(self, metric, remove_noise_spaces=True):
        _check_input(metric, _METRICS)
        # Is noise space desired?
        labels_true, labels_pred = self._get_labelings(remove_noise_spaces)
        if labels_true.shape[1] == 0 or labels_pred.shape[1] == 0:
            raise Exception("labels_true or labels_pred matrix contains zero columns.")
        confusion_matrix = np.zeros((labels_true.shape[1], labels_pred.shape[1]))
        for i in range(labels_true.shape[1]):
            for j in range(labels_pred.shape[1]):
                if metric == "nmi":
                    confusion_matrix[i, j] = nmi(labels_true[:, i], labels_pred[:, j], "arithmetic")
                elif metric == "ami":
                    confusion_matrix[i, j] = ami(labels_true[:, i], labels_pred[:, j], "arithmetic")
                elif metric == "ari":
                    confusion_matrix[i, j] = ari(labels_true[:, i], labels_pred[:, j])
                elif metric == "mi":
                    confusion_matrix[i, j] = mi(labels_true[:, i], labels_pred[:, j])
                elif metric == "vi":
                    confusion_matrix[i, j] = variation_of_information(labels_true[:, i], labels_pred[:, j])
                else:
                    categories = self._get_pair_counting_categories(labels_true, labels_pred)[i][j]
                    if metric == "rand":
                        confusion_matrix[i, j] = categories.rand()
                    elif metric == "f1":
                        confusion_matrix[i, j] = categories.f1()
                    elif metric == "jaccard":
                        confusion_matrix[i, j] = categories.jaccard()
                    elif metric == "precision":
                        confusion_matrix[i, j] = categories.precision()
                    elif metric == "recall":
                        confusion_matrix[i, j] = categories.recall()
        return confusion_matrix

    def is_n_clusters_correct(self, remove_noise_spaces=True, check_subset=False):
        # Is noise space desired?
        labels_true, labels_pred = self._get_labelings(remove_noise_spaces)
        if labels_true.shape[1] > labels_pred.shape[1] or (
                not check_subset and labels_pred.shape[1] > labels_true.shape[1]):
            return False
        unique_labels_true = np.sort([len(np.unique(labels_true[:, i])) for i in range(labels_true.shape[1])])
        unique_labels_pred = np.sort([len(np.unique(labels_pred[:, i])) for i in range(labels_pred.shape[1])])
        if check_subset:
            result = all([gt in unique_labels_pred for gt in unique_labels_true])
        else:
            result = np.array_equal(unique_labels_true, unique_labels_pred)
        return result

    def measure_average_redundancy(self, metric, labelings="pred", remove_noise_spaces=True, confusion_matrix=None):
        _check_input(metric, _METRICS)
        _check_input(labelings, ["pred", "gt"])
        # Is noise space desired?
        labels_true, labels_pred = self._get_labelings(remove_noise_spaces)
        if labelings == "pred":
            labelings_matrix = labels_pred
        elif labelings == "gt":
            labelings_matrix = labels_true
        # Calculate average confusion matrix scores
        if confusion_matrix is None:
            confusion_matrix = self.get_redundancy_confusion_matrix(metric, labelings, remove_noise_spaces)
        if confusion_matrix.shape != (labelings_matrix.shape[1], labelings_matrix.shape[1]):
            raise Exception(
                "Shape of the confusion matrix is wrong! Must be (|labelings| x |labelings|). In this case: ",
                (labelings_matrix.shape[1], labelings_matrix.shape[1]))
        # Return score (ignore identities)
        score = np.sum(confusion_matrix)
        score /= (confusion_matrix.shape[0] * (confusion_matrix.shape[0] - 1))
        return score

    def get_redundancy_confusion_matrix(self, metric, labelings="pred", remove_noise_spaces=True):
        _check_input(metric, ["nmi", "ami", "ari", "mi", "vi"])
        _check_input(labelings, ["pred", "gt"])
        # Is noise space desired?
        labels_true, labels_pred = self._get_labelings(remove_noise_spaces)
        if labelings == "pred":
            labelings_matrix = labels_pred
        elif labelings == "gt":
            labelings_matrix = labels_true
        if labelings_matrix.shape[1] == 0:
            raise Exception("Labelings matrix contains zero columns.")
        confusion_matrix = np.zeros((labelings_matrix.shape[1], labelings_matrix.shape[1]))
        for i in range(labelings_matrix.shape[1] - 1):
            for j in range(i + 1, labelings_matrix.shape[1]):
                if metric == "nmi":
                    result = nmi(labelings_matrix[:, i], labelings_matrix[:, j], "arithmetic")
                elif metric == "ami":
                    result = ami(labelings_matrix[:, i], labelings_matrix[:, j], "arithmetic")
                elif metric == "ari":
                    result = ari(labelings_matrix[:, i], labelings_matrix[:, j])
                elif metric == "mi":
                    result = mi(labelings_matrix[:, i], labelings_matrix[:, j])
                elif metric == "vi":
                    result = variation_of_information(labelings_matrix[:, i], labelings_matrix[:, j])
                confusion_matrix[i, j] = result
                confusion_matrix[j, i] = result
        return confusion_matrix

    """
    CLASS HELPERS
    """

    def _get_labelings(self, remove_noise_spaces):
        if remove_noise_spaces:
            if self.labels_true_no_ns is None:
                self.labels_true_no_ns = _identify_non_noise_spaces(self.labels_true)
            labels_true = self.labels_true[:, self.labels_true_no_ns]
            if self.labels_pred_no_ns is None:
                self.labels_pred_no_ns = _identify_non_noise_spaces(self.labels_pred)
            labels_pred = self.labels_pred[:, self.labels_pred_no_ns]
        else:
            labels_true = self.labels_true
            labels_pred = self.labels_pred
        return labels_true, labels_pred

    def _get_pair_counting_categories_multiple_labelings(self, labels_true, labels_pred):
        if self.categories_multiple_labelings is None:
            self.categories_multiple_labelings = PairCountingScore(labels_true, labels_pred, True)
        return self.categories_multiple_labelings

    def _get_pair_counting_categories(self, labels_true, labels_pred):
        if self.categories is None:
            categories = [[None] * labels_pred.shape[1]] * labels_true.shape[1]
            for i in range(labels_true.shape[1]):
                for j in range(labels_pred.shape[1]):
                    categories[i][j] = PairCountingScore(labels_true[:, i], labels_pred[:, j], False)
            self.categories = categories
        return self.categories

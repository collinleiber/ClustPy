import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari, mutual_info_score as mi
from itertools import permutations
from .pair_counting_scores import PairCountingScore
from .variation_of_information import variation_of_information, _check_number_of_points

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
Non-Redundant Scoring Class
"""


class MultipleLabelingsScoring():
    def __init__(self, ground_truth, prediction):
        _check_number_of_points(ground_truth, prediction)
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.ground_truth_no_ns = None
        self.prediction_no_ns = None
        self.categories = None
        self.categories_multiple_labelings = None

    def calculate_multiple_labelings_score(self, metric, aggregation="permut-max",
                                           remove_noise_spaces=True, confusion_matrix=None):
        """

        Pair-counting measurements "Evaluation of Clusterings – Metrics and VisualSupport"

        Compare ground truth of a dataset with the labels form a non redundant clustering algorithm. Checks for each subspace the chosen scoring
        function and calculates the average over all subspaces. A perfect result will always be 1, the worst 0.
        :param ground_truth: dataset with the true labels for each subspace
        :param metric: the metric can be "nmi", "ami", "rand", "f1" or "pc-f1" (default: "nmi")
        :return: scring result. Between 0 <= score <= 1
        """
        _check_input(metric, _METRICS + _NR_METRICS)
        _check_input(aggregation, _AGGREGATIONS)
        # Is noise space desired?
        ground_truth, prediction = self._get_labelings(remove_noise_spaces)
        # Calculate non-redundant scores
        if metric.startswith("nr-"):
            categories_multiple_labelings = self._get_pair_counting_categories_multiple_labelings(ground_truth,
                                                                                                  prediction)
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
        if confusion_matrix.shape != (ground_truth.shape[1], prediction.shape[1]):
            raise Exception(
                "Shape of the confusion matrix is wrong! Must be (|ground truth labelings| x |prediction labelings|). In this case: ",
                (ground_truth.shape[1], prediction.shape[1]))
        # Return best found score
        if confusion_matrix.shape[0] == 0 or confusion_matrix.shape[1] == 0:
            return 0
        best_score = _get_score_from_confusion_matrix(confusion_matrix, aggregation)
        return best_score

    def get_scoring_confusion_matrix(self, metric, remove_noise_spaces=True):
        _check_input(metric, _METRICS)
        # Is noise space desired?
        ground_truth, prediction = self._get_labelings(remove_noise_spaces)
        if ground_truth.shape[1] == 0 or prediction.shape[1] == 0:
            raise Exception("Ground truth or prediction matrix contains zero columns.")
        confusion_matrix = np.zeros((ground_truth.shape[1], prediction.shape[1]))
        for i in range(ground_truth.shape[1]):
            for j in range(prediction.shape[1]):
                if metric == "nmi":
                    confusion_matrix[i, j] = nmi(ground_truth[:, i], prediction[:, j], "arithmetic")
                elif metric == "ami":
                    confusion_matrix[i, j] = ami(ground_truth[:, i], prediction[:, j], "arithmetic")
                elif metric == "ari":
                    confusion_matrix[i, j] = ari(ground_truth[:, i], prediction[:, j])
                elif metric == "mi":
                    confusion_matrix[i, j] = mi(ground_truth[:, i], prediction[:, j])
                elif metric == "vi":
                    confusion_matrix[i, j] = variation_of_information(ground_truth[:, i], prediction[:, j])
                else:
                    categories = self._get_pair_counting_categories(ground_truth, prediction)[i][j]
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
        ground_truth, prediction = self._get_labelings(remove_noise_spaces)
        if ground_truth.shape[1] > prediction.shape[1] or (
                not check_subset and prediction.shape[1] > ground_truth.shape[1]):
            return False
        unique_ground_truth = np.sort([len(np.unique(ground_truth[:, i])) for i in range(ground_truth.shape[1])])
        unique_prediction = np.sort([len(np.unique(prediction[:, i])) for i in range(prediction.shape[1])])
        if check_subset:
            result = all([gt in unique_prediction for gt in unique_ground_truth])
        else:
            result = np.array_equal(unique_ground_truth, unique_prediction)
        return result

    def measure_average_redundancy(self, metric, labelings="pred", remove_noise_spaces=True, confusion_matrix=None):
        _check_input(metric, _METRICS)
        _check_input(labelings, ["pred", "gt"])
        # Is noise space desired?
        ground_truth, prediction = self._get_labelings(remove_noise_spaces)
        if labelings == "pred":
            labelings_matrix = prediction
        elif labelings == "gt":
            labelings_matrix = ground_truth
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
        ground_truth, prediction = self._get_labelings(remove_noise_spaces)
        if labelings == "pred":
            labelings_matrix = prediction
        elif labelings == "gt":
            labelings_matrix = ground_truth
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
            if self.ground_truth_no_ns is None:
                self.ground_truth_no_ns = _identify_non_noise_spaces(self.ground_truth)
            ground_truth = self.ground_truth[:, self.ground_truth_no_ns]
            if self.prediction_no_ns is None:
                self.prediction_no_ns = _identify_non_noise_spaces(self.prediction)
            prediction = self.prediction[:, self.prediction_no_ns]
        else:
            ground_truth = self.ground_truth
            prediction = self.prediction
        return ground_truth, prediction

    def _get_pair_counting_categories_multiple_labelings(self, ground_truth, prediction):
        if self.categories_multiple_labelings is None:
            self.categories_multiple_labelings = PairCountingScore(ground_truth, prediction, True)
        return self.categories_multiple_labelings

    def _get_pair_counting_categories(self, ground_truth, prediction):
        if self.categories is None:
            categories = [[None] * prediction.shape[1]] * ground_truth.shape[1]
            for i in range(ground_truth.shape[1]):
                for j in range(prediction.shape[1]):
                    categories[i][j] = PairCountingScore(ground_truth[:, i], prediction[:, j], False)
            self.categories = categories
        return self.categories

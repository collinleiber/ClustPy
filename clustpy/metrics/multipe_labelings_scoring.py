import numpy as np
from clustpy.metrics.clustering_metrics import _check_number_of_points
from clustpy.metrics.pair_counting_scores import PairCountingScores, _f1_score, _recall_score, _precision_score, \
    _rand_score, _jaccard_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.metrics.confusion_matrix import ConfusionMatrix, _plot_confusion_matrix
from scipy.optimize import linear_sum_assignment
from collections.abc import Callable

"""
HELPERS
"""


def remove_noise_spaces_from_labels(labels: np.ndarray) -> np.ndarray:
    """
    Remove optional noise spaces (n_clusters=1) from labels.
    If outliers are present (label=-1) but all non-outlier labels (label>=0) are equal, the label column will still be regarded as noise space.

    Parameters
    ----------
    labels : np.ndarray
        The input labels

    Returns
    -------
    labels_new : np.ndarray
        The output labels
    """
    no_noise_spaces = np.zeros(labels.shape[1], dtype=bool)
    for c in range(labels.shape[1]):
        unique_labels = np.unique(labels[:, c])
        # Consider outliers (label < 0)
        no_noise_spaces[c] = len(unique_labels[unique_labels >= 0]) != 1
    labels_new = labels[:, no_noise_spaces]
    return labels_new


"""
Multiple Labelings Pair Counting Scores
"""


def multiple_labelings_pc_jaccard_score(labels_true: np.ndarray, labels_pred: np.ndarray,
                                        remove_noise_spaces: bool = True) -> float:
    """
    Calculate the jaccard score for multiple labelings.
    Jaccard score = n_tp / (n_tp + n_fp + n_fn).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

    Returns
    -------
    score : float
        The jaccard score

    References
    ----------
    Jaccard, Paul. "Lois de distribution florale dans la zone alpine."
    Bull Soc Vaudoise Sci Nat 38 (1902): 69-130.

    and

    Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual support."
    2012 IEEE 28th International Conference on Data Engineering. IEEE, 2012.
    """
    n_tp, n_fp, n_fn, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                           remove_noise_spaces)
    score = _jaccard_score(n_tp, n_fp, n_fn)
    return score


def multiple_labelings_pc_rand_score(labels_true: np.ndarray, labels_pred: np.ndarray,
                                     remove_noise_spaces: bool = True) -> float:
    """
    Calculate the rand score for multiple labelings.
    Rand score = (n_tp + n_tn) / (n_tp + n_fp + n_fn + n_tn).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

    Returns
    -------
    score : float
        The rand score

    References
    ----------
    Rand, William M. "Objective criteria for the evaluation of clustering methods."
    Journal of the American Statistical association 66.336 (1971): 846-850.

    and

    Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual support."
    2012 IEEE 28th International Conference on Data Engineering. IEEE, 2012.
    """
    n_tp, n_fp, n_fn, n_tn = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                              remove_noise_spaces)
    score = _rand_score(n_tp, n_fp, n_fn, n_tn)
    return score


def multiple_labelings_pc_precision_score(labels_true: np.ndarray, labels_pred: np.ndarray,
                                          remove_noise_spaces: bool = True) -> float:
    """
    Calculate the precision for multiple labelings.
    Precision score = n_tp / (n_tp + n_fp).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

    Returns
    -------
    score : float
        The precision

    References
    ----------
    Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
    American Documentation (pre-1986) 6.2 (1955): 93.

    and

    Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual support."
    2012 IEEE 28th International Conference on Data Engineering. IEEE, 2012.
    """
    n_tp, n_fp, _, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred, remove_noise_spaces)
    score = _precision_score(n_tp, n_fp)
    return score


def multiple_labelings_pc_recall_score(labels_true: np.ndarray, labels_pred: np.ndarray,
                                       remove_noise_spaces: bool = True) -> float:
    """
    Calculate the recall for multiple labelings.
    Recall score = n_tp / (n_tp + n_fn).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

    Returns
    -------
    score : float
        The recall

    References
    ----------
    Allen, Kent, et al. "Machine literature searching VIII. Operational criteria for designing information retrieval systems."
    American Documentation (pre-1986) 6.2 (1955): 93.

    and

    Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual support."
    2012 IEEE 28th International Conference on Data Engineering. IEEE, 2012.
    """
    n_tp, _, n_fn, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred, remove_noise_spaces)
    score = _recall_score(n_tp, n_fn)
    return score


def multiple_labelings_pc_f1_score(labels_true: np.ndarray, labels_pred: np.ndarray,
                                   remove_noise_spaces: bool = True) -> float:
    """
    Calculate the f1 score for multiple labelings.
    F1 score = 2 * precision * recall / (precision + recall).
    In the clustering domain the calculation is based on pair-counting as the true label ids are unknown.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

    Returns
    -------
    score : float
        The f1 score

    References
    ----------
    Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth-Heinemann.

    and

    Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual support."
    2012 IEEE 28th International Conference on Data Engineering. IEEE, 2012.
    """
    n_tp, n_fp, n_fn, _ = _get_multiple_labelings_pair_counting_categories(labels_true, labels_pred,
                                                                           remove_noise_spaces)
    score = _f1_score(n_tp, n_fp, n_fn)
    return score


def _get_multiple_labelings_pair_counting_categories(labels_true: np.ndarray, labels_pred: np.ndarray,
                                                     remove_noise_spaces: bool) -> (int, int, int, int):
    """
    Get the number of 'true positives', 'false positives', 'false negatives' and 'true negatives' to calculate pair-counting scores using multiple labelings.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score

    Returns
    -------
    tuple : (int, int, int, int)
        The number of true positives,
        The number of false positives,
        The number of false negatives,
        The number of true negatives
    """
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


def _anywhere_same_cluster(labels: np.ndarray, i: int, j: int) -> bool:
    """
    Check if the two samples i and j share a cluster label in any subspace.

    Parameters
    ----------
    labels : np.ndarray
        The set of labelings
    i : int
        Id of the first sample
    j : int
        Id of the second sample

    Returns
    -------
    anywhere_same : bool
        Boolean indicating of they share a cluster label anywhere
    """
    anywhere_same = False
    for s in range(labels.shape[1]):
        if labels[i, s] == labels[j, s]:
            anywhere_same = True
    return anywhere_same


class MultipleLabelingsPairCountingScores(PairCountingScores):
    """
    Obtain all parameters that are necessary to calculate the pair-counting scores 'jaccard', 'rand', 'precision', 'recall' and 'f1'.
    These parameters are the number of 'true positives', 'false positives', 'false negatives' and 'true negatives'.
    The resulting object can call all pair-counting score methods.
    In contrast to common pair-counting calculations, a match between two samples counts if it occurs in at least one label set.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

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
    Achtert, Elke, et al. "Evaluation of clusterings--metrics and visual support."
    2012 IEEE 28th International Conference on Data Engineering. IEEE, 2012.
    """

    def __init__(self, labels_true: np.ndarray, labels_pred: np.ndarray, remove_noise_spaces: bool = True):
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
    """
    A Multi Labelings Confusion Matrix is a special type of Confusion Matrix where each cell corresonds to the clustering score between one ground truth label set and one predicted label set.
    Therefore, the shape is equal to (number of ground truth labelings, number of predicted labelings).
    The scoring metric used can be freely selected by the user and is called up as follows: metric(labels_gt, labels_pred).
    Additional parameters for the chosen metric can be set by using the metric_params dictionary.
    The default metric is the 'normalized mutual information' from sklearn.metrics.normalized_mutual_info_score.

    The Multi Labelings Confusion Matrix can also be used to calculate the average redundancy of a set of labels.
    Therefore, it is recommended to set metric=clustpy.metrics.variation_of_information and aggregate the confusion matrix with aggregation_strategy="mean_wo_diag".

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)
    metric : Callable
        The chosen scoring metric (default: sklearn.metrics.normalized_mutual_info_score)
    metric_params : dict
        Additional parameters for the scoring metric (default: {})

    Attributes
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix

    Examples
    ----------
    >>> # Calculate average redundancy
    >>> from clustpy.metrics import variation_of_information as vi
    >>> labels = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
    >>>                    [0, 0, 0, 0, 1, 1, 1, 1],
    >>>                    [0, 0, 1, 1, 1, 1, 1, 1],
    >>>                    [1, 2, 3, 4, 5, 6, 7, 8]]).T
    >>> mlcm = MultipleLabelingsConfusionMatrix(labels, labels, metric=vi)
    >>> mlcm.aggregate("mean_wo_diag")
    """

    def __init__(self, labels_true: np.ndarray, labels_pred: np.ndarray, metric: Callable = nmi,
                 remove_noise_spaces: bool = True, metric_params: dict = {}):
        _check_number_of_points(labels_true, labels_pred)
        assert type(metric_params) is dict, "metric_params must be a dict"
        assert callable(metric), "metric must be a method"
        # Reshape labels if we have only a single set of labels
        if labels_true.ndim == 1:
            labels_true = labels_true.reshape((-1, 1))
        if labels_pred.ndim == 1:
            labels_pred = labels_pred.reshape((-1, 1))
        # (Optional) remove noise spaces
        if remove_noise_spaces:
            labels_true = remove_noise_spaces_from_labels(labels_true)
            labels_pred = remove_noise_spaces_from_labels(labels_pred)
        if labels_true.shape[1] == 0 or labels_pred.shape[1] == 0:
            raise Exception("labels_true or labels_pred matrix contains zero columns.")
        # Create confusion matrix
        confusion_matrix = np.zeros((labels_true.shape[1], labels_pred.shape[1]))
        for i in range(labels_true.shape[1]):
            for j in range(labels_pred.shape[1]):
                confusion_matrix[i, j] = metric(labels_true[:, i], labels_pred[:, j], **metric_params)
        self.confusion_matrix = confusion_matrix

    def plot(self, show_text: bool = True, figsize: tuple = (10, 10), cmap: str = "YlGn", textcolor: str = "black",
             vmin: float = 0.0, vmax: float = 1.0) -> None:
        """
        Plot the Multiple Labelings Confusion Matrix.
        Same plot as for a regular Confusion Matrix but vmax is by default set to 1 as it is usually the maximum value for clustering metrics.

        Parameters
        ----------
        show_text : bool
            Show the value in each cell as text (default: True)
        figsize : tuple
            Tuple indicating the height and width of the plot (default: (10, 10))
        cmap : str
            Colormap used for the plot (default: "YlGn")
        textcolor : str
            Color of the text. Only relevant if show_text is True (default: "black")
        vmin : float
            Minimum possible value within a cell of the confusion matrix.
            If None, it will be set as the minimum value within the confusion matrix.
            Used to choose the color from the colormap (default: 0.0)
        vmax : float
            Maximum possible value within a cell of the confusion matrix.
            If None, it will be set as the maximum value within the confusion matrix.
            Used to choose the color from the colormap (default: 1.0)
        """
        _plot_confusion_matrix(self.confusion_matrix, show_text, figsize, cmap, textcolor, vmin=vmin, vmax=vmax)

    def aggregate(self, aggregation_strategy: str = "max") -> float:
        """
        Aggregate the Multiple Labelings Confusion Matrix to a single value.
        Different strategies of aggregations are possible:
            - "max": Choose for each ground truth set of labels the predicted set of labels with the maximum value (prediction labeling can be used multiple times).
            - "min": Choose for each ground truth set of labels the predicted set of labels with the minimum value (prediction labeling can be used multiple times).
            - "permut-max": Assign each ground truth labeling one predicted labeling, so that the sum of the combinations is maximzed (prediction labeling can only be assigned to one ground truth labeling).
            - "permut-min": Assign each ground truth labeling one predicted labeling, so that the sum of the combinations is minimized (prediction labeling can only be assigned to one ground truth labeling).
            - "mean": Calculate the mean value of all values in the confusion matrix.
            - "mean_wo_diag": Calculate mean value ignoring the diagonal. E.g. used to calculate average redundancy. Note: Confusion matrix must be quadratic!
        In the end all results (except for 'mean') are divided by the number of ground truth labelings.

        Parameters
        ----------
        aggregation_strategy : str
            The aggregation strategy (default: "max")

        Returns
        -------
        score : float
            The resulting aggregated score

        Examples
        ----------
        >>> from clustpy.metrics import MultipleLabelingsConfusionMatrix
        >>> mlcm = MultipleLabelingsConfusionMatrix(np.array([0, 1]), np.array([0, 1]))
        >>> # Overwrite confusion matrix (for demonstration purposes only)
        >>> mlcm.confusion_matrix = np.array([[0., 0.1, 0.2],
        >>>                                   [1, 0.9, 0.8],
        >>>                                   [0, 0.2, 0.3]])
        >>> mlcm.aggregate("max") == 1.5 / 3 # True
        >>> mlcm.aggregate("min") == 0.8 / 3 # True
        >>> mlcm.aggregate("permut-max") == 1.4 / 3 # True
        >>> mlcm.aggregate("permut-min") == 0.9 / 3 # True
        >>> mlcm.aggregate("mean") == 3.5 / 9 # True
        """
        possible_aggregations = ["max", "min", "permut-max", "permut-min", "mean", "mean_wo_diag"]
        aggregation_strategy = aggregation_strategy.lower()
        assert aggregation_strategy in possible_aggregations, "Your input '{0}' is not supported. Possibilities are: {1}.".format(
            aggregation_strategy, possible_aggregations)
        # Permutation strategies
        if aggregation_strategy == "permut-max" or aggregation_strategy == "permut-min":
            max_number_labelings = max(self.confusion_matrix.shape)
            rearranged_confusion_matrix = np.zeros((max_number_labelings, max_number_labelings))
            if aggregation_strategy == "permut-max":
                # Linear sum assignment tries to minimize the diagonal sum -> use negative confusion_matrix
                rearranged_confusion_matrix[:self.confusion_matrix.shape[0],
                :self.confusion_matrix.shape[1]] = -self.confusion_matrix
                indices = linear_sum_assignment(rearranged_confusion_matrix)
                rearranged_confusion_matrix = -rearranged_confusion_matrix[:, indices[1]]
            else:
                rearranged_confusion_matrix[:self.confusion_matrix.shape[0],
                :self.confusion_matrix.shape[1]] = self.confusion_matrix
                indices = linear_sum_assignment(rearranged_confusion_matrix)
                rearranged_confusion_matrix = rearranged_confusion_matrix[:, indices[1]]
            score = np.sum(rearranged_confusion_matrix.diagonal())
            score /= self.confusion_matrix.shape[0]
        # Maximum score strategy
        elif aggregation_strategy == "max":
            score = np.sum(np.max(self.confusion_matrix, axis=1))
            score /= self.confusion_matrix.shape[0]
        # Minimum score strategy
        elif aggregation_strategy == "min":
            score = np.sum(np.min(self.confusion_matrix, axis=1))
            score /= self.confusion_matrix.shape[0]
        # Average score strategy
        elif aggregation_strategy == "mean":
            score = np.mean(self.confusion_matrix)
        # Average without diagonal strategy
        elif aggregation_strategy == "mean_wo_diag":
            assert self.confusion_matrix.shape[0] == self.confusion_matrix.shape[
                1], "Confusion matrix must be quadratic."
            score = np.sum(self.confusion_matrix) - np.sum(self.confusion_matrix.diagonal())
            score /= (self.confusion_matrix.shape[0] * (self.confusion_matrix.shape[0] - 1))
        return score


"""
Other Multiple Labelings Scores
"""


def is_multi_labelings_n_clusters_correct(labels_true: np.ndarray, labels_pred: np.ndarray, check_subset: bool = True,
                                          remove_noise_spaces: bool = True) -> bool:
    """
    Check if number of clusters of two sets of labelings matches.
    The parameter check_subset defines, if it is sufficient if the number of clusters of a subset of the predicted label set (n_clusters_pred) is equal to the number of clusters of the true label set (n_clusters_true).
    E.g. assume n_clusters_true is [4, 3, 1] and n_clusters_pred is [4, 2, 1].
    In this case is_multi_labelings_n_clusters_correct(labels_true, labels_pred) will be False.
    Now let us assume n_clusters_true is still [4, 3, 1] but n_clusters_pred is [4, 3, 2, 1].
    In this case is_multi_labelings_n_clusters_correct(labels_true, labels_pred) will be False if check_subset is False and True otherwise.

    Parameters
    ----------
    labels_true : np.ndarray
        The true set of labelings. Shape must match (n_samples, n_subspaces)
    labels_pred : np.ndarray
        The predicted set of labelings. Shape must match (n_samples, n_subspaces)
    check_subset : bool
        Boolean defines if it is sufficient if a subset of n_clusters_pred is equal to n_clusters_true (default: True)
    remove_noise_spaces : bool
        Defines if optional noise spaces should be ignored when calculating the score (default: True)

    Returns
    -------
    is_equal : bool
        Boolean indicating if the number of clusters of labels_true and labels_pred matches
    """
    _check_number_of_points(labels_true, labels_pred)
    if labels_true.ndim == 1:
        labels_true = labels_true.reshape((-1, 1))
    if labels_pred.ndim == 1:
        labels_pred = labels_pred.reshape((-1, 1))
    if remove_noise_spaces:
        labels_true = remove_noise_spaces_from_labels(labels_true)
    # If number of true labelings is larger than number of predicted labelings, return False
    # If number of predicted labelings is larger than number of true labelings and checn_subste is True, return False
    if labels_true.shape[1] > labels_pred.shape[1] or (
            not check_subset and labels_pred.shape[1] > labels_true.shape[1]):
        return False
    # Start main method by calculating n_clusters
    unique_labels_true = [np.unique(labels_true[:, i]) for i in range(labels_true.shape[1])]
    unique_labels_true = np.sort([len(u[u >= 0]) for u in unique_labels_true])  # Ignore outliers with label=-1
    unique_labels_pred = [np.unique(labels_pred[:, i]) for i in range(labels_pred.shape[1])]
    unique_labels_pred = np.sort([len(u[u >= 0]) for u in unique_labels_pred])  # Ignore outliers with label=-1
    if check_subset:
        for gt in unique_labels_true:
            # Check if all n_clusters of the true labelings are contained in the predicted labelings
            if gt in unique_labels_pred:
                index = np.where(gt == unique_labels_pred)[0][0]
                unique_labels_pred = np.delete(unique_labels_pred, index)
            else:
                return False
        is_equal = True
    else:
        is_equal = np.array_equal(unique_labels_true, unique_labels_pred)
    return is_equal

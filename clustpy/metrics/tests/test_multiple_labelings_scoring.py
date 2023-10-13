from clustpy.metrics import remove_noise_spaces_from_labels, MultipleLabelingsPairCountingScores, \
    MultipleLabelingsConfusionMatrix, multiple_labelings_pc_f1_score, multiple_labelings_pc_jaccard_score, \
    multiple_labelings_pc_precision_score, multiple_labelings_pc_rand_score, multiple_labelings_pc_recall_score, \
    is_multi_labelings_n_clusters_correct
from clustpy.metrics.multipe_labelings_scoring import _anywhere_same_cluster
import numpy as np
from unittest.mock import patch


def test_remove_noise_spaces_from_labels():
    # without outliers
    labels = np.array([[1, 2, 1, 2, 1, 2, 1],
                       [0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 2],
                       [1, 2, 3, 4, 5, 6, 7],
                       [1, 1, 1, 1, 1, 1, 1]]).T
    labels_new = remove_noise_spaces_from_labels(labels)
    assert np.array_equal(labels_new, labels[:, [0, 2, 3]])
    # with outliers
    labels = np.array([[1, -1, -1, 2, 1, 2, 1],
                       [0, -1, 0, 0, -1, 0, 0],
                       [1, -1, -1, 1, 1, 1, 2],
                       [1, -1, 3, 4, -1, 6, 7],
                       [1, -1, -1, 1, 1, 1, 1]]).T
    labels_new = remove_noise_spaces_from_labels(labels)
    assert np.array_equal(labels_new, labels[:, [0, 2, 3]])


def test_anywhere_same_cluster():
    labels = np.array([[0, 1, 2], [0, 1, 0], [0, 1, 2]]).T
    assert _anywhere_same_cluster(labels, 0, 1) == False
    assert _anywhere_same_cluster(labels, 0, 2) == True


def test_MultipleLabelingsPairCountingScores():
    labels_true = np.array(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2],
         [0, 1, 2, 0, 1, 2, 0, 1, 2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    labels_pred = np.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 1, 1, 2, 2, 3, 3, 4],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, -1, 0, 0, -1]]).T
    # Without removing noise spaces
    remove_noise = False
    mlpcs = MultipleLabelingsPairCountingScores(labels_true, labels_pred, remove_noise)
    assert mlpcs.n_tp == 36
    assert mlpcs.n_fp == 0
    assert mlpcs.n_fn == 0
    assert mlpcs.n_tn == 0
    assert mlpcs.jaccard() == 1
    assert mlpcs.jaccard() == multiple_labelings_pc_jaccard_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.rand() == 1
    assert mlpcs.rand() == multiple_labelings_pc_rand_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.precision() == 1
    assert mlpcs.precision() == multiple_labelings_pc_precision_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.recall() == 1
    assert mlpcs.recall() == multiple_labelings_pc_recall_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.f1() == 1
    assert mlpcs.f1() == multiple_labelings_pc_f1_score(labels_true, labels_pred, remove_noise)
    # With removing noise spaces
    remove_noise = True
    mlpcs = MultipleLabelingsPairCountingScores(labels_true, labels_pred, remove_noise)
    assert mlpcs.n_tp == 11
    assert mlpcs.n_fp == 6
    assert mlpcs.n_fn == 7
    assert mlpcs.n_tn == 12
    assert mlpcs.jaccard() == 11 / 24
    assert mlpcs.jaccard() == multiple_labelings_pc_jaccard_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.rand() == 23 / 36
    assert mlpcs.rand() == multiple_labelings_pc_rand_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.precision() == 11 / 17
    assert mlpcs.precision() == multiple_labelings_pc_precision_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.recall() == 11 / 18
    assert mlpcs.recall() == multiple_labelings_pc_recall_score(labels_true, labels_pred, remove_noise)
    assert mlpcs.f1() == 2 * (11 / 17) * (11 / 18) / ((11 / 17) + (11 / 18))
    assert mlpcs.f1() == multiple_labelings_pc_f1_score(labels_true, labels_pred, remove_noise)


def test_is_multi_labelings_n_clusters_correct():
    labels_true = np.array([[0, 0, 0, 0, 1],
                            [0, 0, -1, 1, 2],
                            [0, 0, 0, 0, 0]]).T
    labels_pred = np.array([[[0, 0, -1, 0, 1],
                             [0, 0, 0, 1, 2],
                             [0, 0, 1, 2, 3]]]).T
    assert is_multi_labelings_n_clusters_correct(labels_true, labels_pred, check_subset=True,
                                                 remove_noise_spaces=True) == True
    assert is_multi_labelings_n_clusters_correct(labels_true, labels_pred, check_subset=True,
                                                 remove_noise_spaces=False) == False
    assert is_multi_labelings_n_clusters_correct(labels_true, labels_pred, check_subset=False,
                                                 remove_noise_spaces=True) == False
    assert is_multi_labelings_n_clusters_correct(labels_true, labels_pred, check_subset=False,
                                                 remove_noise_spaces=False) == False
    # Check what happens if one count occurs twice
    labels_true = np.array([[0, 0, 0, 0, 1],
                            [0, 0, -1, 1, 2],
                            [0, 0, 0, 1, 2],
                            [0, 0, 0, 0, 0]]).T
    assert is_multi_labelings_n_clusters_correct(labels_true, labels_pred, check_subset=True,
                                                 remove_noise_spaces=True) == False


"""
Tests regarding the MultipleLabelingsConfusionMatrix object
"""


def test_multiple_labelings_confusion_matrix_object():
    # First test
    labels_true = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    labels_pred = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 2, 3, 4, 5, 6, 7, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    mlcm = MultipleLabelingsConfusionMatrix(labels_true, labels_pred, remove_noise_spaces=True)
    expected_cm = np.array([[1, 0.5],
                            [0, 0.5]])
    assert np.allclose(mlcm.confusion_matrix, expected_cm)
    # Second test
    labels_true = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    labels_pred = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 2, 3, 4, 5, 6, 7, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    mlcm = MultipleLabelingsConfusionMatrix(labels_true, labels_pred, remove_noise_spaces=False)
    expected_cm = np.array([[1, 0.5, 0],
                            [0, 0.5, 0],
                            [0, 0, 1],
                            [0, 0, 1]])
    assert np.allclose(mlcm.confusion_matrix, expected_cm)


def test_multiple_labelings_confusion_matrix_aggregate():
    labels_true = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    labels_pred = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 2, 3, 4, 5, 6, 7, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    mlcm = MultipleLabelingsConfusionMatrix(labels_true, labels_pred, remove_noise_spaces=False)
    assert mlcm.aggregate("max") == 3.5 / 4
    assert mlcm.aggregate("min") == 0
    assert mlcm.aggregate("permut-min") == 0
    assert mlcm.aggregate("permut-max") == 2.5 / 4
    assert mlcm.aggregate("mean") == 4 / 12
    # Second test
    mlcm.confusion_matrix = np.array([[0., 0.1, 0.2],
                                      [1, 0.9, 0.8],
                                      [0, 0.2, 0.3]])
    assert mlcm.aggregate("max") == 1.5 / 3
    assert mlcm.aggregate("min") == 0.8 / 3
    assert mlcm.aggregate("permut-min") == 0.9 / 3
    assert mlcm.aggregate("permut-max") == 1.4 / 3
    assert mlcm.aggregate("mean") == 3.5 / 9


def test_multiple_labelings_confusion_matrix_rearrange():
    from clustpy.metrics import unsupervised_clustering_accuracy as acc
    labels_true = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0]]).T
    labels_pred = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 2, 3, 4, 5, 6, 7, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    mlcm = MultipleLabelingsConfusionMatrix(labels_true, labels_pred, metric=acc, remove_noise_spaces=False)
    expected_cm = np.array([[0.5, 0.25, 0.5],
                            [0.5, 0.125, 1],
                            [1, 0.25, 0.5]])
    assert np.allclose(mlcm.confusion_matrix, expected_cm)
    mlcm.rearrange(inplace=True)
    expected_cm = np.array([[0.25, 0.5, 0.5],
                            [0.125, 1, 0.5],
                            [0.25, 0.5, 1]])
    assert np.allclose(mlcm.confusion_matrix, expected_cm)


def test_multiple_labelings_confusion_matrix_average_redundancy():
    from clustpy.metrics import variation_of_information as vi
    labels = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1, 1, 1],
                       [1, 2, 3, 4, 5, 6, 7, 8]]).T
    mlcm = MultipleLabelingsConfusionMatrix(labels, labels, metric=vi)
    expected_cm = np.array([[0, 0, 0.82395922, 1.38629436],
                            [0, 0, 0.82395922, 1.38629436],
                            [0.82395922, 0.82395922, 0, 1.5171064],
                            [1.38629436, 1.38629436, 1.5171064, 0]])
    assert np.allclose(mlcm.confusion_matrix, expected_cm)
    assert np.isclose(mlcm.aggregate("mean_wo_diag"), (0.82395922 * 4 + 1.38629436 * 4 + 1.5171064 * 2) / 12)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_multiple_labelings_confusion_matrix_plot(mock_fig):
    labels_true = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    labels_pred = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 2, 3, 4, 5, 6, 7, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0]]).T
    mlcm = MultipleLabelingsConfusionMatrix(labels_true, labels_pred, remove_noise_spaces=False)
    assert None == mlcm.plot()

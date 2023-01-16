import numpy as np
from cluspy.metrics import ConfusionMatrix
from cluspy.metrics.confusion_matrix import _rearrange
from unittest.mock import patch


def test_rearrange():
    # quadratic confusion matrix
    confusion_matrix = np.array([[0, 45, 3, 2],
                                 [0, 0, 0, 50],
                                 [30, 10, 5, 5],
                                 [5, 5, 35, 5]])
    rearranged_confusion_matrix = _rearrange(confusion_matrix)
    assert np.array_equal(rearranged_confusion_matrix, np.array([[45, 2, 0, 3],
                                                                 [0, 50, 0, 0],
                                                                 [10, 5, 30, 5],
                                                                 [5, 5, 5, 35]]))
    # More prediction labels than ground truth
    confusion_matrix = np.array([[0, 10, 45, 3, 2, 25],
                                 [0, 10, 0, 0, 50, 25],
                                 [30, 10, 10, 5, 5, 25],
                                 [5, 10, 5, 35, 5, 25]])
    rearranged_confusion_matrix = _rearrange(confusion_matrix)
    assert np.array_equal(rearranged_confusion_matrix, np.array([[45, 2, 0, 3, 25, 10],
                                                                 [0, 50, 0, 0, 25, 10],
                                                                 [10, 5, 30, 5, 25, 10],
                                                                 [5, 5, 5, 35, 25, 10]]))
    # More ground truth labels than prediction
    confusion_matrix = np.array([[0, 3, 2],
                                 [0, 0, 50],
                                 [30, 5, 5],
                                 [5, 35, 5]])
    rearranged_confusion_matrix = _rearrange(confusion_matrix)
    assert np.array_equal(rearranged_confusion_matrix, np.array([[0, 2, 0, 3],
                                                                 [0, 50, 0, 0],
                                                                 [0, 5, 30, 5],
                                                                 [0, 5, 5, 35]]))


"""
Tests regarding the ConfusionMatrix object
"""


def test_confusion_matrix_object():
    # First test
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cm = ConfusionMatrix(labels_true, labels_pred)
    expected_cm = np.array([[2, 2, 0, 0],
                            [0, 0, 2, 2]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)
    # Second test
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    expected_cm = np.array([[1, 0, 1, 0],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)


def test_confusion_matrix_rearrange():
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    cm_copy = cm.confusion_matrix.copy()
    rearranged_cm = cm.rearrange(inplace=False)
    assert np.array_equal(cm.confusion_matrix, cm_copy)
    expected_rearranged_cm = np.array([[1, 1, 0, 0],
                                       [1, 1, 0, 0],
                                       [0, 0, 1, 1],
                                       [0, 0, 1, 1]])
    assert np.array_equal(rearranged_cm, expected_rearranged_cm)
    rearranged_cm = cm.rearrange(inplace=True)
    assert np.array_equal(cm.confusion_matrix, rearranged_cm)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_confusion_matrix_plot(mock_fig):
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cm = ConfusionMatrix(labels_true, labels_pred)
    assert None == cm.plot()

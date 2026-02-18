import numpy as np
from clustpy.metrics import ConfusionMatrix
from clustpy.metrics.confusion_matrix import _rearrange
from unittest.mock import patch


def test_rearrange():
    # quadratic confusion matrix
    confusion_matrix = np.array([[0, 45, 3, 2],
                                 [0, 0, 0, 50],
                                 [30, 10, 5, 5],
                                 [5, 5, 35, 5]])
    rearranged_confusion_matrix, rearrange_order = _rearrange(confusion_matrix)
    assert np.array_equal(rearranged_confusion_matrix, np.array([[45, 2, 0, 3],
                                                                 [0, 50, 0, 0],
                                                                 [10, 5, 30, 5],
                                                                 [5, 5, 5, 35]]))
    assert np.array_equal(rearrange_order, np.array([1, 3, 0, 2]))
    # More prediction labels than ground truth
    confusion_matrix = np.array([[0, 10, 45, 3, 2, 25],
                                 [0, 10, 0, 0, 50, 25],
                                 [30, 10, 10, 5, 5, 25],
                                 [5, 10, 5, 35, 5, 25]])
    rearranged_confusion_matrix, rearrange_order = _rearrange(confusion_matrix)
    assert np.array_equal(rearranged_confusion_matrix, np.array([[45, 2, 0, 3, 25, 10],
                                                                 [0, 50, 0, 0, 25, 10],
                                                                 [10, 5, 30, 5, 25, 10],
                                                                 [5, 5, 5, 35, 25, 10]]))
    assert np.array_equal(rearrange_order, np.array([2, 4, 0, 3, 5, 1]))
    # More ground truth labels than prediction
    confusion_matrix = np.array([[0, 3, 2],
                                 [0, 0, 50],
                                 [30, 5, 5],
                                 [5, 35, 5]])
    rearranged_confusion_matrix, rearrange_order = _rearrange(confusion_matrix)
    assert np.array_equal(rearranged_confusion_matrix, np.array([[0, 2, 0, 3],
                                                                 [0, 50, 0, 0],
                                                                 [0, 5, 30, 5],
                                                                 [0, 5, 5, 35]]))
    assert np.array_equal(rearrange_order, np.array([3, 2, 0, 1]))


"""
Tests regarding the ConfusionMatrix object
"""


def test_confusion_matrix_object():
    # First test
    labels_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    expected_cm = np.array([[2, 2, 0, 0],
                            [0, 0, 2, 2]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)
    # Second test
    labels_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    expected_cm = np.array([[1, 0, 1, 0],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)
    # Third test
    labels_true = np.array([0, 1, 2, -1, 0, 1, 2, -1])
    labels_pred = np.array([0, 0, -1, -1, 2, 2, -1, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    expected_cm = np.array([[1, 0, 0, 1],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                            [2, 0, 0, 0]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)



def test_confusion_matrix_object_with_shape():
    # First test
    labels_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cm = ConfusionMatrix(labels_true, labels_pred, "square")
    expected_cm = np.array([[2, 2, 0, 0],
                            [0, 0, 2, 2],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)
    assert np.array_equal(cm.true_clusters, np.array([0,1,-2,-2]))
    assert np.array_equal(cm.pred_clusters, np.array([0,1,2,3]))
    # Second test
    cm = ConfusionMatrix(labels_pred, labels_true, (5, 6))
    expected_cm = np.array([[2, 0, 0, 0, 0, 0],
                            [2, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 0, 0],
                            [0, 2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])
    assert np.array_equal(cm.confusion_matrix, expected_cm)
    assert np.array_equal(cm.true_clusters, np.array([0,1,2,3,-2]))
    assert np.array_equal(cm.pred_clusters, np.array([0,1,-2,-2,-2,-2]))


def test_confusion_matrix_rearrange():
    labels_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    labels_pred = np.array([-1, -1, 1, 1, 2, 2, 3, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    cm_copy = cm.confusion_matrix.copy()
    rearranged_cm = cm.rearrange(inplace=False)
    assert np.array_equal(cm.confusion_matrix, cm_copy)
    assert np.array_equal(cm.true_clusters, np.array([0, 1, 2, 3]))
    assert np.array_equal(cm.pred_clusters, np.array([-1, 1, 2, 3]))
    expected_rearranged_cm = np.array([[1, 1, 0, 0],
                                       [1, 1, 0, 0],
                                       [0, 0, 1, 1],
                                       [0, 0, 1, 1]])
    assert np.array_equal(rearranged_cm, expected_rearranged_cm)
    rearranged_cm = cm.rearrange(inplace=True)
    assert np.array_equal(cm.confusion_matrix, rearranged_cm)
    assert np.array_equal(cm.pred_clusters, np.array([-1, 2, 1, 3]))


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_confusion_matrix_plot(mock_fig):
    labels_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cm = ConfusionMatrix(labels_true, labels_pred)
    assert None == cm.plot()

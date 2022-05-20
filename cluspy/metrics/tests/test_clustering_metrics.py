import numpy as np
from cluspy.metrics.clustering_metrics import unsupervised_clustering_accuracy, variation_of_information, \
    _check_number_of_points
import pytest


def test_check_number_of_points():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    assert _check_number_of_points(l1, l2) == True
    with pytest.raises(Exception):
        _check_number_of_points(l1, l2[1:])


def test_unsupervised_clustering_accuracy():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    assert unsupervised_clustering_accuracy(l1, l2) == 0.9
    l2 = np.array([1, 1, 2, 2, 3, 3, 4, 4, 0, 0])
    assert unsupervised_clustering_accuracy(l1, l2) == 1.0
    l2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert unsupervised_clustering_accuracy(l1, l2) == 0.5
    l2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert unsupervised_clustering_accuracy(l1, l2) == 0.2


def test_variation_of_information():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert variation_of_information(l1, l2) == 0.0

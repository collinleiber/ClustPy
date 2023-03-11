import numpy as np
from clustpy.metrics import unsupervised_clustering_accuracy, variation_of_information, \
    information_theoretic_external_cluster_validity_measure
from clustpy.metrics.clustering_metrics import _check_number_of_points
import pytest


def test_check_number_of_points():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    assert _check_number_of_points(l1, l2) == True
    with pytest.raises(Exception):
        _check_number_of_points(l1, l2[1:])


def test_unsupervised_clustering_accuracy():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([1, 1, 2, 2, 3, 3, 4, 4, 0, 0])
    assert unsupervised_clustering_accuracy(l1, l2) == 1.0
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    assert unsupervised_clustering_accuracy(l1, l2) == 0.9
    l2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert unsupervised_clustering_accuracy(l1, l2) == 0.5
    l2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert unsupervised_clustering_accuracy(l1, l2) == 0.2


def test_variation_of_information():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert variation_of_information(l1, l2) == 0.0
    l1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    l2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert variation_of_information(l1, l2) == 0.0
    l1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    l2 = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    assert np.isclose(variation_of_information(l1, l2), 0.82395922)


def test_information_theoretic_external_cluster_validity_measure():
    # Perfect cluster result
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([1, 1, 2, 2, 3, 3, 4, 4, 0, 0])
    non_scaled_result_1 = information_theoretic_external_cluster_validity_measure(l1, l2, False)
    scaled_result_1 = information_theoretic_external_cluster_validity_measure(l1, l2, True)
    assert scaled_result_1 == 1.0
    # Medium cluster result
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 4])
    non_scaled_result_2 = information_theoretic_external_cluster_validity_measure(l1, l2, False)
    scaled_result_2 = information_theoretic_external_cluster_validity_measure(l1, l2)
    assert scaled_result_2 >= 0 and scaled_result_2 <= 1
    # Poor cluster result
    l2 = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 0])
    non_scaled_result_3 = information_theoretic_external_cluster_validity_measure(l1, l2, False)
    scaled_result_3 = information_theoretic_external_cluster_validity_measure(l1, l2)
    assert scaled_result_3 >= 0 and scaled_result_3 <= 1
    assert non_scaled_result_1 < non_scaled_result_2 and non_scaled_result_2 < non_scaled_result_3
    assert scaled_result_1 > scaled_result_2 and scaled_result_2 > scaled_result_3

import numpy as np
from clustpy.metrics.internal import cvnn_score


def test_cvnn_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    cvnn_1 = cvnn_score(X, L1, n_neighbors=3)
    expected_seperation_1 = 0
    expected_compactness_1 = (1 + 1 + np.sqrt(2) + np.sqrt(2) + 1 + 1) / 6 * 2
    assert cvnn_1 == expected_seperation_1 + expected_compactness_1
    # Check different n_neighbors
    cvnn_2 = cvnn_score(X, L1, n_neighbors=4)
    expected_seperation_2 = 1 / 4
    expected_compactness_2 = (1 + 1 + np.sqrt(2) + np.sqrt(2) + 1 + 1) / 6 * 2
    assert cvnn_2 == expected_seperation_2 + expected_compactness_2
    # Check different metric
    cvnn_3 = cvnn_score(X, L1, n_neighbors=4, metric="sqeuclidean")
    expected_seperation_3 = 1 / 4
    expected_compactness_3 = (1 + 1 + 2 + 2 + 1 + 1) / 6 * 2
    assert cvnn_3 == expected_seperation_3 + expected_compactness_3
    # Check L2
    cvnn_4 = cvnn_score(X, L2, n_neighbors=3)
    expected_seperation_4 = 2 / 3
    expected_compactness_4 = (1 + 1) / 2 + (1 + np.sqrt(41) + np.sqrt(34) + np.sqrt(32) + np.sqrt(25) + np.sqrt(32) + np.sqrt(25) + np.sqrt(25) + np.sqrt(18) + 1 + 1 + np.sqrt(2) + np.sqrt(2) + 1 + 1) / 15
    assert cvnn_4 == expected_seperation_4 + expected_compactness_4
    # Check L3
    cvnn_5 = cvnn_score(X, L3, n_neighbors=3)
    expected_seperation_5 = 2 / 3
    expected_compactness_5 = (1 + np.sqrt(41) + np.sqrt(32) + np.sqrt(34) + np.sqrt(25) + 1) / 6 + (1 + np.sqrt(41) + np.sqrt(34) + np.sqrt(32) + np.sqrt(25) + 1) / 6
    assert cvnn_5 == expected_seperation_5 + expected_compactness_5
    # Check all labels
    cvnn_6 = cvnn_score(X, [L1, L2, L3], n_neighbors=3)
    expected_seperation_6 = np.array([expected_seperation_1, expected_seperation_4, expected_seperation_5])
    expected_compactness_6 = np.array([expected_compactness_1, expected_compactness_4, expected_compactness_5])
    assert np.array_equal(cvnn_6, expected_seperation_6 / expected_seperation_5 + expected_compactness_6 / expected_compactness_5)
    assert cvnn_6[-1] == 2.

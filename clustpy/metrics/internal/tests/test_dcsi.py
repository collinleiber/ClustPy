import numpy as np
from clustpy.metrics.internal import dcsi_score


def test_dcsi_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    dcsi_correct_labels = dcsi_score(X, L1, min_points=1)
    dcsi_wrong_labels1 = dcsi_score(X, L2, min_points=1)
    dcsi_wrong_labels2 = dcsi_score(X, L3, min_points=1)

    assert dcsi_correct_labels > dcsi_wrong_labels1
    assert dcsi_correct_labels > dcsi_wrong_labels2
    assert dcsi_wrong_labels1 > dcsi_wrong_labels2

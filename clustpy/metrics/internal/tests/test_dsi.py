import numpy as np
from clustpy.metrics.internal import dsi_score


def test_dsi_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    dsi_correct_labels = dsi_score(X, L1)
    dsi_wrong_labels1 = dsi_score(X, L2)
    dsi_wrong_labels2 = dsi_score(X, L3)

    assert dsi_correct_labels > dsi_wrong_labels1
    assert dsi_correct_labels > dsi_wrong_labels2
    assert dsi_wrong_labels1 > dsi_wrong_labels2

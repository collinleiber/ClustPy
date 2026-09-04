import numpy as np
from clustpy.metrics.internal import cdbw_score


def test_cdbw_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    cdbw_correct_labels = cdbw_score(X, L1)
    cdbw_wrong_labels1 = cdbw_score(X, L2)
    cdbw_wrong_labels2 = cdbw_score(X, L3)

    assert cdbw_correct_labels > cdbw_wrong_labels1
    assert cdbw_correct_labels > cdbw_wrong_labels2
    assert cdbw_wrong_labels1 > cdbw_wrong_labels2

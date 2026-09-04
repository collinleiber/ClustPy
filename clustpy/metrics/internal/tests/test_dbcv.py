import numpy as np
from clustpy.metrics.internal import dbcv_score


def test_dbcv_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1])

    dbcv_correct_labels = dbcv_score(X, L1)
    dbcv_wrong_labels1 = dbcv_score(X, L2)
    dbcv_wrong_labels2 = dbcv_score(X, L3)

    assert dbcv_correct_labels > dbcv_wrong_labels1
    assert dbcv_correct_labels > dbcv_wrong_labels2
    assert dbcv_wrong_labels1 > dbcv_wrong_labels2

import numpy as np
from clustpy.metrics.internal import cvdd_score


def test_cvdd_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    cvdd_correct_labels = cvdd_score(X, L1)
    cvdd_wrong_labels1 = cvdd_score(X, L2)
    cvdd_wrong_labels2 = cvdd_score(X, L3)

    assert cvdd_correct_labels > cvdd_wrong_labels1
    assert cvdd_correct_labels > cvdd_wrong_labels2
    assert cvdd_wrong_labels1 > cvdd_wrong_labels2

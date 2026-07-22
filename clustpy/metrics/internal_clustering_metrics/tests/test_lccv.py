import numpy as np
from clustpy.metrics import lccv_score


def test_lccv_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 1, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    lccv_correct_labels = lccv_score(X, L1)
    lccv_wrong_labels1 = lccv_score(X, L2)
    lccv_wrong_labels2 = lccv_score(X, L3)

    assert lccv_correct_labels > lccv_wrong_labels1
    assert lccv_correct_labels > lccv_wrong_labels2
    assert lccv_wrong_labels1 > lccv_wrong_labels2

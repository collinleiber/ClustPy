import numpy as np
from clustpy.metrics import s_dbw_score, sd_score


def test_s_dbw_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    s_dbw_correct_labels = s_dbw_score(X, L1)
    s_dbw_wrong_labels1 = s_dbw_score(X, L2)
    s_dbw_wrong_labels2 = s_dbw_score(X, L3)

    assert s_dbw_correct_labels < s_dbw_wrong_labels1
    assert s_dbw_correct_labels < s_dbw_wrong_labels2
    assert s_dbw_wrong_labels1 < s_dbw_wrong_labels2


def test_sd_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    sd_correct_labels = sd_score(X, L1)
    sd_wrong_labels1 = sd_score(X, L2)
    sd_wrong_labels2 = sd_score(X, L3)

    assert sd_correct_labels < sd_wrong_labels1
    assert sd_correct_labels < sd_wrong_labels2
    assert sd_wrong_labels1 < sd_wrong_labels2

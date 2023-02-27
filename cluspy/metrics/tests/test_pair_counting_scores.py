from cluspy.metrics import PairCountingScores, pc_jaccard_score, pc_rand_score, pc_precision_score, pc_recall_score, \
    pc_f1_score
import numpy as np


def testPairCountingScores():
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    # Without removing noise spaces
    pcs = PairCountingScores(labels_true, labels_pred)
    assert pcs.n_tp == 7
    assert pcs.n_fp == 9
    assert pcs.n_fn == 2
    assert pcs.n_tn == 18
    assert pcs.jaccard() == 7 / 18
    assert pcs.jaccard() == pc_jaccard_score(labels_true, labels_pred)
    assert pcs.rand() == 25 / 36
    assert pcs.rand() == pc_rand_score(labels_true, labels_pred)
    assert pcs.precision() == 7 / 16
    assert pcs.precision() == pc_precision_score(labels_true, labels_pred)
    assert pcs.recall() == 7 / 9
    assert pcs.recall() == pc_recall_score(labels_true, labels_pred)
    assert pcs.f1() == 2 * (7 / 16) * (7 / 9) / ((7 / 16) + (7 / 9))
    assert pcs.f1() == pc_f1_score(labels_true, labels_pred)

import numpy as np
from clustpy.metrics.internal import viasckde_score


def test_viasckde_score():
    X = np.array(
        [
            # Cluster 0 (dense, tight)
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.2, 0.0],
            # Cluster 1 (sparser, stretched)
            [3.0, 3.0],
            [3.5, 3.2],
            [4.0, 4.5],
            [2.8, 3.8],
            # Cluster 2 (very sparse / outlier-ish)
            [10.0, 10.0],
            [12.0, 11.0],
        ]
    )
    L1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, -1, -1])

    viasckde_correct_labels = viasckde_score(X, L1)
    viasckde_wrong_labels1 = viasckde_score(X, L2)
    viasckde_wrong_labels2 = viasckde_score(X, L3)

    assert viasckde_correct_labels > viasckde_wrong_labels1
    assert viasckde_correct_labels > viasckde_wrong_labels2
    assert viasckde_wrong_labels1 > viasckde_wrong_labels2

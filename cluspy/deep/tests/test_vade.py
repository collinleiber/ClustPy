from cluspy.data import load_optdigits
from cluspy.deep import VaDE
import numpy as np


def test_simple_vade_with_optdigits():
    X, labels = load_optdigits()
    X = (X - np.mean(X)) / np.std(X)
    vade = VaDE(10, pretrain_epochs=10, clustering_epochs=10, n_gmm_initializations=10)
    assert not hasattr(vade, "labels_")
    vade.fit(X)
    assert vade.labels_.dtype == np.int32
    assert vade.labels_.shape == labels.shape

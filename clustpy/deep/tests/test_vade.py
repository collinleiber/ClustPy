from clustpy.deep.tests._helpers_for_tests import _load_single_label_nrletters
from clustpy.deep import VaDE
import numpy as np


def test_simple_vade_with_nrletters():
    X, labels = _load_single_label_nrletters()
    X = (X - np.mean(X)) / np.std(X)
    vade = VaDE(6, pretrain_epochs=3, clustering_epochs=3, n_gmm_initializations=10)
    assert not hasattr(vade, "labels_")
    vade.fit(X)
    assert vade.labels_.dtype == np.int32
    assert vade.labels_.shape == labels.shape

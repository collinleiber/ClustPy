from clustpy.deep import DEC, IDEC
from clustpy.deep.tests._helpers_for_tests import _load_single_label_nrletters
import numpy as np


def test_simple_dec_with_nrletters():
    X, labels = _load_single_label_nrletters()
    dec = DEC(6, pretrain_epochs=3, clustering_epochs=3)
    assert not hasattr(dec, "labels_")
    dec.fit(X)
    assert dec.labels_.dtype == np.int32
    assert dec.labels_.shape == labels.shape


def test_simple_idec_with_nrletters():
    X, labels = _load_single_label_nrletters()
    idec = IDEC(6, pretrain_epochs=3, clustering_epochs=3)
    assert not hasattr(idec, "labels_")
    idec.fit(X)
    assert idec.labels_.dtype == np.int32
    assert idec.labels_.shape == labels.shape

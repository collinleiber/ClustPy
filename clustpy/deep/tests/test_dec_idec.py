from clustpy.deep import DEC, IDEC
from clustpy.deep.tests._helpers_for_tests import _load_single_label_nrletters
import numpy as np


def test_simple_dec_with_nrletters():
    X, labels = _load_single_label_nrletters()
    dec = DEC(6, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(dec, "labels_")
    dec.fit(X)
    assert dec.labels_.dtype == np.int32
    assert dec.labels_.shape == labels.shape
    # Test if random state is working
    dec2 = DEC(6, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    dec2.fit(X)
    assert np.array_equal(dec.labels_, dec2.labels_)
    assert np.array_equal(dec.cluster_centers_, dec2.cluster_centers_)
    assert np.array_equal(dec.dec_labels_, dec2.dec_labels_)
    assert np.array_equal(dec.dec_cluster_centers_, dec2.dec_cluster_centers_)


def test_simple_idec_with_nrletters():
    X, labels = _load_single_label_nrletters()
    idec = IDEC(6, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(idec, "labels_")
    idec.fit(X)
    assert idec.labels_.dtype == np.int32
    assert idec.labels_.shape == labels.shape
    # Test if random state is working
    idec2 = DEC(6, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    idec2.fit(X)
    assert np.array_equal(idec.labels_, idec2.labels_)
    assert np.array_equal(idec.cluster_centers_, idec2.cluster_centers_)
    assert np.array_equal(idec.dec_labels_, idec2.dec_labels_)
    assert np.array_equal(idec.dec_cluster_centers_, idec2.dec_cluster_centers_)

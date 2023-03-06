from clustpy.deep.tests._helpers_for_tests import _load_single_label_nrletters
from clustpy.deep import VaDE
import numpy as np


def test_simple_vade_with_nrletters():
    X, labels = _load_single_label_nrletters()
    X = (X - np.mean(X)) / np.std(X)
    vade = VaDE(6, pretrain_epochs=3, clustering_epochs=3, n_gmm_initializations=1, random_state=1)
    assert not hasattr(vade, "labels_")
    vade.fit(X)
    assert vade.labels_.dtype == np.int32
    assert vade.labels_.shape == labels.shape
    # Test if random state is working
    vade2 = VaDE(6, pretrain_epochs=3, clustering_epochs=3, n_gmm_initializations=1, random_state=1)
    vade2.fit(X)
    assert np.array_equal(vade.labels_, vade2.labels_)
    assert np.array_equal(vade.cluster_centers_, vade2.cluster_centers_)
    assert np.array_equal(vade.vade_labels_, vade2.vade_labels_)
    assert np.array_equal(vade.vade_cluster_centers_, vade2.vade_cluster_centers_)

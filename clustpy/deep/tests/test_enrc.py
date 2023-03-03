from clustpy.data import load_nrletters
from clustpy.deep import ENRC
import numpy as np


def test_simple_enrc_with_nrletters():
    X, labels = load_nrletters()
    enrc = ENRC([6, 4, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(enrc, "labels_")
    enrc.fit(X)
    assert enrc.labels_.dtype == np.int32
    assert enrc.labels_.shape == labels.shape
    # Test if random state is working
    enrc2 = ENRC([6, 4, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1)
    enrc2.fit(X)
    assert np.array_equal(enrc.labels_, enrc2.labels_)
    assert np.array_equal(enrc.cluster_centers_, enrc2.cluster_centers_)

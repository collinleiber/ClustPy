from clustpy.data import load_nrletters
from clustpy.deep import ENRC
import numpy as np


def test_simple_enrc_with_nrletters():
    X, labels = load_nrletters()
    enrc = ENRC([6, 4, 3], pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(enrc, "labels_")
    enrc.fit(X)
    assert enrc.labels_.dtype == np.int32
    assert enrc.labels_.shape == labels.shape

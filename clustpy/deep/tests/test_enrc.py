from clustpy.deep import ENRC
from clustpy.data import create_nr_data
import numpy as np
import torch


def test_simple_enrc():
    torch.use_deterministic_algorithms(True)
    X, labels = create_nr_data(1500, subspace_features=(3, 3, 50), random_state=1)
    labels = labels[:, :-1] # ignore noise space
    enrc = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(enrc, "labels_")
    enrc.fit(X)
    assert enrc.labels_.dtype == np.int32
    assert enrc.labels_.shape == labels.shape
    # Test if random state is working
    enrc2 = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1)
    enrc2.fit(X)
    assert np.array_equal(enrc.labels_, enrc2.labels_)
    for i in range(len(enrc.cluster_centers_)):
        assert np.allclose(enrc.cluster_centers_[i], enrc2.cluster_centers_[i])

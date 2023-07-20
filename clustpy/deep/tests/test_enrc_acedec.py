from clustpy.deep import ENRC, ACeDeC
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
    enrc2 = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1, debug=True)
    enrc2.fit(X)
    assert np.array_equal(enrc.labels_, enrc2.labels_)
    for i in range(len(enrc.cluster_centers_)):
        assert np.allclose(enrc.cluster_centers_[i], enrc2.cluster_centers_[i])
    # Test if sgd as init is working
    enrc = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, init="sgd")
    enrc.fit(X)
    assert enrc.labels_.dtype == np.int32
    assert enrc.labels_.shape == labels.shapes

def test_simple_acedec():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    acedec = ACeDeC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(acedec, "labels_")
    acedec.fit(X)
    assert acedec.labels_.dtype == np.int32
    assert acedec.labels_.shape == labels.shape
    # Test if random state is working
    acedec2 = ACeDeC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    acedec2.fit(X)
    assert np.array_equal(acedec.labels_, acedec2.labels_)
    assert np.allclose(acedec.cluster_centers_, acedec2.cluster_centers_, atol=1e-1)

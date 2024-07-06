from clustpy.deep import DDC, N2D
from clustpy.data import create_subspace_data
import torch
import numpy as np
from sklearn.manifold import Isomap


def test_simple_ddc():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    ddc = DDC(pretrain_epochs=3, random_state=1)
    assert not hasattr(ddc, "labels_")
    ddc.fit(X)
    assert ddc.labels_.dtype == np.int32
    assert ddc.labels_.shape == labels.shape
    # Test if random state is working
    ddc2 = DDC(pretrain_epochs=3, random_state=1)
    ddc2.fit(X)
    assert np.array_equal(ddc.labels_, ddc2.labels_)


def test_simple_n2d():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    n2d = N2D(n_clusters=3, pretrain_epochs=3, random_state=1)
    assert not hasattr(n2d, "labels_")
    n2d.fit(X)
    assert n2d.labels_.dtype == np.int32
    assert n2d.labels_.shape == labels.shape
    # Test if random state is working
    n2d2 = N2D(n_clusters=3, pretrain_epochs=3, random_state=1)
    n2d2.fit(X)
    assert np.array_equal(n2d.labels_, n2d2.labels_)
    # Check different manifold
    n2d = N2D(n_clusters=3, pretrain_epochs=3, manifold_class=Isomap, manifold_params={"n_components":2, "n_neighbors":20}, random_state=1)
    n2d.fit(X)
    assert n2d.labels_.dtype == np.int32
    assert n2d.labels_.shape == labels.shape

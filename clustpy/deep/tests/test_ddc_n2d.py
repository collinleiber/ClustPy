from clustpy.deep import DDC, N2D
from clustpy.deep.ddc_n2d import DDC_density_peak_clustering
from clustpy.data import create_subspace_data
from sklearn.datasets import make_blobs
import torch
import numpy as np
from sklearn.manifold import Isomap
from sklearn.utils.estimator_checks import check_estimator


def test_ddc_estimator():
    check_estimator(DDC(pretrain_epochs=3), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_n2d_estimator():
    check_estimator(N2D(3, pretrain_epochs=3), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_ddc_density_peak_clustering():
    X, labels = make_blobs(50, 2, centers=3, random_state=1)
    # With small and large ratio
    for ratio in [0.1, 999]:
        ddc_dpc = DDC_density_peak_clustering(ratio=ratio)
        assert not hasattr(ddc_dpc, "labels_")
        ddc_dpc.fit(X)
        assert ddc_dpc.labels_.dtype == np.int32
        assert ddc_dpc.labels_.shape == labels.shape
        assert len(np.unique(ddc_dpc.labels_)) == ddc_dpc.n_clusters_
        assert np.array_equal(np.unique(ddc_dpc.labels_), np.arange(ddc_dpc.n_clusters_))


def test_simple_ddc():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    ddc = DDC(pretrain_epochs=3, random_state=1)
    assert not hasattr(ddc, "labels_")
    ddc.fit(X)
    assert ddc.labels_.dtype == np.int32
    assert ddc.labels_.shape == labels.shape
    # Test if random state is working
    ddc2 = DDC(pretrain_epochs=3, random_state=1, ratio=999)
    ddc2.ratio = 0.1
    ddc2.fit(X)
    assert np.array_equal(ddc.labels_, ddc2.labels_)


def test_simple_n2d():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
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
    n2d = N2D(n_clusters=3, pretrain_epochs=3, manifold_class=Isomap,
              manifold_params={"n_components": 2, "n_neighbors": 20}, random_state=1)
    n2d.fit(X)
    assert n2d.labels_.dtype == np.int32
    assert n2d.labels_.shape == labels.shape

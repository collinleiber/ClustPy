from clustpy.deep import DKM
from clustpy.deep.dkm import _get_default_alphas
from clustpy.data import create_subspace_data, load_optdigits
from clustpy.deep.tests._helpers_for_tests import _get_test_augmentation_dataloaders
import torch
import numpy as np


def test_simple_dkm():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    dkm = DKM(3, pretrain_epochs=3, alphas=(None, 0.1, 1), clustering_epochs=3, random_state=1)
    assert not hasattr(dkm, "labels_")
    dkm.fit(X)
    assert dkm.labels_.dtype == np.int32
    assert dkm.labels_.shape == labels.shape
    # Test if random state is working
    dkm = DKM(3, pretrain_epochs=3, alphas=0.1, clustering_epochs=3, random_state=1)
    dkm.fit(X)
    assert np.array_equal(dkm.labels_, dkm.labels_)
    assert np.allclose(dkm.cluster_centers_, dkm.cluster_centers_, atol=1e-1)
    assert np.array_equal(dkm.dkm_labels_, dkm.dkm_labels_)
    assert np.allclose(dkm.dkm_cluster_centers_, dkm.dkm_cluster_centers_, atol=1e-1)
    # Test predict
    labels_predict = dkm.predict(X)
    assert np.array_equal(dkm.labels_, labels_predict)


def test_get_default_alphas():
    obtained_alphas = _get_default_alphas(init_alpha=0.1, n_alphas=5)
    expected_alphas = [0.1, 0.42320861065570825, 0.7515684111296623, 1.077971160195895, 1.4087110115785935]
    assert np.allclose(obtained_alphas, expected_alphas)


def test_dkm_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(data)
    clusterer = DKM(10, pretrain_epochs=3, alphas=(None, 0.1, 2), clustering_epochs=3, random_state=1,
                    custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(data)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape

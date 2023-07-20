from clustpy.deep import DCN
from clustpy.deep import DEC, IDEC
from clustpy.deep import ACeDeC
from clustpy.data import load_optdigits
from clustpy.deep.tests._helpers_for_tests import _get_test_augmentation_dataloaders
import torch
import numpy as np
import torchvision


def test_dcn_aug():
    torch.use_deterministic_algorithms(True)
    data, labels = load_optdigits()
    X = torch.from_numpy(data[0:100]).reshape(-1, 1, 8, 8)
    X = torchvision.transforms.Resize((32,32))(X)
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(X)
    clusterer = DCN(10, pretrain_epochs=3, clustering_epochs=3, random_state=1,
              custom_dataloaders=[aug_dl, orig_dl], 
              augmentation_invariance=True,
             )
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(X)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape
    # Test if random state is working
    clusterer2 = DCN(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                     custom_dataloaders=[aug_dl, orig_dl], 
                     augmentation_invariance=True,
    )
    clusterer2.fit(X)
    assert np.array_equal(clusterer.labels_, clusterer2.labels_)
    assert np.allclose(clusterer.cluster_centers_, clusterer2.cluster_centers_, atol=1e-1)
    assert np.array_equal(clusterer.dcn_labels_, clusterer2.dcn_labels_)
    assert np.allclose(clusterer.dcn_cluster_centers_, clusterer2.dcn_cluster_centers_, atol=1e-1)

def test_dec_aug():
    torch.use_deterministic_algorithms(True)
    data, labels = load_optdigits()
    X = torch.from_numpy(data[0:100]).reshape(-1, 1, 8, 8)
    X = torchvision.transforms.Resize((32,32))(X)
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(X)

    dec = DEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
              custom_dataloaders=[aug_dl, orig_dl], 
              augmentation_invariance=True,
    )
    assert not hasattr(dec, "labels_")
    dec.fit(X)
    assert dec.labels_.dtype == np.int32
    assert dec.labels_.shape == labels.shape
    # Test if random state is working
    dec2 = DEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                custom_dataloaders=[aug_dl, orig_dl], 
                augmentation_invariance=True,
    )
    dec2.fit(X)
    assert np.array_equal(dec.labels_, dec2.labels_)
    assert np.allclose(dec.cluster_centers_, dec2.cluster_centers_, atol=1e-1)
    assert np.array_equal(dec.dec_labels_, dec2.dec_labels_)
    assert np.allclose(dec.dec_cluster_centers_, dec2.dec_cluster_centers_, atol=1e-1)


def test_idec_aug():
    torch.use_deterministic_algorithms(True)
    data, labels = load_optdigits()
    X = torch.from_numpy(data[0:100]).reshape(-1, 1, 8, 8)
    X = torchvision.transforms.Resize((32,32))(X)
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(X)
    
    idec = IDEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                custom_dataloaders=[aug_dl, orig_dl], 
                augmentation_invariance=True,
                )
    assert not hasattr(idec, "labels_")
    idec.fit(X)
    assert idec.labels_.dtype == np.int32
    assert idec.labels_.shape == labels.shape
    # Test if random state is working
    idec2 = IDEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                 custom_dataloaders=[aug_dl, orig_dl], 
                 augmentation_invariance=True,
    )
    idec2.fit(X)
    assert np.array_equal(idec.labels_, idec2.labels_)
    assert np.allclose(idec.cluster_centers_, idec2.cluster_centers_, atol=5e-1)
    assert np.array_equal(idec.dec_labels_, idec2.dec_labels_)
    assert np.allclose(idec.dec_cluster_centers_, idec2.dec_cluster_centers_, atol=5e-1)

def test_acedec_aug():
    torch.use_deterministic_algorithms(True)
    data, labels = load_optdigits()
    X = torch.from_numpy(data[0:100]).reshape(-1, 1, 8, 8)
    X = torchvision.transforms.Resize((32,32))(X)
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(X)
    
    acedec = ACeDeC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                custom_dataloaders=[aug_dl, orig_dl], 
                augmentation_invariance=True,
                )
    assert not hasattr(acedec, "labels_")
    acedec.fit(X)
    assert acedec.labels_.dtype == np.int32
    assert acedec.labels_.shape == labels.shape
    # Test if random state is working
    acedec2 = acedec(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                 custom_dataloaders=[aug_dl, orig_dl], 
                 augmentation_invariance=True,
    )
    acedec2.fit(X)
    assert np.array_equal(acedec.labels_, acedec2.labels_)
    assert np.allclose(acedec.cluster_centers_, acedec2.cluster_centers_, atol=5e-1)
from clustpy.deep import AEC, get_default_augmented_dataloaders
from clustpy.data import create_subspace_data, load_optdigits
import torch
import numpy as np


def test_simple_aec():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    aec = AEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(aec, "labels_")
    aec.fit(X)
    assert aec.labels_.dtype == np.int32
    assert aec.labels_.shape == labels.shape
    # Test if random state is working
    aec2 = AEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    aec2.fit(X)
    assert np.array_equal(aec.labels_, aec2.labels_)
    assert np.allclose(aec.cluster_centers_, aec2.cluster_centers_, atol=1e-1)
    # Test predict
    labels_predict = aec.predict(X)
    assert np.array_equal(aec.labels_, labels_predict)


def test_aec_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = get_default_augmented_dataloaders(data)
    clusterer = AEC(10, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                    custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(data)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape

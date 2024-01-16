from clustpy.deep import DDC
from clustpy.data import create_subspace_data, load_optdigits
from clustpy.deep.tests._helpers_for_tests import _get_test_augmentation_dataloaders
import torch
import numpy as np


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


def test_ddc_augmentation():
    torch.use_deterministic_algorithms(True)
    data, labels = load_optdigits(flatten=False)
    data = data[:1000]
    labels = labels[:1000]
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(data)
    clusterer = DDC(pretrain_epochs=3, random_state=1,
                    custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(data)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape

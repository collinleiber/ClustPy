from clustpy.deep import DCN
from clustpy.deep.dcn import _compute_centroids
from clustpy.data import create_subspace_data, load_optdigits
from clustpy.deep.tests._helpers_for_tests import _get_test_augmentation_dataloaders
import torch
import numpy as np


def test_simple_dcn():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    dcn = DCN(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(dcn, "labels_")
    dcn.fit(X)
    assert dcn.labels_.dtype == np.int32
    assert dcn.labels_.shape == labels.shape
    # Test if random state is working
    dcn2 = DCN(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    dcn2.fit(X)
    assert np.array_equal(dcn.labels_, dcn2.labels_)
    assert np.allclose(dcn.cluster_centers_, dcn2.cluster_centers_, atol=1e-1)
    assert np.array_equal(dcn.dcn_labels_, dcn2.dcn_labels_)
    assert np.allclose(dcn.dcn_cluster_centers_, dcn2.dcn_cluster_centers_, atol=1e-1)
    # Test predict
    labels_predict = dcn.predict(X)
    assert np.array_equal(dcn.labels_, labels_predict)


def test_compute_centroids():
    embedded = torch.tensor([[0., 1., 1.], [1., 0., 1.], [2., 2., 1.], [1., 2., 2.], [3., 4., 5.]])
    centers = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    count = torch.tensor([1, 3, 1])
    labels = torch.tensor([0, 0, 1, 1, 2])
    new_centers, new_count = _compute_centroids(centers, embedded, count, labels)
    assert torch.equal(new_count, torch.tensor([3, 5, 2]))
    desired_centers = torch.tensor([[2 / 3 * 0.5 + 1 / 3 * 1., 2 / 3 * 1. + 1 / 3 * 0., 2 / 3 * 1. + 1 / 3 * 1.],
                                    [4 / 5 * 2. + 1 / 5 * 1., 4 / 5 * 2. + 1 / 5 * 2., 4 / 5 * 1.75 + 1 / 5 * 2.],
                                    [0.5 * 3. + 0.5 * 3., 0.5 * 3. + 0.5 * 4., 0.5 * 3. + 0.5 * 5.]])
    assert torch.all(torch.isclose(new_centers, desired_centers))  # torch.equal is not working due to numerical issues


def test_dcn_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = _get_test_augmentation_dataloaders(data)
    clusterer = DCN(10, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                    custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(data)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape

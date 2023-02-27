from clustpy.data import load_optdigits
from clustpy.deep import DCN
from clustpy.deep.dcn import _compute_centroids
import torch
import numpy as np


def test_simple_dcn_with_optdigits():
    X, labels = load_optdigits()
    dcn = DCN(10, pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(dcn, "labels_")
    dcn.fit(X)
    assert dcn.labels_.dtype == np.int32
    assert dcn.labels_.shape == labels.shape


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

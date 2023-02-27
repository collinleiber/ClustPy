from clustpy.data import load_optdigits
from clustpy.deep._utils import squared_euclidean_distance, detect_device, encode_batchwise, predict_batchwise, window, \
    int_to_one_hot
from clustpy.deep.tests._helpers_for_tests import _get_test_dataloader, _TestAutoencoder, _TestClusterModule
import torch
import numpy as np


def test_squared_euclidean_distance():
    tensor1 = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    tensor2 = torch.tensor([[1, 1, 1], [3, 4, 5], [5, 5, 5]])
    dist_tensor = squared_euclidean_distance(tensor1, tensor2)
    desired = torch.tensor([[0, 4 + 9 + 16, 16 * 3],
                            [1 * 3, 1 + 4 + 9, 9 * 3],
                            [4 * 3, 0 + 1 + 4, 4 * 3],
                            [9 * 3, 1 + 0 + 1, 1 * 3]])
    assert torch.equal(dist_tensor, desired)
    weights = torch.tensor([0.1, 0.2, 0.3])
    dist_tensor = squared_euclidean_distance(tensor1, tensor2, weights)
    desired = torch.tensor([[0, 0.01 * 4 + 0.04 * 9 + 0.09 * 16, 0.01 * 16 + 0.04 * 16 + 0.09 * 16],
                            [0.01 + 0.04 + 0.09, 0.01 * 1 + 0.04 * 4 + 0.09 * 9, 0.01 * 9 + 0.04 * 9 + 0.09 * 9],
                            [0.01 * 4 + 0.04 * 4 + 0.09 * 4, 0 + 0.04 * 1 + 0.09 * 4, 0.01 * 4 + 0.04 * 4 + 0.09 * 4],
                            [0.01 * 9 + 0.04 * 9 + 0.09 * 9, 0.01 * 1 + 0 + 0.09 * 1, 0.01 + 0.04 + 0.09]])
    assert torch.all(torch.isclose(dist_tensor, desired))  # torch.equal is not working due to numerical issues


def test_detect_device():
    # TODO idea for better test
    device = detect_device()
    assert type(device) is torch.device
    assert device.type == "cpu" or device.type == "cuda"


def test_encode_batchwise():
    # Load dataset
    data, _ = load_optdigits()
    embedding_size = 5
    device = torch.device('cpu')
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    encoded = encode_batchwise(dataloader, autoencoder, device)
    # Each embedded feature should match the sum of the original features
    desired = np.sum(data, axis=1).reshape((-1, 1))
    desired = np.tile(desired, embedding_size)
    assert np.array_equal(encoded, desired)


def test_predict_batchwise():
    # Load dataset
    data, _ = load_optdigits()
    threshold = np.mean(np.sum(data, axis=1))
    embedding_size = 5
    device = torch.device('cpu')
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    cluster_module = _TestClusterModule(threshold)
    predictions = predict_batchwise(dataloader, autoencoder, cluster_module, device)
    # Check whether sum of the features (= embedded samples) is larger than the threshold
    desired = (np.sum(data, axis=1) >= threshold) * 1
    assert np.array_equal(predictions, desired)


def test_window():
    pass  # TODO


def test_int_to_one_hot():
    labels = torch.tensor([0, 0, 1, 2, 1])
    desired = torch.tensor([[1., 0., 0.],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
    onehot = int_to_one_hot(labels, 3)
    assert torch.equal(onehot, desired)

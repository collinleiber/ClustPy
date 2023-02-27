from cluspy.data import load_optdigits
from cluspy.deep import get_dataloader, NeighborEncoder
from cluspy.deep.neighbor_encoder import get_neighbors_batchwise
from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np


def test_simple_neighbor_encoder_with_optdigits():
    data, _ = load_optdigits()
    embedding_dim = 10
    n_neighbors = 3
    dist_matrix = squareform(pdist(data))
    neighbor_ids = np.argsort(dist_matrix, axis=1)
    neighbors = [data[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    dataloader = get_dataloader(data, 256, True, additional_inputs=neighbors)
    # Test fitting
    neighborencoder = NeighborEncoder(layers=[data.shape[1], 128, 64, embedding_dim], n_neighbors=n_neighbors,
                                      decode_self=False)
    assert neighborencoder.fitted is False
    neighborencoder.fit(n_epochs=5, lr=1e-3, dataloader=dataloader)
    assert neighborencoder.fitted is True
    # Test encoding
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedded = neighborencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = neighborencoder.decode(embedded)
    assert decoded.shape == (n_neighbors, batch_size, data.shape[1])
    # Test forwarding
    forwarded = neighborencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)

    # Test fitting with self decoding
    n_neighbors = 2
    neighborencoder_2 = NeighborEncoder(layers=[data.shape[1], 128, 64, embedding_dim], n_neighbors=n_neighbors,
                                        decode_self=True)
    assert neighborencoder_2.fitted is False
    neighborencoder_2.fit(n_epochs=5, lr=1e-3, dataloader=dataloader)
    assert neighborencoder_2.fitted is True
    # Test encoding
    forwarded = neighborencoder_2.forward(data_batch)
    assert forwarded.shape == (n_neighbors + 1, batch_size, data.shape[1])


def test_get_neighbors_batchwise():
    X = np.array([[1, 0], [2, 1], [3, 1], [6, 0], [10, 11], [10, 10], [9, 12]])
    n_neighbors = 2
    neighbors = get_neighbors_batchwise(X, n_neighbors)
    result = [np.array([[2, 1], [3, 1], [2, 1], [3, 1], [10, 10], [10, 11], [10, 11]]),
              np.array([[3, 1], [1, 0], [1, 0], [2, 1], [9, 12], [9, 12], [10, 10]])]
    for i in range(len(result)):
        assert np.array_equal(result[i], neighbors[i])
    # Check if it also works with a smaller batch size
    neighbors = get_neighbors_batchwise(X, n_neighbors, batch_size=2)
    for i in range(len(result)):
        assert np.array_equal(result[i], neighbors[i])

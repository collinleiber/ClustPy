from cluspy.data import load_optdigits
from cluspy.deep import NeighborEncoder
from cluspy.deep import get_dataloader
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
    neighborencoder_2 = NeighborEncoder(layers=[data.shape[1], 128, 64, embedding_dim], n_neighbors=n_neighbors,
                                    decode_self=True)
    assert neighborencoder_2.fitted is False
    neighborencoder_2.fit(n_epochs=5, lr=1e-3, dataloader=dataloader)
    assert neighborencoder_2.fitted is True
    # Test encoding
    forwarded = neighborencoder_2.forward(data_batch)
    assert forwarded.shape == (n_neighbors + 1, batch_size, data.shape[1])

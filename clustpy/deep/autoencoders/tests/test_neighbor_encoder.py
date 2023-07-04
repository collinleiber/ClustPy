from clustpy.deep.autoencoders import NeighborEncoder
from clustpy.deep import get_dataloader, DCN
from clustpy.deep.autoencoders.neighbor_encoder import get_neighbors_batchwise
from clustpy.data import create_subspace_data
from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np


def test_neighbor_encoder():
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    n_neighbors = 3
    # Get dataloader with neighbors
    dist_matrix = squareform(pdist(data))
    neighbor_ids = np.argsort(dist_matrix, axis=1)
    neighbors = [data[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    dataloader = get_dataloader(data, 256, True, additional_inputs=neighbors)
    neighborencoder = NeighborEncoder(layers=[data.shape[1], 128, 64, embedding_dim], n_neighbors=n_neighbors,
                                      decode_self=False)
    # Test encoding
    embedded = neighborencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = neighborencoder.decode(embedded)
    assert decoded.shape == (n_neighbors, batch_size, data.shape[1])
    # Test forwarding
    forwarded = neighborencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)
    # Test loss
    device = torch.device('cpu')
    loss_fn = torch.nn.MSELoss()
    first_batch = next(iter(dataloader))
    loss, embedded, decoded = neighborencoder.loss(first_batch, loss_fn, device)
    assert loss.item() >= 0
    # Test fitting (without self decoding)
    assert neighborencoder.fitted is False
    neighborencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, dataloader=dataloader)
    assert neighborencoder.fitted is True
    # Test fitting with self decoding
    n_neighbors = 2
    neighborencoder_2 = NeighborEncoder(layers=[data.shape[1], 128, 64, embedding_dim], n_neighbors=n_neighbors,
                                        decode_self=True)
    assert neighborencoder_2.fitted is False
    neighborencoder_2.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, dataloader=dataloader)
    assert neighborencoder_2.fitted is True
    # Test encoding
    forwarded = neighborencoder_2.forward(data_batch)
    assert forwarded.shape == (n_neighbors + 1, batch_size, data.shape[1])


def test_neighbor_encoder_in_deep_clustering():
    data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    embedding_dim = 10
    n_neighbors = 3
    # Get dataloader with neighbors
    dist_matrix = squareform(pdist(data))
    neighbor_ids = np.argsort(dist_matrix, axis=1)
    neighbors = [data[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    trainloader = get_dataloader(data, 256, True, additional_inputs=neighbors)
    testloader = get_dataloader(data, 256, False, additional_inputs=neighbors)
    custom_dataloaders = (trainloader, testloader)
    # Test combining the NeighborEncoder with DCN
    neighborencoder = NeighborEncoder(layers=[data.shape[1], 128, 64, embedding_dim], n_neighbors=n_neighbors,
                                      decode_self=False)
    dcn = DCN(3, pretrain_epochs=3, clustering_epochs=3, autoencoder=neighborencoder,
              custom_dataloaders=custom_dataloaders, random_state=1)
    dcn.fit(data)
    assert dcn.labels_.dtype == np.int32
    assert dcn.labels_.shape == labels.shape


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

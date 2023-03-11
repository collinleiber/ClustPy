from clustpy.deep import FlexibleAutoencoder
from clustpy.data import create_subspace_data
import torch


def test_simple_flexible_autoencoder():
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    embedding_dim = 10
    # Test fitting
    autoencoder = FlexibleAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=5, lr=1e-3, data=data)
    assert autoencoder.fitted is True
    # Test encoding
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, data.shape[1])
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)
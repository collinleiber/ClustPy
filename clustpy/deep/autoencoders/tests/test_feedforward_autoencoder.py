from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.data import create_subspace_data
import torch


def test_feedforward_autoencoder():
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    autoencoder = FeedforwardAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, data.shape[1])
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)
    # Test fitting
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True

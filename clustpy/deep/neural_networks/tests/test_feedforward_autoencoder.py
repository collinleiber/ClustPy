from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.data import create_subspace_data
import torch
import pytest
import os


@pytest.fixture
def cleanup_autoencoder():
    yield
    filename = "autoencoder.ae"
    if os.path.isfile(filename):
        os.remove(filename)


def test_feedforward_autoencoder():
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
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


@pytest.mark.usefixtures("cleanup_autoencoder")
def test_feedforward_autoencoder_load():
    data, _ = create_subspace_data(100, subspace_features=(3, 50), random_state=1)
    embedding_dim = 10
    autoencoder = FeedforwardAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    autoencoder.save_parameters("autoencoder.ae")
    # Create second AE
    autoencoder2 = FeedforwardAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
    autoencoder2.load_parameters("autoencoder.ae")
    # Test embedding
    data_torch = torch.Tensor(data)
    embedding1 = autoencoder.encode(data_torch)
    decoded1 = autoencoder.decode(embedding1)
    embedding2 = autoencoder2.encode(data_torch)
    decoded2 = autoencoder2.decode(embedding2)
    assert torch.equal(embedding1, embedding2)
    assert torch.equal(decoded1, decoded2)

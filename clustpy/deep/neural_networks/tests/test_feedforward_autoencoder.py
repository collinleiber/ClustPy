from clustpy.deep.neural_networks import FeedforwardAutoencoder
import torch
import pytest
import os
from clustpy.deep.tests._helpers_for_tests import _get_dc_test_data


@pytest.fixture
def cleanup_autoencoder():
    yield
    filename = "autoencoder.ae"
    if os.path.isfile(filename):
        os.remove(filename)


def test_feedforward_autoencoder():
    data, _ = _get_dc_test_data()
    batch_size = 30
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 4
    autoencoder = FeedforwardAutoencoder(layers=[data.shape[1], 10, embedding_dim])
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    embedded_solo = autoencoder.encode(data_batch[0])
    assert embedded_solo.shape == (embedding_dim, )
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, data.shape[1])
    decoded_solo = autoencoder.decode(embedded[0])
    assert decoded_solo.shape == (data.shape[1], )
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)
    # Test fitting
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True


@pytest.mark.usefixtures("cleanup_autoencoder")
def test_feedforward_autoencoder_load():
    data, _ = _get_dc_test_data()
    embedding_dim = 4
    autoencoder = FeedforwardAutoencoder(layers=[data.shape[1], 10, embedding_dim])
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    autoencoder.save_parameters("autoencoder.ae")
    # Create second AE
    autoencoder2 = FeedforwardAutoencoder(layers=[data.shape[1], 10, embedding_dim])
    autoencoder2.load_parameters("autoencoder.ae")
    # Test embedding
    data_torch = torch.Tensor(data)
    embedding1 = autoencoder.encode(data_torch)
    decoded1 = autoencoder.decode(embedding1)
    embedding2 = autoencoder2.encode(data_torch)
    decoded2 = autoencoder2.decode(embedding2)
    assert torch.equal(embedding1, embedding2)
    assert torch.equal(decoded1, decoded2)

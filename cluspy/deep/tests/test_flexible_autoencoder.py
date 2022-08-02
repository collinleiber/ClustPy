from cluspy.data import load_optdigits
from cluspy.deep import FlexibleAutoencoder
import torch


def test_simple_flexible_autoencoder_with_optdigits():
    data, _ = load_optdigits()
    embedding_dim = 10
    # Test fitting
    autoencoder = FlexibleAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
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
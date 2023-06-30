from clustpy.deep.autoencoders import ConvolutionalAutoencoder
from clustpy.deep._utils import set_torch_seed
import torch
import numpy as np

def test_convolutional_autoencoder():
    set_torch_seed(np.random.RandomState(1))
    data = torch.rand((512, 3, 32, 32))
    embedding_dim = 10
    # Test fitting
    autoencoder = ConvolutionalAutoencoder(32, [512, embedding_dim])
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=1, lr=1e-3, data=data)
    assert autoencoder.fitted is True
    # Test encoding
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)
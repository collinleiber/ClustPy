from clustpy.data import load_optdigits
from clustpy.deep import VariationalAutoencoder
from clustpy.deep.variational_autoencoder import _vae_sampling
import torch


def test_simple_variational_autoencoder_with_optdigits():
    data, _ = load_optdigits()
    embedding_dim = 10
    # Test fitting
    autoencoder = VariationalAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=5, lr=1e-3, data=data)
    assert autoencoder.fitted is True
    # Test encoding
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedded_mean, embedded_var = autoencoder.encode(data_batch)
    assert embedded_mean.shape == (batch_size, embedding_dim)
    assert embedded_var.shape == (batch_size, embedding_dim)
    # Test decoding
    torch.manual_seed(0)
    embedded_sample = _vae_sampling(embedded_mean, embedded_var)
    decoded = autoencoder.decode(embedded_sample)
    assert decoded.shape == (batch_size, data.shape[1])
    # Test forwarding (needs seed, since sampling is random)
    torch.manual_seed(0)
    forward_sample, forward_mean, forward_var, forwarded_reconstruct = autoencoder.forward(data_batch)
    assert torch.equal(forward_sample, embedded_sample)
    assert torch.equal(forward_mean, embedded_mean)
    assert torch.equal(forward_var, embedded_var)
    assert torch.equal(forwarded_reconstruct, decoded)

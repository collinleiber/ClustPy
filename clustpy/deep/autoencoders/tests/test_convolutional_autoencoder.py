from clustpy.deep.autoencoders import ConvolutionalAutoencoder
from clustpy.deep import DCN
import torch
import numpy as np


def test_convolutional_autoencoder_resnet18():
    data = torch.rand((512, 3, 32, 32))
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    autoencoder = ConvolutionalAutoencoder(32, [512, embedding_dim])
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)
    # Test fitting
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=2, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True


def test_convolutional_autoencoder_resnet_50():
    data = torch.rand((512, 3, 32, 32))
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    autoencoder = ConvolutionalAutoencoder(32, [2048, embedding_dim], conv_encoder_name="resnet50")
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)


def test_mixed_convolutional_autoencoder():
    data = torch.rand((512, 3, 32, 32))
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    autoencoder = ConvolutionalAutoencoder(32, [2048, embedding_dim], fc_decoder_layers=[embedding_dim, 512],
                                           conv_encoder_name="resnet50", conv_decoder_name="resnet18")
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)


def test_convolutional_autoencoder_in_deep_clustering():
    data = np.random.random((512, 3, 32, 32))
    embedding_dim = 10
    # Test combining the Convolutional Autoencoder with DCN
    autoencoder = ConvolutionalAutoencoder(32, [512, embedding_dim])
    dcn = DCN(3, pretrain_epochs=1, clustering_epochs=3, autoencoder=autoencoder, random_state=1)
    dcn.fit(data)
    assert dcn.labels_.dtype == np.int32
    assert dcn.labels_.shape == (512,)

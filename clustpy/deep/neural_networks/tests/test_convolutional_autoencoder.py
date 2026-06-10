from clustpy.deep.neural_networks import ConvolutionalAutoencoder
from clustpy.deep import DCN
import torch
import numpy as np
import pytest


def test_convolutional_autoencoder_resnet18():
    data = torch.rand((512, 3, 32, 32))
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    autoencoder = ConvolutionalAutoencoder(32, [512, embedding_dim])
    autoencoder.eval()
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    embedded_solo = autoencoder.encode(data_batch[0])
    assert embedded_solo.shape == (embedding_dim, )
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    decoded_solo = autoencoder.decode(embedded[0])
    assert decoded_solo.shape == (3, 32, 32)
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
    autoencoder.eval()
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    embedded_solo = autoencoder.encode(data_batch[0])
    assert embedded_solo.shape == (embedding_dim, )
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    decoded_solo = autoencoder.decode(embedded[0])
    assert decoded_solo.shape == (3, 32, 32)
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
    autoencoder.eval()
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert embedded.shape == (batch_size, embedding_dim)
    embedded_solo = autoencoder.encode(data_batch[0])
    assert embedded_solo.shape == (embedding_dim, )
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert decoded.shape == (batch_size, 3, 32, 32)
    decoded_solo = autoencoder.decode(embedded[0])
    assert decoded_solo.shape == (3, 32, 32)
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(decoded, forwarded)


def test_convolutional_autoencoder_in_deep_clustering():
    data = np.random.random((100, 3, 32, 32))
    embedding_dim = 4
    # Test combining the Convolutional Autoencoder with DCN
    autoencoder = ConvolutionalAutoencoder(32, [512, embedding_dim])
    dcn = DCN(3, pretrain_epochs=1, clustering_epochs=3, neural_network=autoencoder, random_state=1, embedding_size=embedding_dim)
    dcn.fit(data)
    assert dcn.labels_.dtype == np.int32
    assert dcn.labels_.shape == (100,)
    X_embed = dcn.transform(data)
    assert X_embed.shape == (data.shape[0], dcn.embedding_size)


def test_convolutional_autoencoder_errors():
    with pytest.raises(ValueError):
        # Wrong input height (must be 32 x X)
        ConvolutionalAutoencoder(16, [512, 10])
    with pytest.raises(ValueError):
        # Wrong fc_layers for resnet 18
        ConvolutionalAutoencoder(32, conv_encoder_name="resnet18", fc_layers=[2048, 10])
    with pytest.raises(ValueError):
        # Wrong fc_layers for resnet 50
        ConvolutionalAutoencoder(32, conv_encoder_name="resnet50", fc_layers=[512, 10])

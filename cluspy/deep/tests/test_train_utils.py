from cluspy.data import load_optdigits
from cluspy.deep import FlexibleAutoencoder, VariationalAutoencoder
from cluspy.deep._train_utils import get_trained_autoencoder, get_default_layers
from cluspy.deep.tests._test_helpers import _get_test_dataloader
import numpy as np
import torch


def test_get_default_layers():
    input_dim = 64
    embedding_dim = 5
    layers = get_default_layers(input_dim, embedding_dim)
    assert np.array_equal(layers, [input_dim, 500, 500, 2000, embedding_dim])
    input_dim = 256
    embedding_dim = 10
    layers = get_default_layers(input_dim, embedding_dim)
    assert np.array_equal(layers, [input_dim, 500, 500, 2000, embedding_dim])


def test_get_trained_autoencoder():
    # Load dataset
    data, _ = load_optdigits()
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get AE using the default AE class
    device = torch.device('cpu')
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=data.shape[1], embedding_size=10)
    # Check output of get_trained_autoencoder
    assert type(ae) is FlexibleAutoencoder
    assert ae.fitted == True


def test_get_trained_autoencoder_with_custom_ae_class():
    # Load dataset
    data, _ = load_optdigits()
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get AE using a custom AE class
    device = torch.device('cpu')
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=data.shape[1], embedding_size=10,
                                 autoencoder_class=VariationalAutoencoder)
    # Check output of get_trained_autoencoder
    assert type(ae) is VariationalAutoencoder
    assert ae.fitted == True


def test_get_trained_autoencoder_with_custom_ae():
    # Load dataset
    data, _ = load_optdigits()
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get trained version of custom AE
    device = torch.device('cpu')
    ae = FlexibleAutoencoder(layers=[data.shape[1], 256, 128, 64, 10])
    assert ae.fitted == False
    encoder_0_params = ae.encoder.block[0].weight.data.detach().clone()
    decoder_0_params = ae.decoder.block[0].weight.data.detach().clone()
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=data.shape[1], embedding_size=10,
                                 autoencoder=ae)
    # Check output of get_trained_autoencoder
    assert type(ae) is FlexibleAutoencoder
    assert ae.fitted == True
    assert not torch.equal(encoder_0_params, ae.encoder.block[0].weight.data)
    assert not torch.equal(decoder_0_params, ae.decoder.block[0].weight.data)


def test_get_trained_autoencoder_with_custom_pretrained_ae():
    # Load dataset
    data, _ = load_optdigits()
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get same pretrained version out of get_trained_autoencoder
    device = torch.device('cpu')
    ae = FlexibleAutoencoder(layers=[data.shape[1], 256, 128, 64, 10])
    assert ae.fitted == False
    ae.fit(dataloader=dataloader, lr=1e-3, n_epochs=5, device=device,
           optimizer_class=torch.optim.Adam,
           loss_fn=torch.nn.MSELoss())
    encoder_0_params = ae.encoder.block[0].weight.data.detach().clone()
    decoder_0_params = ae.decoder.block[0].weight.data.detach().clone()
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=data.shape[1], embedding_size=10,
                                 autoencoder=ae)
    # Check output of get_trained_autoencoder
    assert type(ae) is FlexibleAutoencoder
    assert ae.fitted == True
    assert torch.equal(encoder_0_params, ae.encoder.block[0].weight.data)
    assert torch.equal(decoder_0_params, ae.decoder.block[0].weight.data)

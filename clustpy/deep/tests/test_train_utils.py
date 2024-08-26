from clustpy.deep.neural_networks import FeedforwardAutoencoder, VariationalAutoencoder
from clustpy.deep._train_utils import get_trained_network, _get_default_layers
from clustpy.deep.tests._helpers_for_tests import _get_test_dataloader
from clustpy.data import create_subspace_data
import numpy as np
import torch
import os
import pytest


def test_get_default_layers():
    input_dim = 64
    embedding_dim = 5
    layers = _get_default_layers(input_dim, embedding_dim)
    assert np.array_equal(layers, [input_dim, 500, 500, 2000, embedding_dim])
    input_dim = 256
    embedding_dim = 10
    layers = _get_default_layers(input_dim, embedding_dim)
    assert np.array_equal(layers, [input_dim, 500, 500, 2000, embedding_dim])


def test_get_trained_network():
    # Load dataset
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get AE using the default AE class
    device = torch.device('cpu')
    ae = get_trained_network(trainloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, device=device,
                             optimizer_class=torch.optim.Adam, ssl_loss_fn=torch.nn.MSELoss(), embedding_size=10)
    # Check output of get_trained_network
    assert type(ae) is FeedforwardAutoencoder
    assert ae.fitted == True


def test_get_trained_network_with_custom_ae_class():
    # Load dataset
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get AE using a custom AE class
    device = torch.device('cpu')
    ae = get_trained_network(trainloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, device=device,
                             optimizer_class=torch.optim.Adam, ssl_loss_fn=torch.nn.MSELoss(), embedding_size=60,
                             neural_network_class=VariationalAutoencoder,
                             neural_network_params={"layers": [53, 25, 10]})
    # Check output of get_trained_network
    assert type(ae) is VariationalAutoencoder
    assert ae.fitted == True


def test_get_trained_network_with_custom_ae():
    # Load dataset
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get trained version of custom AE
    device = torch.device('cpu')
    ae = FeedforwardAutoencoder(layers=[data.shape[1], 256, 128, 64, 10], work_on_copy=False)
    assert ae.fitted == False
    encoder_0_params = ae.encoder.block[0].weight.data.detach().clone()
    decoder_0_params = ae.decoder.block[0].weight.data.detach().clone()
    ae_out = get_trained_network(trainloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam, ssl_loss_fn=torch.nn.MSELoss(), embedding_size=10,
                                 neural_network=ae)
    # Check output of get_trained_network
    assert type(ae_out) is FeedforwardAutoencoder
    assert ae_out.fitted == True
    assert not torch.equal(encoder_0_params, ae_out.encoder.block[0].weight.data)
    assert not torch.equal(decoder_0_params, ae_out.decoder.block[0].weight.data)
    assert ae is ae_out


def test_get_trained_network_with_custom_pretrained_ae():
    # Load dataset
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get same pretrained version out of get_trained_network
    device = torch.device('cpu')
    ae = FeedforwardAutoencoder(layers=[data.shape[1], 256, 128, 64, 10], work_on_copy=True)
    assert ae.fitted == False
    ae.fit(dataloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, optimizer_class=torch.optim.Adam,
           ssl_loss_fn=torch.nn.MSELoss())
    encoder_0_params = ae.encoder.block[0].weight.data.detach().clone()
    decoder_0_params = ae.decoder.block[0].weight.data.detach().clone()
    ae_out = get_trained_network(trainloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam, ssl_loss_fn=torch.nn.MSELoss(), embedding_size=10,
                                 neural_network=ae)
    # Check output of get_trained_network
    assert type(ae_out) is FeedforwardAutoencoder
    assert ae_out.fitted == True
    assert torch.equal(encoder_0_params, ae_out.encoder.block[0].weight.data)
    assert torch.equal(decoder_0_params, ae_out.decoder.block[0].weight.data)
    assert ae is not ae_out


@pytest.fixture
def cleanup_autoencoders():
    yield
    filename = "autoencoder.ae"
    if os.path.isfile(filename):
        os.remove(filename)


@pytest.mark.usefixtures("cleanup_autoencoders")
def test_get_trained_network_using_saved_ae():
    path = "autoencoder.ae"
    layers = [53, 4, 2]
    # Load dataset
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dataloader = _get_test_dataloader(data, 256, True, False)
    # Get pretrained ae
    device = torch.device('cpu')
    ae = FeedforwardAutoencoder(layers=layers, work_on_copy=True)
    assert ae.fitted == False
    ae.fit(dataloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, optimizer_class=torch.optim.Adam,
           ssl_loss_fn=torch.nn.MSELoss(), model_path=path)
    assert ae.fitted is True
    ae2 = get_trained_network(trainloader=dataloader, optimizer_params={"lr": 1e-3}, n_epochs=5, device=device,
                              optimizer_class=torch.optim.Adam, ssl_loss_fn=torch.nn.MSELoss(), embedding_size=10,
                              neural_network=(FeedforwardAutoencoder, {"layers": layers}),
                              neural_network_weights=path)
    assert ae2.fitted is True
    # Check if all parameters are equal
    for p1, p2 in zip(ae.parameters(), ae2.parameters()):
        assert torch.equal(p1.data, p2.data)

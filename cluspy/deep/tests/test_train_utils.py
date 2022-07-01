from cluspy.deep import FlexibleAutoencoder
from cluspy.deep._train_utils import get_trained_autoencoder, get_default_layers
import numpy as np
import torch
from cluspy.deep import VariationalAutoencoder


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
    # Create dataset
    N = 1000
    d = 50
    data = np.random.random_sample((N, d))
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.arange(0, N), torch.from_numpy(data).float())),
        batch_size=256,
        shuffle=True,
        drop_last=False)
    # Get AE using the default AE class
    device = torch.device('cpu')
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=d, embedding_size=10)
    # Check output of get_trained_autoencoder
    assert type(ae) is FlexibleAutoencoder
    assert ae.fitted == True


def test_get_trained_autoencoder_with_custom_ae_class():
    # Create dataset
    N = 1000
    d = 50
    data = np.random.random_sample((N, d))
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.arange(0, N), torch.from_numpy(data).float())),
        batch_size=256,
        shuffle=True,
        drop_last=False)
    # Get AE using a custom AE class
    device = torch.device('cpu')
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=d, embedding_size=10,
                                 autoencoder_class=VariationalAutoencoder)
    # Check output of get_trained_autoencoder
    assert type(ae) is VariationalAutoencoder
    assert ae.fitted == True


def test_get_trained_autoencoder_with_custom_ae():
    # Create dataset
    N = 1000
    d = 50
    data = np.random.random_sample((N, d))
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.arange(0, N), torch.from_numpy(data).float())),
        batch_size=256,
        shuffle=True,
        drop_last=False)
    # Get trained version of custom AE
    device = torch.device('cpu')
    ae = FlexibleAutoencoder(layers=[50, 256, 128, 64, 10])
    assert ae.fitted == False
    encoder_0_params = ae.encoder.block[0].weight.data.detach().clone()
    decoder_0_params = ae.decoder.block[0].weight.data.detach().clone()
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=5, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=d, embedding_size=10,
                                 autoencoder=ae)
    # Check output of get_trained_autoencoder
    assert type(ae) is FlexibleAutoencoder
    assert ae.fitted == True
    assert not torch.equal(encoder_0_params, ae.encoder.block[0].weight.data)
    assert not torch.equal(decoder_0_params, ae.decoder.block[0].weight.data)


def test_get_trained_autoencoder_with_custom_pretrained_ae():
    # Create dataset
    N = 1000
    d = 50
    data = np.random.random_sample((N, d))
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.arange(0, N), torch.from_numpy(data).float())),
        batch_size=256,
        shuffle=True,
        drop_last=False)
    # Get same pretrained version out of get_trained_autoencoder
    device = torch.device('cpu')
    ae = FlexibleAutoencoder(layers=[50, 256, 128, 64, 10])
    assert ae.fitted == False
    ae.fit(dataloader=dataloader, lr=1e-3, n_epochs=5, device=device,
           optimizer_class=torch.optim.Adam,
           loss_fn=torch.nn.MSELoss())
    encoder_0_params = ae.encoder.block[0].weight.data.detach().clone()
    decoder_0_params = ae.decoder.block[0].weight.data.detach().clone()
    ae = get_trained_autoencoder(trainloader=dataloader, learning_rate=1e-3, n_epochs=10, device=device,
                                 optimizer_class=torch.optim.Adam,
                                 loss_fn=torch.nn.MSELoss(), input_dim=d, embedding_size=10,
                                 autoencoder=ae)
    # Check output of get_trained_autoencoder
    assert type(ae) is FlexibleAutoencoder
    assert ae.fitted == True
    assert torch.equal(encoder_0_params, ae.encoder.block[0].weight.data)
    assert torch.equal(decoder_0_params, ae.decoder.block[0].weight.data)

from clustpy.deep.neural_networks import StackedAutoencoder
from clustpy.data import create_subspace_data
import torch


def test_stacked_autoencoder():
    data, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 10
    autoencoder = StackedAutoencoder(layers=[data.shape[1], 128, 64, embedding_dim])
    # Test fitting
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs_per_layer=3, n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True

from clustpy.deep import DipEncoder, get_dataloader, detect_device, get_default_augmented_dataloaders
from clustpy.deep.dipencoder import plot_dipencoder_embedding, _get_ssl_loss_of_first_batch
from clustpy.data import create_subspace_data, load_optdigits
from clustpy.deep.neural_networks import FeedforwardAutoencoder, ConvolutionalAutoencoder
import numpy as np
import torch
from unittest.mock import patch
from sklearn.utils.estimator_checks import check_estimator


def test_dipencoder_estimator():
    check_estimator(DipEncoder(3, pretrain_epochs=3, clustering_epochs=3), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_simple_dipencoder():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dipencoder = DipEncoder(3, pretrain_epochs=3, clustering_epochs=3, max_cluster_size_diff_factor=2.2, random_state=1)
    assert not hasattr(dipencoder, "labels_")
    dipencoder.fit(X)
    assert dipencoder.labels_.dtype == np.int32
    assert dipencoder.labels_.shape == labels.shape
    # Test if random state is working
    # TODO Does not work every time -> Check why
    # dipencoder2 = DipEncoder(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    # dipencoder2.fit(X)
    # assert np.array_equal(dipencoder.labels_, dipencoder2.labels_)
    # assert np.allclose(dipencoder.projection_axes_, dipencoder2.projection_axes_, atol=1e-1)
    # assert dipencoder.index_dict_ == dipencoder2.index_dict_
    # Test predict
    labels_predict = dipencoder.predict(X)
    assert np.array_equal(dipencoder.labels_, labels_predict)


def test_supervised_dipencoder():
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dipencoder = DipEncoder(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(dipencoder, "labels_")
    dipencoder.fit(X, labels)
    assert dipencoder.labels_.dtype == np.int32
    assert np.array_equal(labels, dipencoder.labels_)


def test_dipencoder_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = get_default_augmented_dataloaders(data)
    clusterer = DipEncoder(10, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                           custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(data)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape


def test_plot_dipencoder_embedding():
    embedded_data = np.array(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
         [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17],
         [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21],
         [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37],
         [38, 38, 38], [39, 39, 39]])
    n_clusters = 3
    cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    projection_axes = np.array([[11, 11, 11], [30, 30, 30], [19, 19, 19]])
    index_dict = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    assert None == plot_dipencoder_embedding(embedded_data, n_clusters, cluster_labels, projection_axes, index_dict,
                                             show_plot=False)


def test_get_rec_loss_of_first_batch():
    torch.use_deterministic_algorithms(True)
    X = torch.rand((512, 3, 32, 32))
    device = detect_device()
    # Test with FeedforwardAutoencoder
    X_flat = X.reshape(512, -1)
    ff_trainloader = get_dataloader(X_flat, 32, shuffle=True)
    ff_autoencoder = FeedforwardAutoencoder([X_flat.shape[1], 32, 10])
    ae_loss = _get_ssl_loss_of_first_batch(ff_trainloader, ff_autoencoder, torch.nn.MSELoss(), device)
    assert ae_loss > 0
    # Test with ConvolutionalAutoencoder
    conv_trainloader = get_dataloader(X, 32, shuffle=True)
    conv_autoencoder = ConvolutionalAutoencoder(X.shape[-1], [512, 10])
    ae_loss = _get_ssl_loss_of_first_batch(conv_trainloader, conv_autoencoder, torch.nn.MSELoss(), device)
    assert ae_loss > 0


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_dipencoder_obj(mock_fig):
    X, _ = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dipencoder = DipEncoder(3, pretrain_epochs=1, clustering_epochs=1, random_state=1)
    dipencoder.fit(X)
    assert None == dipencoder.plot(X)

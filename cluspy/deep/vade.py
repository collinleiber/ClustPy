"""
Jiang, Zhuxi, et al. "Variational deep embedding: An
unsupervised and generative approach to clustering." arXiv
preprint arXiv:1611.05148 (2016).

@authors: Donatella Novakovic
"""

import torch
from cluspy.deep._utils import detect_device, encode_batchwise
from cluspy.deep._data_utils import get_dataloader
from cluspy.deep._train_utils import get_trained_autoencoder
from cluspy.deep.variational_autoencoder import VariationalAutoencoder
from cluspy.deep.flexible_autoencoder import FlexibleAutoencoder, FullyConnectedBlock
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClusterMixin


def _vade(X, n_clusters, batch_size, pretrain_learning_rate, clustering_learning_rate, pretrain_epochs,
          clustering_epochs, optimizer_class, loss_fn, autoencoder, embedding_size, n_gmm_initializations):
    device = detect_device()
    trainloader = get_dataloader(X, batch_size, True, False)
    testloader = get_dataloader(X, batch_size, False, False)
    if autoencoder is None:
        autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                              optimizer_class, loss_fn, X.shape[1], embedding_size, autoencoder,
                                              _VaDE_AE)
        # Execute EM in embedded space
        embedded_data = encode_batchwise(testloader, autoencoder, device)
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=n_gmm_initializations)
        gmm.fit(embedded_data)
        # Initialize VaDE
        vae = VariationalAutoencoder(layers=autoencoder.layers, n_distributions=n_clusters,
                                             batch_norm=autoencoder.batch_norm,
                                             dropout=autoencoder.dropout, activation_fn=autoencoder.activation_fn,
                                             bias=autoencoder.bias, decoder_layers=autoencoder.decoder_layers,
                                             decoder_output_fn=autoencoder.decoder_output_fn)
        # Copy data from _VadeAE to VAE
        vae.pi.data = torch.from_numpy(gmm.weights_).float()
        vae.p_mean.data = torch.from_numpy(gmm.means_).float()
        vae.p_var.data = torch.log(torch.from_numpy(gmm.covariances_)).float()
        vae.q_mean_ = autoencoder.q_mean_
        vae.encoder = autoencoder.encoder
        vae.decoder = autoencoder.decoder
        vae.to(device)
    else:
        vae = autoencoder
        vae.to(device)
    # Vade Training loop
    vae.fit(n_epochs=clustering_epochs, lr=clustering_learning_rate, batch_size=batch_size,
                    dataloader=trainloader,
                    optimizer_class=optimizer_class, loss_fn=loss_fn, device=device)
    # Get labels
    vade_labels = _vade_predict_batchwise(testloader, vae, device)
    vade_centers = vae.p_mean.detach().cpu().numpy()
    vade_covariances = vae.p_var.detach().cpu().numpy()
    # Do reclustering with GMM
    embedded_data = _vade_encode_batchwise(testloader, vae, device)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=100)
    gmm_labels = gmm.fit_predict(embedded_data)
    # Return results
    return gmm_labels, gmm.means_, gmm.covariances_, vade_labels, vade_centers, vade_covariances, vae


def _vade_predict_batchwise(dataloader, vade_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        q_mean, q_var = vade_module.encode(batch_data)
        prediction = vade_module.predict(q_mean, q_var).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()


def _vade_encode_batchwise(dataloader, vade_module, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        q_mean, _ = vade_module.encode(batch_data)
        embeddings.append(q_mean.detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


class _VaDE_AE(FlexibleAutoencoder):
    """
    A special autoencoder used for VaDE.

    Parameters
    ----------
    layers : list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the dimension of the embedding.
             If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    batch_norm : bool, default=False, set True if you want to use torch.nn.BatchNorm1d
    dropout : float, default=None, set the amount of dropout you want to use.
    activation: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers, if None then it will be linear.
    bias : bool, default=True, set False if you do not want to use a bias term in the linear layers
    decoder_layers : list, default=None, list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers.
    decoder_output_fn : activation function from torch.nn, default=None, set the activation function for the decoder output layer, if None then it will be linear.
                        e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1.
    Attributes
    ----------
    encoder : encoder part of the autoencoder, responsible for embedding data points
    decoder : decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    fitted  : boolean value indicating whether the autoencoder is already fitted.
    """

    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU, bias=True,
                 decoder_layers=None, decoder_output_fn=torch.nn.Sigmoid):
        super(_VaDE_AE, self).__init__(layers, batch_norm, dropout, activation_fn, bias,
                                       decoder_layers, decoder_output_fn)
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bias = bias
        self.decoder_output_fn = decoder_output_fn
        self.fitted = False
        if decoder_layers is None:
            self.decoder_layers = self.layers[::-1]
        else:
            self.decoder_layers = decoder_layers
        if (self.layers[-1] != self.decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {self.layers[-1]} and {self.decoder_layers[0]} respectively.")
        if (self.layers[0] != self.decoder_layers[-1]):
            raise ValueError(
                f"Output and input dimension do not match, they are {self.layers[0]} and {self.decoder_layers[-1]} respectively.")

        # Initialize encoder
        self.encoder = FullyConnectedBlock(layers=self.layers[:-1], batch_norm=self.batch_norm, dropout=self.dropout,
                                           activation_fn=self.activation_fn, bias=self.bias,
                                           output_fn=self.activation_fn)
        # naming is used for later correspondence in VAE
        self.q_mean_ = torch.nn.Linear(self.layers[-2], self.layers[-1])
        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(layers=self.decoder_layers, batch_norm=self.batch_norm, dropout=self.dropout,
                                           activation_fn=self.activation_fn, bias=self.bias,
                                           output_fn=self.decoder_output_fn)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        x : input data point, can also be a mini-batch of points

        Returns
        -------
        q_mean : mean value of the central VAE layer
        q_var : variance value of the central VAE layer
        """
        embedded = self.encoder(x)
        q_mean = self.q_mean_(embedded)
        return q_mean


class VaDE(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, batch_size=256, pretrain_learning_rate=1e-3, clustering_learning_rate=1e-4,
                 pretrain_epochs=100, clustering_epochs=150, optimizer_class=torch.optim.Adam,
                 loss_fn=torch.nn.BCELoss(reduction='sum'), autoencoder=None, embedding_size=10,
                 n_gmm_initializations=100):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.n_gmm_initializations = n_gmm_initializations

    def fit(self, X, y=None):
        gmm_labels, gmm_means, gmm_covariances, vade_labels, vade_centers, vade_covariances, autoencoder = _vade(X,
                                                                                                                 self.n_clusters,
                                                                                                                 self.batch_size,
                                                                                                                 self.pretrain_learning_rate,
                                                                                                                 self.clustering_learning_rate,
                                                                                                                 self.pretrain_epochs,
                                                                                                                 self.clustering_epochs,
                                                                                                                 self.optimizer_class,
                                                                                                                 self.loss_fn,
                                                                                                                 self.autoencoder,
                                                                                                                 self.embedding_size,
                                                                                                                 self.n_gmm_initializations)
        self.labels_ = gmm_labels
        self.cluster_centers_ = gmm_means
        self.covariances_ = gmm_covariances
        self.vade_labels_ = vade_labels
        self.vade_cluster_centers_ = vade_centers
        self.vade_covariances_ = vade_covariances
        self.autoencoder = autoencoder
        return self

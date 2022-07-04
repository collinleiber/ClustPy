"""
@authors:
Donatella Novakovic
Lukas Miklautz
Collin Leiber
"""

import torch
from cluspy.deep._utils import detect_device
from cluspy.deep._data_utils import get_dataloader
from cluspy.deep._train_utils import get_trained_autoencoder
from cluspy.deep.variational_autoencoder import VariationalAutoencoder, _vae_sampling
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClusterMixin


def _vade(X, n_clusters, batch_size, pretrain_learning_rate, clustering_learning_rate, pretrain_epochs,
          clustering_epochs, optimizer_class, loss_fn, autoencoder, embedding_size, n_gmm_initializations):
    """
    Start the actual VaDE clustering procedure on the input data set.
    First an variational autoencoder (VAE) will be trained (will be skipped if input autoencoder is given).
    Afterwards, a GMM will be fit to identify the initial clustering structures.
    Last, the VAE will be optimized using the VaDE loss function from the _VaDE_Module.

    Parameters
    ----------
    X : the given data set
    n_clusters : int, number of clusters
    batch_size : int, size of the data batches
    pretrain_learning_rate : double, learning rate for the pretraining of the autoencoder
    clustering_learning_rate : double, learning rate of the actual clustering procedure
    pretrain_epochs : int, number of epochs for the pretraining of the autoencoder
    clustering_epochs : int, number of epochs for the actual clustering procedure
    optimizer_class : Optimizer
    loss_fn : loss function for the reconstruction
    autoencoder : the input autoencoder. If None a variation of a VariationalAutoencoder will be created
    embedding_size : int, size of the embedding within the autoencoder (central layer with mean and variance)
    n_gmm_initializations : int, number of initializations for the initial GMM clustering within the embedding

    Returns
    -------
    gmm_labels
    gmm.means_
    gmm.covariances_
    vade_labels
    vade_centers
    vade_covariances
    autoencoder
    """
    device = detect_device()
    trainloader = get_dataloader(X, batch_size, True, False)
    testloader = get_dataloader(X, batch_size, False, False)
    autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, X.shape[1], embedding_size, autoencoder,
                                          _VaDE_VAE)
    # Execute EM in embedded space
    embedded_data = _vade_encode_batchwise(testloader, autoencoder, device)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=n_gmm_initializations)
    gmm.fit(embedded_data)
    # Initialize VaDE
    vade_module = _VaDE_Module(n_clusters=n_clusters, embedding_size=embedding_size, weights=gmm.weights_,
                               means=gmm.means_, variances=gmm.covariances_).to(device)
    # Use vade learning_rate (usually pretrain_learning_rate reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(vade_module.parameters()),
                                lr=clustering_learning_rate)
    # Vade Training loop
    vade_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn)
    # Get labels
    vade_labels = _vade_predict_batchwise(testloader, autoencoder, vade_module, device)
    vade_centers = vade_module.p_mean.detach().cpu().numpy()
    vade_covariances = vade_module.p_var.detach().cpu().numpy()
    # Do reclustering with GMM
    embedded_data = _vade_encode_batchwise(testloader, autoencoder, device)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=100)
    gmm_labels = gmm.fit_predict(embedded_data)
    # Return results
    return gmm_labels, gmm.means_, gmm.covariances_, vade_labels, vade_centers, vade_covariances, autoencoder


class _VaDE_VAE(VariationalAutoencoder):
    """
    A special variational autoencoder used for VaDE.
    Has a slightly different forward function while pretraining the autoencoder.
    Further, Loss function is more similar to the FlexibleAutoencoder while pretraining.

    Attributes
    ----------
    encoder : encoder part of the autoencoder, responsible for embedding data points (class is FullyConnectedBlock)
    decoder : decoder part of the autoencoder, responsible for reconstructing data points from the embedding (class is FullyConnectedBlock)
    mean : mean value of the central layer
    log_variance : logarithmic variance of the central layer (use logarithm of variance - numerical purposes)
    fitted  : boolean value indicating whether the autoencoder is already fitted.
    """

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Applies both encode and decode function.
        The forward function is automatically called if we call self(x).
        Matches forward behavior from FlexibleAutoencoder for pretraining and from VariationalAutoencoder afterwards.
        Overwrites function from VariationalAutoencoder.

        Parameters
        ----------
        x : input data point, can also be a mini-batch of embedded points

        Returns
        -------
        reconstruction: returns the reconstruction of the data point
        """
        if not self.fitted:
            # While pretraining a forward method similar to a regular autoencoder (FlexibleAutoencoder) should be used
            mean, _ = self.encode(x)
            reconstruction = self.decode(mean)
            z, q_mean, q_logvar = None, None, None
        else:
            # After pretraining the usual forward of a VAE should be used. Super() uses function from VariationalAutoencoder
            z, q_mean, q_logvar, reconstruction = super().forward(x)
        return z, q_mean, q_logvar, reconstruction

    def loss(self, batch_data, loss_fn, beta=1):
        """
        Calculate the loss of a single batch of data.
        Matches loss calculation from FlexibleAutoencoder for pretraining and from VariationalAutoencoder afterwards.
        Overwrites function from VariationalAutoencoder.

        Parameters
        ----------
        batch_data : torch.Tensor, the samples
        loss_fn : torch.nn, loss function to be used for reconstruction
        beta : Not used at the moment

        Returns
        -------
        loss: returns the reconstruction loss of the input sample
        """
        if not self.fitted:
            # While pretraining a loss similar to a regular autoencoder (FlexibleAutoencoder) should be used
            _, _, _, reconstruction = self.forward(batch_data)
            loss = loss_fn(reconstruction, batch_data)
        else:
            # After pretraining the usual loss of a VAE should be used. Super() uses function from VariationalAutoencoder
            loss = super().loss(batch_data, loss_fn, beta)
        return loss


def _vade_predict_batchwise(dataloader, autoencoder, vade_module, device):
    """
    Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader, dataloader to be used
    autoencoder : the VariationalAutoencoder
    vade_module : the _VaDE_Module
    device : torch.device, device on which the prediction should take place

    Returns
    -------
    The cluster labels of the data set
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        q_mean, q_logvar = autoencoder.encode(batch_data)
        prediction = vade_module.predict(q_mean, q_logvar).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()


def _vade_encode_batchwise(dataloader, autoencoder, device):
    """
    Utility function for embedding the whole data set in a mini-batch fashion

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader, dataloader to be used
    autoencoder : the VariationalAutoencoder
    device : torch.device, device on which the embedding should take place

    Returns
    -------
    The encoded version of the data set
    """
    embeddings = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        q_mean, _ = autoencoder.encode(batch_data)
        embeddings.append(q_mean.detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def _get_gamma(pi, p_mean, p_var, z):
    """
    Calculate the gamma of samples created by the VAE.

    Parameters
    ----------
    pi : softmax version of the soft cluster assignments in the _VaDE_Module
    p_mean : cluster centers of the _VaDE_Module
    p_var : variances of the _VaDE_Module
    z : the created samples

    Returns
    -------
    The gamma values
    """
    z = z.unsqueeze(1)
    p_var = p_var.unsqueeze(0)
    pi = pi.unsqueeze(0)

    p_z_c = -torch.sum(0.5 * (np.log(2 * np.pi)) + p_var + ((z - p_mean).pow(2) / (2. * torch.exp(p_var))), dim=2)
    p_c_z_c = torch.exp(torch.log(pi) + p_z_c) + 1e-10
    p_c_z = p_c_z_c / torch.sum(p_c_z_c, dim=1, keepdim=True)

    return p_c_z


def _compute_vade_loss(pi, p_mean, p_var, q_mean, q_var, batch_data, p_c_z, reconstruction, loss_fn):
    """
    Calculate the final loss of the input samples for the VaDE algorithm.

    Parameters
    ----------
    pi : softmax version of the soft cluster assignments in the _VaDE_Module
    p_mean : cluster centers of the _VaDE_Module
    p_var : variances of the _VaDE_Module
    q_mean : mean value of the central layer of the VAE
    q_var : logarithmic variance of the central layer of the VAE
    batch_data : torch.Tensor, the samples
    p_c_z : result of the _get_gamma function
    reconstruction : the reconstructed version of the input samples
    loss_fn : torch.nn, loss function to be used for reconstruction

    Returns
    -------
    Tha VaDE loss
    """
    q_mean = q_mean.unsqueeze(1)
    p_var = p_var.unsqueeze(0)

    p_x_z = loss_fn(reconstruction, batch_data)

    p_z_c = torch.sum(p_c_z * (0.5 * np.log(2 * np.pi) + 0.5 * (
            torch.sum(p_var, dim=2) + torch.sum(torch.exp(q_var.unsqueeze(1)) / torch.exp(p_var),
                                                dim=2) + torch.sum((q_mean - p_mean).pow(2) / torch.exp(p_var),
                                                                   dim=2))))
    p_c = torch.sum(p_c_z * torch.log(pi))
    q_z_x = 0.5 * (np.log(2 * np.pi)) + 0.5 * torch.sum(1 + q_var)
    q_c_x = torch.sum(p_c_z * torch.log(p_c_z))

    loss = p_z_c - p_c - q_z_x + q_c_x
    loss /= batch_data.size(0)
    loss += p_x_z  # Beware that we do not divide two times by number of samples
    return loss


class _VaDE_Module(torch.nn.Module):
    """
    The _VaDE_Module. Contains most of the algorithm specific procedures.

    Attributes
    ----------
    pi : the soft assignments
    p_mean : the cluster centers
    p_var : the variances of the clusters
    normalize_prob : torch.nn.Softmax function for the prediction method
    """

    def __init__(self, n_clusters, embedding_size, weights=None, means=None, variances=None):
        """
        Create an instance of the _VaDE_Module.

        Parameters
        ----------
        n_clusters : int, number of clusters
        embedding_size : int, size of the central layer within the VAE
        weights : the initial soft cluster assignments (default: None)
        means : the initial means of the VAE (default: None)
        variances : the initial variances of the VAE (default: None)
        """
        super(_VaDE_Module, self).__init__()
        if weights is None:
            # if not initialized then use uniform distribution
            weights = torch.ones(n_clusters) / n_clusters
        self.pi = torch.nn.Parameter(torch.tensor(weights), requires_grad=True)
        if means is None:
            # if not initialized then use torch.randn
            means = torch.randn(n_clusters, embedding_size)
        self.p_mean = torch.nn.Parameter(torch.tensor(means), requires_grad=True)
        if variances is None:
            variances = torch.ones(n_clusters, embedding_size)
        self.p_var = torch.nn.Parameter(torch.tensor(variances), requires_grad=True)
        self.normalize_prob = torch.nn.Softmax(dim=0)

    def predict(self, q_mean, q_logvar) -> torch.Tensor:
        """
        Predict the labels given the specific means and variances of given samples.
        Uses argmax to return a hard cluster label.

        Parameters
        ----------
        q_mean : mean values of the central layer of the VAE
        q_logvar : logarithmic variances of the central layer of the VAE (use logarithm of variance - numerical purposes)

        Returns
        -------
        The predicted label
        """
        z = _vae_sampling(q_mean, q_logvar)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        pred = torch.argmax(p_c_z, dim=1)
        return pred

    def vade_loss(self, autoencoder, batch_data, loss_fn):
        """
        Calculate the VaDE loss of given samples.

        Parameters
        ----------
        autoencoder : the VariationalAutoencoder
        batch_data : torch.Tensor, the samples
        loss_fn : torch.nn, loss function to be used for reconstruction

        Returns
        -------
        loss: returns the reconstruction loss of the input sample
        """
        z, q_mean, q_logvar, reconstruction = autoencoder.forward(batch_data)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        loss = _compute_vade_loss(pi_normalized, self.p_mean, self.p_var, q_mean, q_logvar, batch_data, p_c_z,
                                  reconstruction, loss_fn)
        return loss

    def fit(self, autoencoder, trainloader, n_epochs, device, optimizer, loss_fn):
        """
        Trains the _VaDE_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module, Variational Autoencoder
        trainloader : torch.utils.data.DataLoader, dataloader to be used for training
        n_epochs : int, number of epochs for the clustering procedure
        device : torch.device, device to be trained on
        optimizer : the optimizer
        loss_fn : loss function for the reconstruction
        """
        # lr_decrease = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # training loop
        for _ in range(n_epochs):
            self.train()
            for batch in trainloader:
                # load batch on device
                batch_data = batch[1].to(device)
                loss = self.vade_loss(autoencoder, batch_data, loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class VaDE(BaseEstimator, ClusterMixin):
    """
    The Variational Deep Embedding (VaDE) algorithm.

    Attributes
    ----------
    labels_
    cluster_centers_
    covariances_
    vade_labels_
    vade_cluster_centers_
    vade_covariances_
    autoencoder

    Examples
    ----------
    from cluspy.data import load_mnist
    data, labels = load_mnist()
    data = (data - np.mean(data)) / np.std(data)
    vade = VaDE(n_clusters=10)
    vade.fit(data)

    References
    ----------
    Jiang, Zhuxi, et al. "Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering." IJCAI. 2017.
    """

    def __init__(self, n_clusters, batch_size=256, pretrain_learning_rate=1e-3, clustering_learning_rate=1e-4,
                 pretrain_epochs=100, clustering_epochs=150, optimizer_class=torch.optim.Adam,
                 loss_fn=torch.nn.BCELoss(), autoencoder=None, embedding_size=10,
                 n_gmm_initializations=100):
        """
        Create an instance of the VaDE algorithm.

        Parameters
        ----------
        n_clusters : int, number of clusters
        batch_size : int, size of the data batches (default: 256)
        pretrain_learning_rate : double, learning rate for the pretraining of the autoencoder (default: 1e-3)
        clustering_learning_rate : double, learning rate of the actual clustering procedure (default: 1e-4)
        pretrain_epochs : int, number of epochs for the pretraining of the autoencoder (default: 100)
        clustering_epochs : int, number of epochs for the actual clustering procedure (default: 150)
        optimizer_class : Optimizer (default: torch.optim.Adam)
        loss_fn : loss function for the reconstruction (default: torch.nn.BCELoss())
        autoencoder : the input autoencoder. If None a variation of a VariationalAutoencoder will be created (default: None)
        embedding_size : int, size of the embedding within the autoencoder (central layer with mean and variance) (default: 10)
        n_gmm_initializations : int, number of initializations for the initial GMM clustering within the embedding (default: 100)
        """
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
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels are contained in the labels_ attribute.

        Parameters
        ----------
        X : the given data set
        y : the labels (can be ignored)

        Returns
        -------
        returns the clustering object
        """
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

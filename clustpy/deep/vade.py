"""
@authors:
Donatella Novakovic,
Lukas Miklautz,
Collin Leiber
"""

import torch
from clustpy.deep._utils import detect_device, set_torch_seed, encode_batchwise
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep.autoencoders.variational_autoencoder import VariationalAutoencoder, _vae_sampling
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _vade(X: np.ndarray, n_clusters: int, batch_size: int, pretrain_optimizer_params: dict,
          clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
          optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss, autoencoder: torch.nn.Module,
          embedding_size: int, custom_dataloaders: tuple, initial_clustering_class: ClusterMixin,
          initial_clustering_params: dict, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual VaDE clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    clustering_epochs : int
        number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        the optimizer class
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a variation of a VariationalAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder (central layer with mean and variance)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final Gaussian Mixture Model,
        The cluster centers as identified by a final Gaussian Mixture Model,
        The covariance matrices as identified by a final Gaussian Mixture Model,
        The labels as identified by VaDE after the training terminated,
        The cluster centers as identified by VaDE after the training terminated,
        The covariance matrices as identified by VaDE after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, autoencoder, _, n_clusters, _, init_means, init_clustering_algo = get_standard_initial_deep_clustering_setting(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, loss_fn, autoencoder,
        embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, random_state,
        _VaDE_VAE)
    # Get parameters from initial clustering algorithm
    init_weights = None if not hasattr(init_clustering_algo, "weights_") else init_clustering_algo.weights_
    init_covs = None if not hasattr(init_clustering_algo, "covariances_") else init_clustering_algo.covariances_
    # Initialize VaDE
    vade_module = _VaDE_Module(n_clusters=n_clusters, embedding_size=embedding_size, weights=init_weights,
                               means=init_means, variances=init_covs).to(device)
    # Use vade learning_rate (usually pretrain_optimizer_params reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(vade_module.parameters()),
                                **clustering_optimizer_params)
    # Vade Training loop
    vade_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn)
    # Get labels
    vade_labels = _vade_predict_batchwise(testloader, autoencoder, vade_module, device)
    vade_centers = vade_module.p_mean.detach().cpu().numpy()
    vade_covariances = vade_module.p_var.detach().cpu().numpy()
    # Do reclustering with GMM
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', n_init=1, random_state=random_state,
                          means_init=vade_centers)
    gmm_labels = gmm.fit_predict(embedded_data).astype(np.int32)
    # Return results
    return gmm_labels, gmm.means_, gmm.covariances_, gmm.weights_, vade_labels, vade_centers, vade_covariances, autoencoder


class _VaDE_VAE(VariationalAutoencoder):
    """
    A special variational autoencoder used for VaDE.
    Has a slightly different forward function while pretraining the autoencoder.
    Further, Loss function is more similar to the FeedforwardAutoencoder while pretraining.

    Attributes
    ----------
    encoder : FullyConnectedBlock
        encoder part of the autoencoder, responsible for embedding data points
    decoder : FullyConnectedBlock
        decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    mean : torch.nn.Linear
        mean value of the central layer
    log_variance : torch.nn.Linear
        logarithmic variance of the central layer (use logarithm of variance - numerical purposes)
    fitted : bool
        indicating whether the autoencoder is already fitted
    """

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Applies both the encode and decode function.
        The forward function is automatically called if we call self(x).
        Matches forward behavior from FeedforwardAutoencoder for pretraining and from VariationalAutoencoder afterwards.
        Overwrites function from VariationalAutoencoder.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of embedded points

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            sampling using q_mean and q_logvar if self.fitted=True, else None
            Mean value of the central VAE layer if self.fitted=True, else None
            Logarithmic variance value of the central VAE layer if self.fitted=True, else None
            The reconstruction of the data point
        """
        if not self.fitted:
            # While pretraining a forward method similar to a regular autoencoder (FeedforwardAutoencoder) should be used
            mean, _ = self.encode(x)
            reconstruction = self.decode(mean)
            z, q_mean, q_logvar = None, None, None
        else:
            # After pretraining the usual forward of a VAE should be used. Super() uses function from VariationalAutoencoder
            z, q_mean, q_logvar, reconstruction = super().forward(x)
        return z, q_mean, q_logvar, reconstruction

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
             beta: float = 1) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the loss of a single batch of data.
        Matches loss calculation from FeedforwardAutoencoder for pretraining and from VariationalAutoencoder afterwards.
        Overwrites function from VariationalAutoencoder.

        Parameters
        ----------
        batch: list
            the different parts of a dataloader (id, samples, ...)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on
        beta : float
            Not used at the moment

        Returns
        -------
        loss: (torch.Tensor, torch.Tensor, torch.Tensor)
            the reconstruction loss of the input sample,
            the sampling,
            the reconstruction of the data point
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        if not self.fitted:
            # While pretraining a loss similar to a regular autoencoder (FeedforwardAutoencoder) should be used
            batch_data = batch[1].to(device)
            z, _, _, reconstruction = self.forward(batch_data)
            loss = loss_fn(reconstruction, batch_data)
        else:
            # After pretraining the usual loss of a VAE should be used. Super() uses function from VariationalAutoencoder
            loss = super().loss(batch, loss_fn, beta)
        return loss, z, reconstruction


class _VaDE_Module(torch.nn.Module):
    """
    The _VaDE_Module. Contains most of the algorithm specific procedures.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    embedding_size : int
        size of the central layer within the VAE
    weights : torch.Tensor
        the initial soft cluster assignments (default: None)
    means : torch.Tensor
        the initial means of the VAE (default: None)
    variances : torch.Tensor
        the initial variances of the VAE (default: None)

    Attributes
    ----------
    pi : torch.nn.Parameter
        the soft assignments
    p_mean : torch.nn.Parameter
        the cluster centers
    p_var : torch.nn.Parameter
        the variances of the clusters
    normalize_prob : torch.nn.Softmax
        torch.nn.Softmax function for the prediction method
    """

    def __init__(self, n_clusters: int, embedding_size: int, weights: torch.Tensor = None, means: torch.Tensor = None,
                 variances: torch.Tensor = None):
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
        assert variances.shape == (n_clusters,
                                   embedding_size), "Shape of the initial variances for the Vade_Module must be (n_clusters, embedding_size)"
        self.p_var = torch.nn.Parameter(torch.tensor(variances), requires_grad=True)
        self.normalize_prob = torch.nn.Softmax(dim=0)

    def predict(self, q_mean: torch.Tensor, q_logvar: torch.Tensor) -> torch.Tensor:
        """
        Predict the labels given the specific means and variances of given samples.
        Uses argmax to return a hard cluster label.

        Parameters
        ----------
        q_mean : torch.Tensor
            mean values of the central layer of the VAE
        q_logvar : torch.Tensor
            logarithmic variances of the central layer of the VAE (use logarithm of variance - numerical purposes)

        Returns
        -------
        pred: torch.Tensor
            The predicted label
        """
        z = _vae_sampling(q_mean, q_logvar)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        pred = torch.argmax(p_c_z, dim=1)
        return pred

    def vade_loss(self, autoencoder: VariationalAutoencoder, batch_data: torch.Tensor,
                  loss_fn: torch.nn.modules.loss._Loss) -> torch.Tensor:
        """
        Calculate the VaDE loss of given samples.

        Parameters
        ----------
        autoencoder : VariationalAutoencoder
            the VariationalAutoencoder
        batch_data : torch.Tensor
            the samples
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction

        Returns
        -------
        loss : torch.Tensor
            returns the reconstruction loss of the input samples
        """
        z, q_mean, q_logvar, reconstruction = autoencoder.forward(batch_data)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        loss = _compute_vade_loss(pi_normalized, self.p_mean, self.p_var, q_mean, q_logvar, batch_data, p_c_z,
                                  reconstruction, loss_fn)
        return loss

    def fit(self, autoencoder: VariationalAutoencoder, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss) -> '_VaDE_Module':
        """
        Trains the _VaDE_Module in place.

        Parameters
        ----------
        autoencoder : VariationalAutoencoder
            The VariationalAutoencoder
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction

        Returns
        -------
        self : _VaDE_Module
            this instance of the _VaDE_Module
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
        return self


def _vade_predict_batchwise(dataloader: torch.utils.data.DataLoader, autoencoder: VariationalAutoencoder,
                            vade_module: _VaDE_Module, device: torch.device) -> np.ndarray:
    """
    Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    autoencoder : VariationalAutoencoder
        the VariationalAutoencoder
    vade_module : _VaDE_Module
        the _VaDE_Module
    device : torch.device
        device on which the prediction should take place

    Returns
    -------
    predictions_numpy : np.ndarray
        The cluster labels of the data set
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        q_mean, q_logvar = autoencoder.encode(batch_data)
        prediction = vade_module.predict(q_mean, q_logvar).detach().cpu()
        predictions.append(prediction)
    predictions_numpy = torch.cat(predictions, dim=0).numpy()
    return predictions_numpy


def _get_gamma(pi: torch.Tensor, p_mean: torch.Tensor, p_var: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Calculate the gamma of samples created by the VAE.

    Parameters
    ----------
    pi : torch.Tensor
        softmax version of the soft cluster assignments in the _VaDE_Module
    p_mean : torch.Tensor
        cluster centers of the _VaDE_Module
    p_var : torch.Tensor
        variances of the _VaDE_Module
    z : torch.Tensor
        the created samples

    Returns
    -------
    p_c_z : torch.Tensor
        The gamma values
    """
    z = z.unsqueeze(1)
    p_var = p_var.unsqueeze(0)
    pi = pi.unsqueeze(0)

    p_z_c = -torch.sum(0.5 * (np.log(2 * np.pi)) + p_var + ((z - p_mean).pow(2) / (2. * torch.exp(p_var))), dim=2)
    p_c_z_c = torch.exp(torch.log(pi) + p_z_c) + 1e-10
    p_c_z = p_c_z_c / torch.sum(p_c_z_c, dim=1, keepdim=True)

    return p_c_z


def _compute_vade_loss(pi: torch.Tensor, p_mean: torch.Tensor, p_var: torch.Tensor, q_mean: torch.Tensor,
                       q_var: torch.Tensor, batch_data: torch.Tensor, p_c_z: torch.Tensor, reconstruction: torch.Tensor,
                       loss_fn: torch.nn.modules.loss._Loss) -> torch.Tensor:
    """
    Calculate the final loss of the input samples for the VaDE algorithm.

    Parameters
    ----------
    pi : torch.Tensor
        softmax version of the soft cluster assignments in the _VaDE_Module
    p_mean : torch.Tensor
        cluster centers of the _VaDE_Module
    p_var : torch.Tensor
        variances of the _VaDE_Module
    q_mean : torch.Tensor
        mean value of the central layer of the VAE
    q_var : torch.Tensor
        logarithmic variance of the central layer of the VAE
    batch_data : torch.Tensor
        the samples
    p_c_z : torch.Tensor
        result of the _get_gamma function
    reconstruction : torch.Tensor
        the reconstructed version of the input samples
    loss_fn : torch.nn.modules.loss._Loss
        loss function to be used for reconstruction

    Returns
    -------
    loss: torch.Tensor
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


class VaDE(BaseEstimator, ClusterMixin):
    """
    The Variational Deep Embedding (VaDE) algorithm.
    First, an variational autoencoder (VAE) will be trained (will be skipped if input autoencoder is given).
    Afterward, a GMM will be fit to identify the initial clustering structures.
    Last, the VAE will be optimized using the VaDE loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.BCELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a variation of a VariationalAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (central layer with mean and variance) (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: GaussianMixture)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {"n_init": 10, "covariance_type": "diag"})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The labels as identified by a final Gaussian Mixture Model
    cluster_centers_ : np.ndarray
        The cluster centers as identified by a final Gaussian Mixture Model
    covariances_ : np.ndarray
        The covariance matrices as identified by a final Gaussian Mixture Model
    weights_ : np.ndarray
        The weights as identified by a final Gaussian Mixture Model
    vade_labels_ : np.ndarray
        The labels as identified by VaDE after the training terminated
    vade_cluster_centers_ : np.ndarray
        The cluster centers as identified by VaDE after the training terminated
    vade_covariances_ : np.ndarray
        The covariance matrices as identified by VaDE after the training terminated
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> data = (data - np.mean(data)) / np.std(data)
    >>> vade = VaDE(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> vade.fit(data)

    References
    ----------
    Jiang, Zhuxi, et al. "Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering." IJCAI. 2017.
    """

    def __init__(self, n_clusters: int, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 150, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.BCELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, custom_dataloaders: tuple = None,
                 initial_clustering_class: ClusterMixin = GaussianMixture,
                 initial_clustering_params: dict = None,
                 random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {
            "lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.custom_dataloaders = custom_dataloaders
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = {"n_init": 10,
                                          "covariance_type": "diag"} if initial_clustering_params is None else initial_clustering_params
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'VaDE':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : VaDE
            this instance of the VaDE algorithm
        """
        assert type(self.loss_fn) != torch.nn.modules.loss.BCELoss or (np.min(X) >= 0 and np.max(
            X) <= 1), "Your dataset contains values that are not in the value range [0, 1]. Therefore, BCE is not a valid loss function, an alternative might be a MSE loss function."
        gmm_labels, gmm_means, gmm_covariances, gmm_weights, vade_labels, vade_centers, vade_covariances, autoencoder = _vade(
            X,
            self.n_clusters,
            self.batch_size,
            self.pretrain_optimizer_params,
            self.clustering_optimizer_params,
            self.pretrain_epochs,
            self.clustering_epochs,
            self.optimizer_class,
            self.loss_fn,
            self.autoencoder,
            self.embedding_size,
            self.custom_dataloaders,
            self.initial_clustering_class,
            self.initial_clustering_params,
            self.random_state)
        self.labels_ = gmm_labels
        self.cluster_centers_ = gmm_means
        self.covariances_ = gmm_covariances
        self.weights_ = gmm_weights
        self.vade_labels_ = vade_labels
        self.vade_cluster_centers_ = vade_centers
        self.vade_covariances_ = vade_covariances
        self.autoencoder = autoencoder
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        device = detect_device()
        dataloader = get_dataloader(X, self.batch_size, False, False)
        embedded_data = encode_batchwise(dataloader, self.autoencoder, device)
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='full')
        gmm.means_ = self.cluster_centers_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covariances_))
        predicted_labels = gmm.predict(embedded_data).astype(np.int32)
        return predicted_labels

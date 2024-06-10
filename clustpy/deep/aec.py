"""
@authors:
Collin Leiber
"""

from clustpy.deep._utils import embedded_kmeans_prediction, encode_batchwise
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from clustpy.deep._data_utils import augmentation_invariance_check
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
import torch
import numpy as np
from sklearn.base import ClusterMixin
from clustpy.deep.dcn import _DCN_Module


def _aec(X: np.ndarray, n_clusters: int, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss, autoencoder: torch.nn.Module,
         embedding_size: int, clustering_loss_weight: float, reconstruction_loss_weight: float,
         custom_dataloaders: tuple, augmentation_invariance: bool, initial_clustering_class: ClusterMixin,
         initial_clustering_params: dict,
         random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual AEC clustering procedure on the input data set.

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
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    clustering_loss_weight : float
        weight of the clustering loss
    reconstruction_loss_weight : float
        weight of the reconstruction loss
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by AEC after the training terminated,
        The cluster centers as identified by AEC after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, autoencoder, _, n_clusters, init_labels, init_centers, _ = get_standard_initial_deep_clustering_setting(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, loss_fn, autoencoder,
        embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, random_state)
    # Setup AEC Module
    aec_module = _AEC_Module(n_clusters, init_labels, init_centers, augmentation_invariance).to_device(device)
    # Use AEC optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()), **clustering_optimizer_params)
    # AEC Training loop
    aec_module.fit(autoencoder, trainloader, testloader, clustering_epochs, device, optimizer, loss_fn,
                   clustering_loss_weight, reconstruction_loss_weight)
    # Get labels and centers as numpy arrays
    aec_labels = aec_module.labels.detach().cpu().numpy().astype(np.int32)
    aec_centers = aec_module.centers.detach().cpu().numpy()
    return aec_labels, aec_centers, autoencoder


class _AEC_Module(_DCN_Module):
    """
    The _AEC_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    init_np_labels : np.ndarray
        The initial numpy labels
    init_np_centers : np.ndarray
        The initial numpy centers
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    labels : torch.Tensor
        the labels
    centers : torch.Tensor
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(self, n_clusters: int, init_np_labels: np.ndarray, init_np_centers: np.ndarray,
                 augmentation_invariance: bool = False):
        super().__init__(init_np_centers, augmentation_invariance)
        self.labels = torch.tensor(init_np_labels)
        self.n_clusters = n_clusters

    def to_device(self, device: torch.device) -> '_AEC_Module':
        """
        Move the _AEC_Module and the cluster labels and centers to the specified device (cpu or cuda).

        Parameters
        ----------
        device : torch.device
            device to be trained on

        Returns
        -------
        self : _AEC_Module
            this instance of the _AEC_Module
        """
        super().to_device(device)
        self.labels = self.labels.to(device)
        return self

    def update_centroids(self, embedded: np.ndarray) -> torch.Tensor:
        """
        Update the cluster centers of the _AEC_Module.

        Parameters
        ----------
        embedded : np.ndarray
            the embedded samples

        Returns
        -------
        centers : torch.Tensor
            The updated centers
        """
        labels_cpu = self.labels.cpu().detach().numpy()
        centers = torch.from_numpy(np.array(
            [np.mean(embedded[labels_cpu == i], axis=0) for i in range(self.n_clusters)]))
        return centers

    def _loss(self, batch: list, autoencoder: torch.nn.Module, loss_fn: torch.nn.modules.loss._Loss,
              reconstruction_loss_weight: float, clustering_loss_weight: float, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete AEC + Autoencoder loss.

        Parameters
        ----------
        batch : list
            the minibatch
        autoencoder : torch.nn.Module
            the autoencoder
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        clustering_loss_weight : float
            weight of the clustering loss
        reconstruction_loss_weight : float
            weight of the reconstruction loss
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final AEC loss
        """
        # compute reconstruction loss
        if self.augmentation_invariance:
            # Convention is that the augmented sample is at the first position and the original one at the second position
            ae_loss, embedded, _ = autoencoder.loss([batch[0], batch[2]], loss_fn, device)
            ae_loss_aug, embedded_aug, _ = autoencoder.loss([batch[0], batch[1]], loss_fn, device)
            ae_loss = (ae_loss + ae_loss_aug) / 2
        else:
            ae_loss, embedded, _ = autoencoder.loss(batch, loss_fn, device)

        # compute cluster loss
        assignments = self.labels[batch[0]]
        cluster_loss = self.dcn_loss(embedded, assignments)
        if self.augmentation_invariance:
            # assign augmented samples to the same cluster as original samples
            cluster_loss_aug = self.dcn_loss(embedded_aug, assignments)
            cluster_loss = (cluster_loss + cluster_loss_aug) / 2

        # compute total loss
        loss = reconstruction_loss_weight * ae_loss + clustering_loss_weight * cluster_loss

        return loss

    def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
            n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
            clustering_loss_weight: float, reconstruction_loss_weight: float) -> '_AEC_Module':
        """
        Trains the _AEC_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            the autoencoder
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        testloader : torch.utils.data.DataLoader
            dataloader to be used for updating the clustering parameters
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        clustering_loss_weight : float
            weight of the clustering loss
        reconstruction_loss_weight : float
            weight of the reconstruction loss

        Returns
        -------
        self : _AE_Module
            this instance of the _AEC_Module
        """
        # AEC training loop
        for _ in range(n_epochs):
            # Update Network
            for batch in trainloader:
                loss = self._loss(batch, autoencoder, loss_fn, reconstruction_loss_weight, clustering_loss_weight,
                                  device)
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update Assignments and Centroids
            embedded = encode_batchwise(testloader, autoencoder)
            # update centroids
            centers = self.update_centroids(embedded)
            self.centers = centers.to(device)
            # update assignments
            labels = self.predict_hard(torch.tensor(embedded).to(device))
            self.labels = labels.to(device)
        return self


class AEC(_AbstractDeepClusteringAlgo):
    """
    The Auto-encoder Based Data Clustering (AEC) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the AEC loss function.

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
         loss function for the reconstruction (default: torch.nn.MSELoss())
    clustering_loss_weight : float
        weight of the clustering loss (default: 0.05)
    reconstruction_loss_weight : float
        weight of the reconstruction loss (default: 1.0)
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining.
        If this is None, random labels will be used (default: None)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import AEC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> aec = AEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> AEC.fit(data)

    References
    ----------
    Song, Chunfeng, et al. "Auto-encoder based data clustering."
    Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications: 18th Iberoamerican Congress,
    CIARP 2013, Havana, Cuba, November 20-23, 2013, Proceedings, Part I 18. Springer Berlin Heidelberg, 2013.
    """

    def __init__(self, n_clusters: int, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 50, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), clustering_loss_weight: float = 0.1,
                 reconstruction_loss_weight: float = 1.0, autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 initial_clustering_class: ClusterMixin = None, initial_clustering_params: dict = None,
                 random_state: np.random.RandomState = None):
        super().__init__(batch_size, autoencoder, embedding_size, random_state)
        self.n_clusters = n_clusters
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {
            "lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = {} if initial_clustering_params is None else initial_clustering_params

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'AEC':
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
        self : AEC
            this instance of the AEC algorithm
        """
        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        aec_labels, aec_centers, autoencoder = _aec(X, self.n_clusters, self.batch_size,
                                                    self.pretrain_optimizer_params,
                                                    self.clustering_optimizer_params,
                                                    self.pretrain_epochs,
                                                    self.clustering_epochs,
                                                    self.optimizer_class, self.loss_fn,
                                                    self.autoencoder,
                                                    self.embedding_size,
                                                    self.clustering_loss_weight,
                                                    self.reconstruction_loss_weight,
                                                    self.custom_dataloaders,
                                                    self.augmentation_invariance,
                                                    self.initial_clustering_class,
                                                    self.initial_clustering_params,
                                                    self.random_state)
        self.labels_ = aec_labels
        self.cluster_centers_ = aec_centers
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
        X_embed = self.transform(X)
        predicted_labels = embedded_kmeans_prediction(X_embed, self.cluster_centers_)
        return predicted_labels

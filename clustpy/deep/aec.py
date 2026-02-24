"""
@authors:
Collin Leiber
"""

from clustpy.deep._utils import encode_batchwise, mean_squared_error
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
import torch
import numpy as np
from sklearn.base import ClusterMixin
from clustpy.deep.dcn import _DCN_Module
import tqdm
from collections.abc import Callable


def _aec(X: np.ndarray, val_set: np.ndarray | None, n_clusters: int, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
         neural_network: torch.nn.Module | tuple, neural_network_weights: str,
         embedding_size: int, clustering_loss_weight: float, ssl_loss_weight: float,
         custom_dataloaders: tuple, augmentation_invariance: bool, initial_clustering_class: ClusterMixin,
         initial_clustering_params: dict, device: torch.device,
         log_fn: Callable | None,random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual AEC clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, that can determine the number of clusters, e.g. DBSCAN
    val_set : np.ndarray | None
        Optional validation set for early stopping. If not None, Early stopping will be used    
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network
    clustering_epochs : int
        number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        the optimizer class
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    neural_network : torch.nn.Module | tuple
        the input neural network.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network.
    embedding_size : int
        size of the embedding within the neural network
    clustering_loss_weight : float
        weight of the clustering loss
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    device : torch.device
        The device on which to perform the computations
    log_fn : Callable | None
        function for logging training history values (e.g. loss values) during training
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution



    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by AEC after the training terminated,
        The cluster centers as identified by AEC after the training terminated,
        The final neural network
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, _, neural_network, _, n_clusters, init_labels, init_centers, _ = get_default_deep_clustering_initialization(
        X, val_set, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, ssl_loss_fn,
        neural_network, embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, device,
        random_state, log_fn=log_fn, neural_network_weights=neural_network_weights)
    # Setup AEC Module
    aec_module = _AEC_Module(init_labels, init_centers, augmentation_invariance,log_fn).to_device(device)
    # Use AEC optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(neural_network.parameters()), **clustering_optimizer_params)
    # AEC Training loop
    aec_module.fit(neural_network, trainloader, testloader, clustering_epochs, device, optimizer, ssl_loss_fn,
                   clustering_loss_weight, ssl_loss_weight)
    # Get labels and centers as numpy arrays
    aec_labels = aec_module.labels.detach().cpu().numpy().astype(np.int32)
    aec_centers = aec_module.centers.detach().cpu().numpy()
    return aec_labels, aec_centers, neural_network


class _AEC_Module(_DCN_Module):
    """
    The _AEC_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
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
    log_fn : Callable | None
        function for logging training history values (e.g. loss values) during training
    """

    def __init__(self, init_np_labels: np.ndarray, init_np_centers: np.ndarray,
                 augmentation_invariance: bool = False, log_fn: Callable | None = None):
        super().__init__(init_np_labels, init_np_centers, augmentation_invariance,log_fn)

    def update_centroids(self, embedded: np.ndarray, labels: np.ndarray) -> torch.Tensor:
        """
        Update the cluster centers of the _AEC_Module.

        Parameters
        ----------
        embedded : np.ndarray
            the embedded samples
        labels : np.ndarray
            The current hard labels

        Returns
        -------
        centers : torch.Tensor
            The updated centers
        """
        centers = self.centers.cpu().detach().numpy()
        for i in range(self.centers.shape[0]):
            X_in_cluster = embedded[labels == i]
            if X_in_cluster.shape[0] > 0:
                    centers[i] = np.mean(X_in_cluster, axis=0)
        centers_torch = torch.from_numpy(centers)
        return centers_torch

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader, n_epochs: int, device: torch.device,
            optimizer: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss, clustering_loss_weight: float,
            ssl_loss_weight: float) -> '_AEC_Module':
        """
        Trains the _AEC_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
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
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss

        Returns
        -------
        self : _AE_Module
            this instance of the _AEC_Module
        """
        # AEC training loop
        tbar = tqdm.trange(n_epochs, desc="AEC training")
        for _ in tbar:
            # Update Network
            total_loss = 0
            total_ssl_loss = 0 
            total_clustering_loss = 0
            for batch in trainloader:
                # Beware that the clustering loss of DCN is divided by 2, therefore we use 2 * clustering_loss_weight
                loss = self._loss(batch, neural_network, ssl_loss_fn, ssl_loss_weight,
                                  2 * clustering_loss_weight, device)
                total_loss += loss[0].item()
                total_ssl_loss += loss[1].item()
                total_clustering_loss += loss[2].item()
                # Backward pass - update weights
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()
            postfix_str = {"Loss": total_loss}
            tbar.set_postfix(postfix_str)
            # Update Assignments and Centroids
            embedded = encode_batchwise(testloader, neural_network)
            # update centroids
            centers = self.update_centroids(embedded, self.labels.cpu().detach().numpy())
            self.centers = centers.to(device)
            # update assignments
            labels = self.predict_hard(torch.tensor(embedded).to(device))
            self.labels = labels.to(device)
            if self.log_fn is not None:
                self.log_fn("Total Loss", total_loss)
                self.log_fn("SSL Loss", total_ssl_loss)
                self.log_fn("Clustering Loss", total_clustering_loss)
        return self


class AEC(_AbstractDeepClusteringAlgo):
    """
    The Auto-encoder Based Data Clustering (AEC) algorithm.
    First, a neural network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the AEC loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, that can determine the number of clusters, e.g. DBSCAN (default: 8)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate. If None, it will be set to {"lr": 1e-3} (default: None)
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate. If None, it will be set to {"lr": 1e-4} (default: None)
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: mean_squared_error)
    clustering_loss_weight : float
        weight of the clustering loss (default: 0.1)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining.
        If this is None, random labels will be used (default: None)
    initial_clustering_params : dict
        parameters for the initial clustering class. If None, it will be set to {} (default: None)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    neural_network_trained_ : torch.nn.Module
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

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

    def __init__(self, n_clusters: int = 8, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 150, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error, clustering_loss_weight: float = 0.1,
                 ssl_loss_weight: float = 1.0, neural_network: torch.nn.Module | tuple = None,
                 neural_network_weights: str = None, embedding_size: int = 10, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = None,
                 initial_clustering_params: dict = None, device: torch.device = None,
                 random_state: np.random.RandomState | int = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters = n_clusters
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params

    def fit(self, X: np.ndarray, val_set: np.ndarray = None, y: np.ndarray = None) -> 'AEC':
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
        X, _, random_state, pretrain_optimizer_params, clustering_optimizer_params, initial_clustering_params = self._check_parameters(X, y=y)
        aec_labels, aec_centers, neural_network = _aec(X, val_set, self.n_clusters, self.batch_size,
                                                       pretrain_optimizer_params,
                                                       clustering_optimizer_params,
                                                       self.pretrain_epochs,
                                                       self.clustering_epochs,
                                                       self.optimizer_class, self.ssl_loss_fn,
                                                       self.neural_network,
                                                       self.neural_network_weights,
                                                       self.embedding_size,
                                                       self.clustering_loss_weight,
                                                       self.ssl_loss_weight,
                                                       self.custom_dataloaders,
                                                       self.augmentation_invariance,
                                                       self.initial_clustering_class,
                                                       initial_clustering_params,
                                                       self.device,
                                                       self._log_history,
                                                       random_state)
        self.labels_ = aec_labels
        self.cluster_centers_ = aec_centers
        self.neural_network_trained_ = neural_network
        self.set_n_featrues_in(X)
        return self

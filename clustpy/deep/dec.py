"""
@authors:
Lukas Miklautz,
Dominik Mautz,
Collin Leiber
"""

from clustpy.deep._utils import encode_batchwise, squared_euclidean_distance, predict_batchwise, \
    embedded_kmeans_prediction
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
import tqdm


def _dec(X: np.ndarray, n_clusters: int, alpha: float, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss,
         neural_network: torch.nn.Module | tuple, neural_network_weights: str, embedding_size: int,
         ssl_loss_weight: float, clustering_loss_weight: float, custom_dataloaders: tuple,
         augmentation_invariance: bool, initial_clustering_class: ClusterMixin, initial_clustering_params: dict,
         device: torch.device, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DEC clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, that can determine the number of clusters, e.g. DBSCAN
    alpha : float
        alpha value for the prediction
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
        the optimizer
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    neural_network : torch.nn.Module | tuple
        the input neural network.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network.
    embedding_size : int
        size of the embedding within the neural network
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss
    clustering_loss_weight : float
        weight of the clustering loss
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
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DEC after the training terminated,
        The cluster centers as identified by DEC after the training terminated,
        The final neural network
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, _, neural_network, _, n_clusters, _, init_centers, _ = get_default_deep_clustering_initialization(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, ssl_loss_fn,
        neural_network, embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, device,
        random_state, neural_network_weights=neural_network_weights)
    # Setup DEC Module
    dec_module = _DEC_Module(init_centers, alpha, augmentation_invariance).to(device)
    # Use DEC optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(neural_network.parameters()) + list(dec_module.parameters()),
                                **clustering_optimizer_params)
    # DEC Training loop
    dec_module.fit(neural_network, trainloader, clustering_epochs, device, optimizer, ssl_loss_fn,
                   ssl_loss_weight, clustering_loss_weight)
    # Get labels
    dec_labels = predict_batchwise(testloader, neural_network, dec_module)
    dec_centers = dec_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, neural_network)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dec_labels, dec_centers, neural_network


def _dec_predict(centers: torch.Tensor, embedded: torch.Tensor, alpha: float, weights: torch.Tensor) -> torch.Tensor:
    """
    Predict soft cluster labels given embedded samples.

    Parameters
    ----------
    centers : torch.Tensor
        the cluster centers
    embedded : torch.Tensor
        the embedded samples
    alpha : float
        the alpha value
    weights : torch.Tensor
        feature weights for the squared Euclidean distance


    Returns
    -------
    prob : torch.Tensor
        The predicted soft labels
    """
    squared_diffs = squared_euclidean_distance(embedded, centers, weights)
    numerator = (1.0 + squared_diffs / alpha).pow(-1.0 * (alpha + 1.0) / 2.0)
    denominator = numerator.sum(1)
    prob = numerator / denominator.unsqueeze(1)
    return prob


def _dec_compression_value(pred_labels: torch.Tensor) -> torch.Tensor:
    """
    Get the DEC compression values.

    Parameters
    ----------
    pred_labels : torch.Tensor
        the predictions of the embedded samples.

    Returns
    -------
    p : torch.Tensor
        The compression values
    """
    soft_freq = pred_labels.sum(0)
    squared_pred = pred_labels.pow(2)
    normalized_squares = squared_pred / soft_freq.unsqueeze(0)
    sum_normalized_squares = normalized_squares.sum(1)
    p = normalized_squares / sum_normalized_squares.unsqueeze(1)
    return p


def _dec_compression_loss_fn(pred_labels: torch.Tensor, target_p: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the loss of DEC by computing the DEC compression value.

    Parameters
    ----------
    pred_labels : torch.Tensor
        the predictions of the embedded samples.
    target_p : torch.Tensor
        dec_compression_value used as pseudo target labels

    Returns
    -------
    loss : torch.Tensor
        The final loss
    """
    if target_p is None:
        target_p = _dec_compression_value(pred_labels).detach().data
    loss = -1.0 * torch.mean(torch.sum(target_p * torch.log(pred_labels + 1e-8), dim=1))
    return loss


class _DEC_Module(torch.nn.Module):
    """
    The _DEC_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers
    alpha : double
        alpha value for the prediction method
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    alpha : float
        the alpha value
    centers : torch.Tensor:
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(self, init_centers: np.ndarray, alpha: float, augmentation_invariance: bool = False):
        super().__init__()
        self.alpha = alpha
        self.augmentation_invariance = augmentation_invariance
        # Centers are learnable parameters
        self.centers = torch.nn.Parameter(torch.tensor(init_centers), requires_grad=True)

    def predict(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Soft prediction of given embedded samples. Returns the corresponding soft labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        pred : torch.Tensor
            The predicted soft labels
        """
        pred = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        return pred

    def predict_hard(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the soft prediction method and then applies argmax.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        pred_hard : torch.Tensor
            The predicted hard labels
        """
        pred_hard = self.predict(embedded, weights=weights).argmax(1)
        return pred_hard

    def dec_loss(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DEC loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        prediction = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        loss = _dec_compression_loss_fn(prediction)
        return loss

    def dec_augmentation_invariance_loss(self, embedded: torch.Tensor, embedded_aug: torch.Tensor,
                                         weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DEC loss of given embedded samples with augmentation invariance.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor
            the embedded augmented samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        prediction = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        # Predict pseudo cluster labels with clean samples
        clean_target_p = _dec_compression_value(prediction).detach().data
        # Calculate loss from clean prediction and clean targets
        clean_loss = _dec_compression_loss_fn(prediction, clean_target_p)

        # Predict pseudo cluster labels with augmented samples
        aug_prediction = _dec_predict(self.centers, embedded_aug, self.alpha, weights=weights)
        # Calculate loss from augmented prediction and reused clean targets to enforce that the cluster assignment is invariant against augmentations
        aug_loss = _dec_compression_loss_fn(aug_prediction, clean_target_p)

        # average losses
        loss = (clean_loss + aug_loss) / 2
        return loss

    def _loss(self, batch: list, neural_network: torch.nn.Module, clustering_loss_weight: float,
              ssl_loss_weight: float, ssl_loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete DEC + optional neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        neural_network : torch.nn.Module
            the neural network
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the clustering loss
        ssl_loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        loss = torch.tensor(0.).to(device)
        # Reconstruction loss is not included in DEC
        if ssl_loss_weight != 0:
            if self.augmentation_invariance:
                ssl_loss, embedded, _, embedded_aug, _ = neural_network.loss_augmentation(batch, ssl_loss_fn, device)
            else:
                ssl_loss, embedded, _ = neural_network.loss(batch, ssl_loss_fn, device)
            loss += ssl_loss_weight * ssl_loss
        else:
            if self.augmentation_invariance:
                aug_data = batch[1].to(device)
                embedded_aug = neural_network.encode(aug_data)
                orig_data = batch[2].to(device)
                embedded = neural_network.encode(orig_data)
            else:
                batch_data = batch[1].to(device)
                embedded = neural_network.encode(batch_data)

        # CLuster loss
        if self.augmentation_invariance:
            cluster_loss = self.dec_augmentation_invariance_loss(embedded, embedded_aug)
        else:
            cluster_loss = self.dec_loss(embedded)
        loss += cluster_loss * clustering_loss_weight

        return loss

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss,
            ssl_loss_weight: float, clustering_loss_weight: float) -> '_DEC_Module':
        """
        Trains the _DEC_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        clustering_loss_weight : float
            weight of the clustering loss

        Returns
        -------
        self : _DEC_Module
            this instance of the _DEC_Module
        """
        tbar = tqdm.trange(n_epochs, desc="DEC training")
        for _ in tbar:
            total_loss = 0
            for batch in trainloader:
                loss = self._loss(batch, neural_network, clustering_loss_weight, ssl_loss_weight, ssl_loss_fn,
                                  device)
                total_loss += loss.item()
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Loss": total_loss}
            tbar.set_postfix(postfix_str)
        return self


class DEC(_AbstractDeepClusteringAlgo):
    """
    The Deep Embedded Clustering (DEC) algorithm.
    First, a neural_network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the DEC loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, that can determine the number of clusters, e.g. DBSCAN
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    clustering_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
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
    dec_labels_ : np.ndarray
        The final DEC labels
    dec_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    neural_network : torch.nn.Module
        The final neural network

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DEC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> dec = DEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> dec.fit(data)

    References
    ----------
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. 2016.
    """

    def __init__(self, n_clusters: int, alpha: float = 1.0, batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 1., custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, device: torch.device = None,
                 random_state: np.random.RandomState | int = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {
            "lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = {} if initial_clustering_params is None else initial_clustering_params
        self.ssl_loss_weight = 0  # DEC does not use ssl loss when clustering

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DEC':
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
        self : DEC
            this instance of the DEC algorithm
        """
        super().fit(X, y)
        kmeans_labels, kmeans_centers, dec_labels, dec_centers, neural_network = _dec(X, self.n_clusters, self.alpha,
                                                                                      self.batch_size,
                                                                                      self.pretrain_optimizer_params,
                                                                                      self.clustering_optimizer_params,
                                                                                      self.pretrain_epochs,
                                                                                      self.clustering_epochs,
                                                                                      self.optimizer_class,
                                                                                      self.ssl_loss_fn,
                                                                                      self.neural_network,
                                                                                      self.neural_network_weights,
                                                                                      self.embedding_size,
                                                                                      self.ssl_loss_weight,
                                                                                      self.clustering_loss_weight,
                                                                                      self.custom_dataloaders,
                                                                                      self.augmentation_invariance,
                                                                                      self.initial_clustering_class,
                                                                                      self.initial_clustering_params,
                                                                                      self.device, self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dec_labels_ = dec_labels
        self.dec_cluster_centers_ = dec_centers
        self.neural_network = neural_network
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


class IDEC(DEC):
    """
    The Improved Deep Embedded Clustering (IDEC) algorithm.
    Is equal to the DEC algorithm but uses the self-supervised learning loss also during the clustering optimization.
    Further, clustering_loss_weight is set to 0.1 instead of 1 when using the default settings.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, that can determine the number of clusters, e.g. DBSCAN
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    clustering_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 0.1)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
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
    dec_labels_ : np.ndarray
        The final DEC labels
    dec_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    neural_network : torch.nn.Module
        The final neural network

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import IDEC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> idec = IDEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> idec.fit(data)

    References
    ----------
    Guo, Xifeng, et al. "Improved deep embedded clustering with local structure preservation." IJCAI. 2017.
    """

    def __init__(self, n_clusters: int, alpha: float = 1.0, batch_size: int = 256,
                 pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 150, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 0.1, ssl_loss_weight: float = 1.0,
                 custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 initial_clustering_class: ClusterMixin = KMeans, initial_clustering_params: dict = None,
                 device: torch.device = None, random_state: np.random.RandomState | int = None):
        super().__init__(n_clusters, alpha, batch_size, pretrain_optimizer_params, clustering_optimizer_params,
                         pretrain_epochs, clustering_epochs, optimizer_class, ssl_loss_fn, neural_network,
                         neural_network_weights, embedding_size, clustering_loss_weight, custom_dataloaders,
                         augmentation_invariance, initial_clustering_class,
                         initial_clustering_params, device, random_state)
        self.ssl_loss_weight = ssl_loss_weight

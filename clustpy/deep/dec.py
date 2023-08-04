"""
@authors:
Lukas Miklautz,
Dominik Mautz,
Collin Leiber
"""

from clustpy.deep._utils import detect_device, encode_batchwise, squared_euclidean_distance, predict_batchwise, \
    set_torch_seed, run_initial_clustering, embedded_kmeans_prediction
from clustpy.deep._data_utils import get_dataloader, augmentation_invariance_check
from clustpy.deep._train_utils import get_trained_autoencoder
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dec(X: np.ndarray, n_clusters: int, alpha: float, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
         autoencoder: torch.nn.Module, embedding_size: int, use_reconstruction_loss: bool,
         cluster_loss_weight: float, custom_dataloaders: tuple, augmentation_invariance: bool,
         initial_clustering_class: ClusterMixin,
         initial_clustering_params: dict, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DEC clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    alpha : float
        alpha value for the prediction
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
        the optimizer
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    use_reconstruction_loss : bool
        defines whether the reconstruction loss will be used during clustering training
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss
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
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DEC after the training terminated,
        The cluster centers as identified by DEC after the training terminated,
        The final autoencoder
    """
    device = detect_device()
    if custom_dataloaders is None:
        trainloader = get_dataloader(X, batch_size, True, False)
        testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
    autoencoder = get_trained_autoencoder(trainloader, pretrain_optimizer_params, pretrain_epochs, device,
                                          optimizer_class, loss_fn, embedding_size, autoencoder)
    # Execute initial clustering in embedded space
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    n_clusters, _, init_centers, _ = run_initial_clustering(embedded_data, n_clusters,
                                                            initial_clustering_class,
                                                            initial_clustering_params, random_state)
    # Setup DEC Module
    dec_module = _DEC_Module(init_centers, alpha, augmentation_invariance).to(device)
    # Use DEC optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(dec_module.parameters()),
                                **clustering_optimizer_params)
    # DEC Training loop
    dec_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn,
                   use_reconstruction_loss, cluster_loss_weight)
    # Get labels
    dec_labels = predict_batchwise(testloader, autoencoder, dec_module, device)
    dec_centers = dec_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dec_labels, dec_centers, autoencoder


def _dec_predict(centers: torch.Tensor, embedded: torch.Tensor, alpha: float, weights) -> torch.Tensor:
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
        feature weights for the squared euclidean distance (default: None)


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
        If True, augmented samples provided in will be used to learn 
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
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        pred : torch.Tensor
            The predicted soft labels
        """
        pred = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        return pred

    def predict_hard(self, embedded: torch.Tensor, weights=None) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the soft prediction method and then applies argmax.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

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
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

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
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

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

    def _loss(self, batch, autoencoder, cluster_loss_weight, use_reconstruction_loss, loss_fn, device) -> torch.Tensor:
        """
        Calculate the complete DEC + optional Autoencoder loss.

        Parameters
        ----------
        batch : list
            the minibatch
        autoencoder : torch.nn.Module
            the autoencoder
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss
        use_reconstruction_loss : bool
            defines whether the reconstruction loss will be used during clustering training
        loss_fn : torch.nn.modules.loss._Loss
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
        if use_reconstruction_loss:
            if self.augmentation_invariance:
                # Convention is that the augmented sample is at the first position and the original one at the second position
                ae_loss, embedded, _ = autoencoder.loss([batch[0], batch[2]], loss_fn, device)
                ae_loss_aug, embedded_aug, _ = autoencoder.loss([batch[0], batch[1]], loss_fn, device)
                loss += ((ae_loss + ae_loss_aug) / 2)
            else:
                ae_loss, embedded, _ = autoencoder.loss(batch, loss_fn, device)
                loss += ae_loss
        else:
            if self.augmentation_invariance:
                aug_data = batch[1].to(device)
                embedded_aug = autoencoder.encode(aug_data)
                orig_data = batch[2].to(device)
                embedded = autoencoder.encode(orig_data)
            else:
                batch_data = batch[1].to(device)
                embedded = autoencoder.encode(batch_data)

        # CLuster loss
        if self.augmentation_invariance:
            cluster_loss = self.dec_augmentation_invariance_loss(embedded, embedded_aug)
        else:
            cluster_loss = self.dec_loss(embedded)
        loss += cluster_loss * cluster_loss_weight

        return loss

    def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
            use_reconstruction_loss: bool, cluster_loss_weight: float) -> '_DEC_Module':
        """
        Trains the _DEC_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            the autoencoder
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        use_reconstruction_loss : bool
            defines whether the reconstruction loss will be used during clustering training
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss

        Returns
        -------
        self : _DEC_Module
            this instance of the _DEC_Module
        """
        for _ in range(n_epochs):
            for batch in trainloader:
                loss = self._loss(batch, autoencoder, cluster_loss_weight, use_reconstruction_loss, loss_fn, device)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self


class DEC(BaseEstimator, ClusterMixin):
    """
    The Deep Embedded Clustering (DEC) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DEC loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    alpha : float
        alpha value for the prediction (default: 1.0)
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
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 1)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
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
    dec_labels_ : np.ndarray
        The final DEC labels
    dec_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder

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
                 pretrain_optimizer_params: dict = {"lr": 1e-3}, clustering_optimizer_params: dict = {"lr": 1e-4},
                 pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, cluster_loss_weight: float = 1, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = {},
                 random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.cluster_loss_weight = cluster_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params
        self.random_state = check_random_state(random_state)
        self.use_reconstruction_loss = False
        set_torch_seed(self.random_state)

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
        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        kmeans_labels, kmeans_centers, dec_labels, dec_centers, autoencoder = _dec(X, self.n_clusters, self.alpha,
                                                                                   self.batch_size,
                                                                                   self.pretrain_optimizer_params,
                                                                                   self.clustering_optimizer_params,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.use_reconstruction_loss,
                                                                                   self.cluster_loss_weight,
                                                                                   self.custom_dataloaders,
                                                                                   self.augmentation_invariance,
                                                                                   self.initial_clustering_class,
                                                                                   self.initial_clustering_params,
                                                                                   self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dec_labels_ = dec_labels
        self.dec_cluster_centers_ = dec_centers
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
        dataloader = get_dataloader(X, self.batch_size, False, False)
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, self.autoencoder)
        return predicted_labels


class IDEC(DEC):
    """
    The Improved Deep Embedded Clustering (IDEC) algorithm.
    Is equal to the DEC algorithm but uses the reconstruction loss also during the clustering optimization.
    Further, cluster_loss_weight is set to 0.1 instead of 1 when using the default settings.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    alpha : float
        alpha value for the prediction (default: 1.0)
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
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 0.1)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
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
    dec_labels_ : np.ndarray
        The final DEC labels
    dec_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder

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
                 pretrain_optimizer_params: dict = {"lr": 1e-3},
                 clustering_optimizer_params: dict = {"lr": 1e-4}, pretrain_epochs: int = 100,
                 clustering_epochs: int = 150, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, cluster_loss_weight: float = 0.1, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = {},
                 random_state: np.random.RandomState = None):
        super().__init__(n_clusters, alpha, batch_size, pretrain_optimizer_params, clustering_optimizer_params,
                         pretrain_epochs, clustering_epochs, optimizer_class, loss_fn, autoencoder, embedding_size,
                         cluster_loss_weight, custom_dataloaders, augmentation_invariance,
                         initial_clustering_class, initial_clustering_params, random_state)
        self.use_reconstruction_loss = True

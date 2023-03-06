"""
@authors:
Lukas Miklautz,
Dominik Mautz
"""

from clustpy.deep._utils import detect_device, encode_batchwise, \
    squared_euclidean_distance, predict_batchwise
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._train_utils import get_trained_autoencoder
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dcn(X: np.ndarray, n_clusters: int, batch_size: int, pretrain_learning_rate: float,
         clustering_learning_rate: float, pretrain_epochs: int,
         clustering_epochs: int, optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
         autoencoder: torch.nn.Module, embedding_size: int, degree_of_space_distortion: float,
         degree_of_space_preservation: float, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DCN clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters
    batch_size : int
        size of the data batches
    pretrain_learning_rate : float
        learning rate for the pretraining of the autoencoder
    clustering_learning_rate : float
        learning rate of the actual clustering procedure
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    clustering_epochs : int
        number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        the optimizer class
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FlexibleAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    degree_of_space_distortion : float
        weight of the reconstruction loss
    degree_of_space_preservation : float
        weight of the clustering loss
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DCN after the training terminated,
        The cluster centers as identified by DCN after the training terminated,
        The final autoencoder
    """
    device = detect_device()
    trainloader = get_dataloader(X, batch_size, True, False)
    testloader = get_dataloader(X, batch_size, False, False)
    autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, X.shape[1], embedding_size, autoencoder)
    # Execute kmeans in embedded space
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    # Setup DCN Module
    dcn_module = _DCN_Module(init_centers).to_device(device)
    # Use DCN learning_rate (usually pretrain_learning_rate reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()), lr=clustering_learning_rate)
    # DEC Training loop
    dcn_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn,
                   degree_of_space_distortion, degree_of_space_preservation)
    # Get labels
    dcn_labels = predict_batchwise(testloader, autoencoder, dcn_module, device)
    dcn_centers = dcn_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dcn_labels, dcn_centers, autoencoder


def _compute_centroids(centers: torch.Tensor, embedded: torch.Tensor, count: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, torch.Tensor):
    """
    Update the centers and amount of object ever assigned to a center.

    New center is calculated by (see Eq. 8 in the paper):
    center - eta (center - embedded[i])
    => center - eta * center + eta * embedded[i]
    => (1 - eta) center + eta * embedded[i]

    Parameters
    ----------
    centers : torch.Tensor
        The current cluster centers
    embedded : torch.Tensor
        The embedded samples
    count : torch.Tensor
        The total amount of objects that ever got assigned to a cluster. Affects the learning rate of the center update
    labels : torch.Tensor
        The current hard labels

    Returns
    -------
    centers, count : (torch.Tensor, torch.Tensor)
        The updated centers and the updated counts
    """
    for i in range(embedded.shape[0]):
        c = labels[i].item()
        count[c] += 1
        eta = 1.0 / count[c].item()
        centers[c] = (1 - eta) * centers[c] + eta * embedded[i]
    return centers, count


class _DCN_Module(torch.nn.Module):
    """
    The _DCN_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    init_np_centers : np.ndarray
        The initial numpy centers

    Attributes
    ----------
    centers : torch.Tensor
        the cluster centers
    """

    def __init__(self, init_np_centers: np.ndarray):
        super().__init__()
        self.centers = torch.tensor(init_np_centers)

    def dcn_loss(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DCN loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance (default: None)

        Returns
        -------
        loss: torch.Tensor
            the final DCN loss
        """
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        loss = (dist.min(dim=1)[0]).mean()
        return loss

    def predict_hard(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the minimum squared euclidean distance to the cluster centers to get the labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance (default: None)

        Returns
        -------
        labels : torch.Tensor
            the final labels
        """
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        labels = (dist.min(dim=1)[1])
        return labels

    def update_centroids(self, embedded: torch.Tensor, count: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Update the cluster centers of the _DCN_Module.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        count : torch.Tensor
            The total amount of objects that ever got assigned to a cluster. Affects the learning rate of the center update
        labels : torch.Tensor
            The current hard labels

        Returns
        -------
        count : torch.Tensor
            The new amount of objects that ever got assigned to a cluster
        """
        self.centers, count = _compute_centroids(self.centers, embedded, count, labels)
        return count

    def to_device(self, device: torch.device) -> '_DCN_Module':
        """
        Move the _DCN_Module and the cluster centers to the specified device (cpu or cuda).

        Parameters
        ----------
        device : torch.device
            device to be trained on

        Returns
        -------
        self : _DCN_Module
            this instance of the _DCN_Module
        """
        self.centers = self.centers.to(device)
        self.to(device)
        return self

    def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
            degree_of_space_distortion: float, degree_of_space_preservation: float) -> '_DCN_Module':
        """
        Trains the _DCN_Module in place.

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
        degree_of_space_distortion : float
            weight of the clustering loss
        degree_of_space_preservation : float
            weight of the reconstruction loss

        Returns
        -------
        self : _DCN_Module
            this instance of the _DCN_Module
        """
        # DCN training loop
        # Init for count from original DCN code (not reported in Paper)
        # This means centroid learning rate at the beginning is scaled by a hundred
        count = torch.ones(self.centers.shape[0], dtype=torch.int32) * 100
        for _ in range(n_epochs):
            # Update Network
            for batch in trainloader:
                batch_data = batch[1].to(device)
                embedded = autoencoder.encode(batch_data)
                reconstruction = autoencoder.decode(embedded)
                # compute reconstruction loss
                ae_loss = loss_fn(batch_data, reconstruction)
                # compute cluster loss
                cluster_loss = self.dcn_loss(embedded)
                # compute total loss
                loss = degree_of_space_preservation * ae_loss + 0.5 * degree_of_space_distortion * cluster_loss
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update Assignments and Centroids
            with torch.no_grad():
                for batch in trainloader:
                    batch_data = batch[1].to(device)
                    embedded = autoencoder.encode(batch_data)

                    ## update centroids [on gpu] About 40 seconds for 1000 iterations
                    ## No overhead from loading between gpu and cpu
                    # count = cluster_module.update_centroid(embedded, count, s)

                    # update centroids [on cpu] About 30 Seconds for 1000 iterations
                    # with additional overhead from loading between gpu and cpu
                    embedded = embedded.cpu()
                    self.centers = self.centers.cpu()

                    # update assignments
                    labels = self.predict_hard(embedded)

                    # update centroids
                    count = self.update_centroids(embedded, count.cpu(), labels.cpu())
                    # count = count.to(device)
                    self.centers = self.centers.to(device)
        return self


class DCN(BaseEstimator, ClusterMixin):
    """
    The Deep Clustering Network (DCN) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterwards, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DCN loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    batch_size : int
        size of the data batches (default: 256)
    pretrain_learning_rate : float
        learning rate for the pretraining of the autoencoder (default: 1e-3)
    clustering_learning_rate : float
        learning rate of the actual clustering procedure (default: 1e-4)
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction (default: torch.nn.MSELoss())
    degree_of_space_distortion : float
        weight of the clustering loss (default: 0.05)
    degree_of_space_preservation : float
        weight of the reconstruction loss (default: 1.0)
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FlexibleAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dcn_labels_ : np.ndarray
        The final DCN labels
    dcn_cluster_centers_ : np.ndarray
        The final DCN cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    ----------
    from clustpy.data import load_mnist
    from clustpy.deep import DCN
    data, labels = load_mnist()
    dcn = DCN(n_clusters=10)
    dcn.fit(data)

    References
    ----------
    Yang, Bo, et al. "Towards k-means-friendly spaces:
    Simultaneous deep learning and clustering." international
    conference on machine learning. PMLR, 2017.
    """

    def __init__(self, n_clusters: int, batch_size: int = 256, pretrain_learning_rate: float = 1e-3,
                 clustering_learning_rate: float = 1e-4, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), degree_of_space_distortion: float = 0.05,
                 degree_of_space_preservation: float = 1.0, autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.random_state = check_random_state(random_state)
        torch.manual_seed(self.random_state.get_state()[1][0])

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DCN':
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
        self : DCN
            this instance of the DCN algorithm
        """
        kmeans_labels, kmeans_centers, dcn_labels, dcn_centers, autoencoder = _dcn(X, self.n_clusters, self.batch_size,
                                                                                   self.pretrain_learning_rate,
                                                                                   self.clustering_learning_rate,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.degree_of_space_distortion,
                                                                                   self.degree_of_space_preservation,
                                                                                   self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dcn_labels_ = dcn_labels
        self.dcn_cluster_centers_ = dcn_centers
        self.autoencoder = autoencoder
        return self

"""
@authors:
Lukas Miklautz,
Dominik Mautz
"""

from clustpy.deep._utils import encode_batchwise, squared_euclidean_distance, predict_batchwise, \
    set_torch_seed, int_to_one_hot, embedded_kmeans_prediction
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from clustpy.deep._data_utils import get_dataloader, augmentation_invariance_check
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dcn(X: np.ndarray, n_clusters: int, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss, autoencoder: torch.nn.Module,
         embedding_size: int, degree_of_space_distortion: float, degree_of_space_preservation: float,
         custom_dataloaders: tuple, augmentation_invariance: bool, initial_clustering_class: ClusterMixin,
         initial_clustering_params: dict,
         random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DCN clustering procedure on the input data set.

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
    degree_of_space_distortion : float
        weight of the reconstruction loss
    degree_of_space_preservation : float
        weight of the clustering loss
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
        The labels as identified by DCN after the training terminated,
        The cluster centers as identified by DCN after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, autoencoder, _, n_clusters, _, init_centers, _ = get_standard_initial_deep_clustering_setting(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, loss_fn, autoencoder,
        embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, random_state)
    # Setup DCN Module
    dcn_module = _DCN_Module(init_centers, augmentation_invariance).to_device(device)
    # Use DCN optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()), **clustering_optimizer_params)
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
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)


    Attributes
    ----------
    centers : torch.Tensor
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(self, init_np_centers: np.ndarray, augmentation_invariance: bool = False):
        super().__init__()
        self.augmentation_invariance = augmentation_invariance
        self.centers = torch.tensor(init_np_centers)

    def dcn_loss(self, embedded: torch.Tensor, assignment_matrix: torch.Tensor = None,
                 weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DCN loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        assignment_matrix : torch.Tensor
            cluster assignments per sample as a one-hot-matrix to compute the loss. 
            If None then loss will be computed based on the closest centroids for each data sample (default: None)
        weights : torch.Tensor
            feature weights for the squared euclidean distance (default: None)
        
        Returns
        -------
        loss: torch.Tensor
            the final DCN loss
        """
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        if assignment_matrix is None:
            loss = (dist.min(dim=1)[0]).mean()
        else:
            loss = (dist * assignment_matrix).mean()
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

    def _loss(self, batch: list, autoencoder: torch.nn.Module, loss_fn: torch.nn.modules.loss._Loss,
              degree_of_space_preservation: float, degree_of_space_distortion: float, device: torch.device):
        """
        Calculate the complete DCN + Autoencoder loss.

        Parameters
        ----------
        batch : list
            the minibatch
        autoencoder : torch.nn.Module
            the autoencoder
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        degree_of_space_distortion : float
            weight of the clustering loss
        degree_of_space_preservation : float
            weight of the reconstruction loss
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DCN loss
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
        assignments = self.predict_hard(embedded)
        assignment_matrix = int_to_one_hot(assignments, self.centers.shape[0])
        cluster_loss = self.dcn_loss(embedded, assignment_matrix)
        if self.augmentation_invariance:
            # assign augmented samples to the same cluster as original samples
            cluster_loss_aug = self.dcn_loss(embedded_aug, assignment_matrix)
            cluster_loss = (cluster_loss + cluster_loss_aug) / 2

        # compute total loss
        loss = degree_of_space_preservation * ae_loss + 0.5 * degree_of_space_distortion * cluster_loss

        return loss

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
                loss = self._loss(batch, autoencoder, loss_fn, degree_of_space_preservation, degree_of_space_distortion,
                                  device)
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update Assignments and Centroids
            with torch.no_grad():
                for batch in trainloader:
                    if self.augmentation_invariance:
                        # Convention is that the augmented sample is at the first position and the original one at the second position
                        # We only use the original sample for updating the centroids and assignments
                        batch_data = batch[2].to(device)
                    else:
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
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DCN loss function.

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
    degree_of_space_distortion : float
        weight of the clustering loss (default: 0.05)
    degree_of_space_preservation : float
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
    dcn_labels_ : np.ndarray
        The final DCN labels
    dcn_cluster_centers_ : np.ndarray
        The final DCN cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DCN
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> dcn = DCN(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> dcn.fit(data)

    References
    ----------
    Yang, Bo, et al. "Towards k-means-friendly spaces:
    Simultaneous deep learning and clustering." international
    conference on machine learning. PMLR, 2017.
    """

    def __init__(self, n_clusters: int, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 150, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), degree_of_space_distortion: float = 0.05,
                 degree_of_space_preservation: float = 1.0, autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 initial_clustering_class: ClusterMixin = KMeans, initial_clustering_params: dict = None,
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
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = {} if initial_clustering_params is None else initial_clustering_params
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)

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
        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        kmeans_labels, kmeans_centers, dcn_labels, dcn_centers, autoencoder = _dcn(X, self.n_clusters, self.batch_size,
                                                                                   self.pretrain_optimizer_params,
                                                                                   self.clustering_optimizer_params,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.degree_of_space_distortion,
                                                                                   self.degree_of_space_preservation,
                                                                                   self.custom_dataloaders,
                                                                                   self.augmentation_invariance,
                                                                                   self.initial_clustering_class,
                                                                                   self.initial_clustering_params,
                                                                                   self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dcn_labels_ = dcn_labels
        self.dcn_cluster_centers_ = dcn_centers
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

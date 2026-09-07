"""
@authors:
Pascal Weber and Peter Salah
"""

from __future__ import annotations
import numpy as np
import torch
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep._data_utils import get_train_and_test_dataloader
from clustpy.deep._train_utils import get_trained_network
from clustpy.deep._utils import (
    detect_device,
    encode_batchwise,
    mean_squared_error,
    run_initial_clustering,
)
from clustpy.utils.checks import check_parameters
from SHiP import SHiP
import tqdm
from typing import Callable, Optional
from sklearn.base import ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from scipy.stats import mode


class DeepSynC(_AbstractDeepClusteringAlgo):
    """
    The Deep Synchronisation Clustering (DeepSynC) algorithm.
    A neural network (autoencoder AE) will be trained with the reconstruction loss and the sync loss function.
    At the beginning, SHiP identifies an initial clustering of detected core points. Those clusterings then
    propagate their cluster labels to non-labeled data points, while the embedding is optimized with a sync loss.

    Parameters
    ----------
    clustering_class : ClusterMixin
        clustering class to obtain the cluster labels after getting the embedding (default: SHiP)
    clustering_params : dict
        parameters for the clustering class. (default: {"treeType": "DCTree", "hierarchy": 2, "partitioningMethod": "ThresholdElbow"}).
    batch_size : int
        Size of the data batches. (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate. (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate. (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network. (default: 100)
    clustering_max_epochs : int
        max number of epochs for the actual clustering procedure (default: 300)
    high_confidence_epochs: int
        number of epochs that a data point needs to get assigned the same label to be a data point with a high confidence label
        (default: 3)
    early_stopping_epochs: int
        number of epochs with no label changes after which the training should stop early
        (default: 3)
    k_nearest_neighbors: int
        number of k nearest neighbors to be considered in the initial core points detection
        (default: 25)
    percent_core_points: float
        find percent of total points as the initial core points
        (default: 0.1)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
        self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: mean_squared_error)
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
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers defined as the mean of assigned samples within the AE embedding
    neural_network_trained_ : torch.nn.Module
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

    Examples
    --------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DeepSynC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> deepsync = DeepSynC()
    >>> deepsync.fit(data)

    References
    ----------
    Deep Synchronisation-based Clustering
    Lena G. M. Bauer; Peter Salah; Pascal Weber; Anna Beer; Christian Böhm; Yllka Velaj; Claudia Plant
    IEEE International Conference on Data Mining (ICDM), Shenyang, China, 2026
    """

    def __init__(
        self,
        clustering_class: type[ClusterMixin] = SHiP,
        clustering_params: dict = {"treeType": "DCTree", "hierarchy": 2, "partitioningMethod": "ThresholdElbow"},
        batch_size: int = 256,
        pretrain_optimizer_params: dict = {"lr": 1e-3},
        clustering_optimizer_params: dict = {"lr": 1e-4},
        pretrain_epochs: int = 100,
        clustering_max_epochs: int = 300,
        high_confidence_epochs: int = 3,
        early_stopping_epochs: int = 3,
        k_nearest_neighbors: int = 25,
        percent_core_points: float = 0.1,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
        neural_network: torch.nn.Module | tuple = None,
        neural_network_weights: Optional[str] = None,
        embedding_size: int = 10,
        custom_dataloaders: Optional[tuple] = None,
        device: torch.device = None,
        random_state: Optional[np.random.RandomState | int] = None,
    ):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.clustering_class = clustering_class
        self.clustering_params = clustering_params
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_max_epochs = clustering_max_epochs
        self.high_confidence_epochs = high_confidence_epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.k_nearest_neighbors = k_nearest_neighbors
        self.percent_core_points = percent_core_points
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DeepSynC:
        """
        Cluster the input dataset with the DeepSynC algorithm.
        The resulting cluster labels will be stored in the `labels_` attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set.
        y : np.ndarray
            The labels. (can be ignored)

        Returns
        -------
        self : DeepSynC
            This instance of the DeepSynC algorithm.
        """
        X, _, random_state, pretrain_optimizer_params, _, _ = self._check_parameters(X, y=y)
        device = detect_device(self.device)
        trainloader, testloader, batch_size = get_train_and_test_dataloader(X, self.batch_size, self.custom_dataloaders)
        # Create and pretrain Autoencoder
        neural_network_params = {"layers": [X.shape[1], 512, 256, 128, self.embedding_size]}
        neural_network = get_trained_network(
            trainloader,
            n_epochs=self.pretrain_epochs,
            optimizer_params=pretrain_optimizer_params,
            optimizer_class=self.optimizer_class,
            device=device,
            ssl_loss_fn=self.ssl_loss_fn,
            embedding_size=self.embedding_size,
            neural_network=self.neural_network,
            neural_network_weights=self.neural_network_weights,
            neural_network_params=neural_network_params,
            random_state=random_state,
        )
        # Setup labels of initial core points
        assert (
            self.k_nearest_neighbors <= X.shape[0]
        ), f"WARNING: `k_nearest_neighbors` ({self.k_nearest_neighbors}) is larger than dataset size ({X.shape[0]})."

        embedded = encode_batchwise(testloader, neural_network)
        core_points_mask = _find_local_core_points_same(embedded, self.k_nearest_neighbors, self.percent_core_points)
        if not core_points_mask.any():
            core_points_mask[:] = True
        core_points = embedded[core_points_mask]
        n_clusters, core_points_labels, _cluster_centers, _ = run_initial_clustering(
            X=core_points,
            n_clusters=None,
            clustering_class=self.clustering_class,
            clustering_params=self.clustering_params,
            random_state=random_state,
        )
        initial_labels = np.full(X.shape[0], -1)
        initial_labels[core_points_mask] = core_points_labels

        # Setup DeepSynC module
        deepsync_module = _DeepSynC_Module(
            n_max_epochs=self.clustering_max_epochs,
            neural_network=neural_network,
            device=device,
            ssl_loss_fn=self.ssl_loss_fn,
            high_confidence_epochs=self.high_confidence_epochs,
            early_stopping_epochs=self.early_stopping_epochs,
            k_nearest_neighbors=self.k_nearest_neighbors,
        )
        optimizer = self.optimizer_class(list(neural_network.parameters()), **self.clustering_optimizer_params)
        deepsync_module.fit(trainloader, optimizer, initial_labels)
        labels = deepsync_module.labels_

        # Get labels
        self.n_clusters_ = n_clusters
        self.labels_ = np.array(labels, dtype=np.int32)
        embedded = encode_batchwise(testloader, neural_network)
        self.cluster_centers_ = np.array([np.mean(embedded[labels == i], axis=0) for i in np.unique(labels) if i >= 0])
        self.neural_network_trained_ = neural_network
        self.set_n_featrues_in(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.
        Note that this is just a very imprecise estimation as we are not using the DeepSynC method to predict the labels.
        The prediction is calculated by checking the distance to the clostest mean of samples in a cluster within the embedding of the AE.

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        check_is_fitted(self, ["labels_", "neural_network_trained_", "n_features_in_"])
        X, _, _ = check_parameters(
            X, allow_size_1=True, allow_nd=self.neural_network_trained_.allow_nd_input, estimator_obj=self
        )
        print(
            "WARNING: predict does not use the embedding of the manifold and is, therefore, just a very rough estimate"
        )
        predicted_labels = super().predict(X)
        return predicted_labels


class _DeepSynC_Module(torch.nn.Module):
    """
    The _DeepSynC_Module. Contains most of the algorithm specific procedures like the loss function.

    Parameters
    ----------
    n_max_epochs : int
        max number of epochs for the clustering procedure
    neural_network : torch.nn.Module
        the neural network
    device : torch.device
        device to be trained on
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
        self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    high_confidence_epochs: int
        number of epochs that a data point needs to get assigned the same label to be a data point with a high confidence label
        (default: 3)
    early_stopping_epochs: int
        number of epochs with no label changes after which the training should stop early
        (default: 3)
    k_nearest_neighbors: int
        number of k nearest neighbors to be considered in the initial core points detection
        (default: 25)
    """

    def __init__(
        self,
        n_max_epochs: int,
        neural_network: torch.nn.Module,
        device: torch.device,
        ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
        high_confidence_epochs: int,
        early_stopping_epochs: int,
        k_nearest_neighbors: int,
    ):
        super().__init__()
        self.n_max_epochs = n_max_epochs
        self.neural_network = neural_network
        self.device = device
        self.ssl_loss_fn = ssl_loss_fn
        self.high_confidence_epochs = high_confidence_epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.k_nearest_neighbors = k_nearest_neighbors

    def fit(
        self,
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        initial_labels: np.ndarray,
    ):
        """
        Trains the _DeepSynC_Module in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        optimizer : torch.optim.Optimizer
            the optimizer for training
        initial_labels: np.ndarray
            the initial labels of the detected core points

        Returns
        -------
        self : _DeepSynC_Module
            This instance of the _DeepSynC_Module.
        """
        temporary_labels = np.zeros((1, len(trainloader.dataset)))
        temporary_labels[0, :] = initial_labels.copy()
        no_new_labels_epochs = 0

        self.train()
        tbar = tqdm.trange(self.n_max_epochs, desc="DeepSynC training")
        for epoch in tbar:
            # Update Network
            for batch in trainloader:
                loss = self._loss(batch, temporary_labels[0, :])
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Loss": loss}
            tbar.set_postfix(postfix_str)

            # Label Assignment
            train_embedded_data = encode_batchwise(trainloader, self.neural_network)

            # find high confidence points
            high_confidence_labels = np.full(temporary_labels.shape[1], True, dtype=bool)
            high_confidence_labels[temporary_labels[0, :] < 0] = False  # not assigned points always low confidence
            if epoch >= self.high_confidence_epochs:  # check last high_conf_epochs for same labels
                for idx in range(temporary_labels.shape[0] - 1):
                    check = temporary_labels[idx, :] == temporary_labels[idx + 1, :]
                    high_confidence_labels = np.logical_and(high_confidence_labels, check)

            current_epoch_labels = temporary_labels[0, :].copy()
            previous_epoch_labels = temporary_labels[0, :]

            # 1 - consider points with low confidence as -1 so that they can be assigned to other clusters
            current_epoch_labels[~high_confidence_labels] = -1
            current_epoch_labels = _knn_assign_unlabeled_points(
                train_embedded_data, current_epoch_labels, self.k_nearest_neighbors
            )

            # 2 - prevent high confidence labels from changing
            current_epoch_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]

            # 3 - prevent labeled points from getting unlabeled
            prev_labeled_mask = np.logical_and(current_epoch_labels == -1, previous_epoch_labels != -1)
            current_epoch_labels[prev_labeled_mask] = previous_epoch_labels[prev_labeled_mask]

            # 4 - Store the labels
            temporary_labels = np.vstack((current_epoch_labels, temporary_labels))
            if temporary_labels.shape[0] > self.high_confidence_epochs:
                temporary_labels = temporary_labels[:-1, :]

            # 5 - Early Stopping
            if np.all(high_confidence_labels):
                print(f"Early stopping after {epoch} iterations, all points are labeled confidently.")
                break

            if (current_epoch_labels == previous_epoch_labels).all():
                no_new_labels_epochs += 1
            else:
                no_new_labels_epochs = 0
            if no_new_labels_epochs >= self.early_stopping_epochs:
                print(
                    f"Early stopping after {epoch} iterations, the algorithm is not assigning any new points for {self.early_stopping_epochs} iterations."
                )
                break
        self.labels_ = temporary_labels[0, :]
        self.neural_network.eval()
        self.eval()
        return self

    def _loss(
        self,
        batch: list,
        current_labels: np.ndarray,
    ) -> torch.Tensor:
        """
        Calculate the autoencoder reconstruction + sync loss.

        Parameters
        ----------
        X : np.ndarray
            The data.
        batch : list
            The minibatch.
        current_labels : list
            Current labels of the data points.

        Returns
        -------
        loss : torch.Tensor
            The final DeepSynC loss.
        """
        # Reconstruction
        ssl_loss, embedded, _ = self.neural_network.loss(batch, self.ssl_loss_fn, self.device)

        # Sync Loss
        idxs = batch[0].to(self.device)
        current_batch_labels = current_labels[idxs]

        def get_scaled_outlier_dists(out_dists):
            # input: distance matrix of a batch masked for the distances of outliers
            # output: scaled distances for outliers based on distance
            max_vec = np.max(out_dists, 1)
            s = max_vec / 4
            s = np.transpose(np.repeat([s], out_dists.shape[0], axis=0))
            x = np.multiply(out_dists, 1 / (s + 1e9))
            return np.exp(-1 / 2 * np.power(x, 2))

        squared_diffs = (embedded.unsqueeze(0) - embedded.unsqueeze(1)).pow(2).sum(2)
        squared_diffs_cpu = squared_diffs.detach().cpu().numpy()
        n = len(current_batch_labels)
        outliers = np.eye(n)
        for j in range(0, n):
            if current_batch_labels[j] < 0:
                outliers[:, j] = 1
                outliers[j, :] = 1
        outlier_dists = squared_diffs_cpu * outliers
        scaled_weights = get_scaled_outlier_dists(outlier_dists)
        for k in range(0, n):
            if current_batch_labels[k] >= 0:
                for l in range(0, n):
                    if current_batch_labels[l] >= 0:
                        if current_batch_labels[k] == current_batch_labels[l]:
                            scaled_weights[k, l] = 1
                            scaled_weights[l, k] = 1
                        else:
                            scaled_weights[k, l] = 0
                            scaled_weights[l, k] = 0
        scaled_weights_torch = torch.from_numpy(scaled_weights)
        sync_loss = 1 - (1 / n**2) * (torch.exp(-squared_diffs * scaled_weights_torch)).sum(0).sum()

        # Total loss
        loss = ssl_loss + sync_loss
        return loss


def _find_local_core_points_same(X, k_nearest_neighbors, percent_core_points):
    # precomputes the k-th nearest neighbor distance (core_dist / kappa) and reuses this within every `percent_core_points` nearest neighbor subset
    subset = int(np.floor(X.shape[0] * percent_core_points))
    p_dist = pairwise_distances(X, metric="euclidean")
    core_dists = np.partition(p_dist, k_nearest_neighbors - 1, axis=0)[k_nearest_neighbors - 1]
    nn = np.argpartition(p_dist, subset, axis=1)[:, :subset]
    refined_medians = np.median(core_dists[nn], axis=1)
    return core_dists < refined_medians


def _knn_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k_nearest_neighbors):
    unlabeled_points_mask = current_epoch_labels < 0
    unlabeled_points = train_embedded_data[unlabeled_points_mask, :]
    if len(unlabeled_points) == 0:
        return current_epoch_labels
    knn_for_labelling = NearestNeighbors(n_neighbors=k_nearest_neighbors, metric="euclidean").fit(train_embedded_data)
    indices_for_labelling = knn_for_labelling.kneighbors(unlabeled_points, return_distance=False)
    labels_of_neighbors = current_epoch_labels[indices_for_labelling]
    most_common_labels = mode(labels_of_neighbors, axis=1, keepdims=False)
    new_labels = most_common_labels.mode
    current_epoch_labels[unlabeled_points_mask] = new_labels
    return current_epoch_labels

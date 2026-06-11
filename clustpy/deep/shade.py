"""
@authors:
Pascal Weber
"""

from __future__ import annotations
import numpy as np
import torch
from clustpy.utils import DCTree
from clustpy.hierarchical import DCTree_Clusterer
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep._data_utils import get_train_and_test_dataloader
from clustpy.deep._train_utils import get_trained_network
from clustpy.deep._utils import detect_device, squared_euclidean_distance, encode_batchwise, run_initial_clustering, mean_squared_error
from clustpy.utils.checks import check_parameters
import tqdm
from typing import Callable, Optional, Tuple
from sklearn.utils.validation import check_is_fitted
from sklearn.base import ClusterMixin


class SHADE(_AbstractDeepClusteringAlgo):
    """
    The Structure-preserving High-dimensional Analysis with Density-based Exploration (SHADE) algorithm.
    A neural network (autoencoder AE) will be trained with the reconstruction loss and the d_dc loss function.
    Afterward, KMeans or HDBSCAN identifies the initial clusters.

    Parameters
    ----------
    clustering_class : ClusterMixin
        clustering class to obtain the cluster labels after getting the embedding (default: DCTree_Clusterer)
    clustering_params : dict
        parameters for the clustering class. If None, it will be set to {"min_points": min_points} (default: None)
    min_points : int
        the minimum number of points (default: 5)
    use_complete_dc_tree : bool
        Defines whether the complete DC Tree should be used instead of a batch-wise version (default: True)
    use_matrix_dc_distance: bool
        Defines whether the matrix DC distance should be stored - can cause memory issues (default: True)
    use_less_memory: bool
      Use less memory when constructing the DCTree.
      This will, however, increase the runtime (default: False)
    batch_size : int
        Size of the data batches. (default: 500)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate.
        If None, it will be set to {"lr": 1e-3}. (default: None)
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate. If None, it will be set to {"lr": 1e-4} (default: None)
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network. (default: 0)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 100)
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
    density_loss_weight : float
        weight of the density loss compared to the reconstruction loss (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
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
    dc_tree_ : DCTree
        The dc tree
    neural_network_trained_ : torch.nn.Module
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

    Examples
    --------
    >>> from clustpy.data import create_subspace_data
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> shade = SHADE()
    >>> shade.fit(data)

    References
    ----------
    SHADE: Deep Density-based Clustering
    Anna Beer; Pascal Weber; Lukas Miklautz; Collin Leiber; Walid Durani; Christian Böhm
    IEEE International Conference on Data Mining (ICDM), Abu Dhabi, United Arab Emirates, 2024, pp. 675-680, doi: 10.1109/ICDM59182.2024.
    """

    def __init__(
        self,
        clustering_class : Optional[ClusterMixin] = DCTree_Clusterer,
        clustering_params : dict = None,
        min_points : int = 5,
        use_complete_dc_tree: bool = True,
        use_matrix_dc_distance: bool = True,
        use_less_memory: bool = False,
        batch_size: int = 500,
        pretrain_optimizer_params: dict = None,
        clustering_optimizer_params : dict = None,
        pretrain_epochs : int = 0,
        clustering_epochs : int = 100,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss = mean_squared_error,
        neural_network : torch.nn.Module | tuple = None,
        neural_network_weights : str = None,
        embedding_size : int = 10,
        density_loss_weight : float = 1.0,
        ssl_loss_weight : float = 1.0,
        custom_dataloaders : tuple = None,
        device : torch.device = None,
        random_state : np.random.RandomState | int = None,
    ):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.clustering_class = clustering_class
        self.clustering_params = clustering_params
        self.min_points = min_points
        self.use_complete_dc_tree = use_complete_dc_tree
        self.use_matrix_dc_distance = use_matrix_dc_distance
        self.use_less_memory = use_less_memory
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.density_loss_weight = density_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray=None) -> SHADE:
        """
        Cluster the input dataset with the SHADE algorithm.
        The resulting cluster labels will be stored in the `labels_` attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set.
        y : np.ndarray
            The labels. (can be ignored)

        Returns
        -------
        self : SHADE
            This instance of the SHADE algorithm.
        """
        X, _, random_state, pretrain_optimizer_params, _, _ = self._check_parameters(X, y=y)
        clustering_optimizer_params = {"lr": 1e-3} if self.clustering_optimizer_params is None else self.clustering_optimizer_params
        clustering_params = {"min_points": self.min_points, "use_less_memory": self.use_less_memory} if self.clustering_params is None else self.clustering_params
        device = detect_device(self.device)
        trainloader, testloader, batch_size = get_train_and_test_dataloader(X, self.batch_size, self.custom_dataloaders)
        assert batch_size >= self.min_points, f"Batch_size ({batch_size}) cannot be smaller than min_points ({self.min_points})"
        # Create dc_tree
        if self.use_complete_dc_tree:
            self.dc_tree_ = DCTree(X, min_points=self.min_points, use_less_memory=self.use_less_memory)
        else:
            self.dc_tree_ = None
        # Create and pretrain Autoencoder
        neural_network_params = {"layers": [X.shape[1], 512, 256, 128, self.embedding_size]}
        neural_network = get_trained_network(trainloader, n_epochs=self.pretrain_epochs,
                                            optimizer_params=pretrain_optimizer_params, optimizer_class=self.optimizer_class,
                                            device=device, ssl_loss_fn=self.ssl_loss_fn, embedding_size=self.embedding_size,
                                            neural_network=self.neural_network,
                                            neural_network_weights=self.neural_network_weights, neural_network_params=neural_network_params,
                                            random_state=random_state)
        # Setup SHADE Module
        shade_module = _SHADE_Module(
            n_epochs=self.clustering_epochs,
            neural_network=neural_network,
            min_points=self.min_points,
            dc_tree=self.dc_tree_,
            use_matrix_dc_distance=self.use_matrix_dc_distance,
            device=device,
            ssl_loss_fn=self.ssl_loss_fn,
            density_loss_weight=self.density_loss_weight,
            ssl_loss_weight=self.ssl_loss_weight
        )
        optimizer = self.optimizer_class(list(neural_network.parameters()), **clustering_optimizer_params)
        shade_module.fit(X, trainloader, optimizer)
        # Get labels
        embedded_data = encode_batchwise(testloader, neural_network)
        n_clusters, labels, cluster_centers, _ = run_initial_clustering(
            X=embedded_data,
            n_clusters=None,
            clustering_class=self.clustering_class,
            clustering_params=clustering_params,
            random_state=random_state,
        )
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = cluster_centers
        self.neural_network_trained_ = neural_network
        self.set_n_featrues_in(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.
        Note that this is just a very imprecise estimation as we are not using the DC Tree to predict the labels.
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
        X, _, _ = check_parameters(X, allow_size_1=True, allow_nd=self.neural_network_trained_.allow_nd_input, estimator_obj=self)
        print("WARNING: predict does not use the embedding of the manifold and is, therefore, just a very rough estimate")
        predicted_labels = super().predict(X)
        return predicted_labels


class _SHADE_Module(torch.nn.Module):
    """
    The _SHADE_Module. Contains most of the algorithm specific procedures like the loss function.

    Parameters
    ----------
    n_epochs : int
        number of epochs for the clustering procedure
    neural_network : torch.nn.Module
        the neural network
    min_points : int
        the minimum number of points
    dc_tree : Optional[DCTree]
        the DCTree
    use_matrix_dc_distance: bool
        Defines whether the matrix DC distance should be stored - can cause memory issues
    device : torch.device
        device to be trained on
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
        self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    density_loss_weight : float
        weight of the clustering loss
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss
    """

    def __init__(
        self,
        n_epochs : int,
        neural_network: torch.nn.Module,
        min_points: int,
        dc_tree: Optional[DCTree],
        use_matrix_dc_distance: bool,
        device: torch.device,
        ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
        density_loss_weight: float,
        ssl_loss_weight: float
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.neural_network = neural_network
        self.min_points = min_points
        self.dc_tree = dc_tree
        self.use_matrix_dc_distance = use_matrix_dc_distance
        self.device = device
        self.ssl_loss_fn = ssl_loss_fn
        self.density_loss_weight = density_loss_weight
        self.ssl_loss_weight = ssl_loss_weight

    def fit(
        self,
        X: np.ndarray,
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> _SHADE_Module:
        """
        Trains the _SHADE_Module in place.

        Parameters
        ----------
        X : np.ndarray
            The data
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        optimizer : torch.optim.Optimizer
            the optimizer for training

        Returns
        -------
        self : _SHADE_Module
            This instance of the _SHADE_Module.
        """
        if self.dc_tree is not None and self.use_matrix_dc_distance:
            matrix_dc_distance = self.dc_tree.dc_distances()
            matrix_dc_distance_torch = torch.tensor(matrix_dc_distance, device=self.device)
        else:
            matrix_dc_distance_torch = None
        self.train()
        tbar = tqdm.trange(self.n_epochs, desc="SHADE training")
        for _ in tbar:
            # Update Network
            for batch in trainloader:
                if len(batch[0]) <= self.min_points:
                    continue
                loss = self._loss(X, batch, matrix_dc_distance_torch)
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Loss": loss}
            tbar.set_postfix(postfix_str)
        self.neural_network.eval()
        self.eval()
        return self

    def _loss(
        self,
        X: np.ndarray,
        batch: list,
        matrix_dc_distance_torch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the autoencoder reconstruction + d_dc loss.

        Parameters
        ----------
        X : np.ndarray
            The data
        batch : list
            The minibatch.
        matrix_dc_distance_torch : torch.Tensor
            A matrix containing pairwise dc distances

        Returns
        -------
        loss : torch.Tensor
            The final SHADE loss.
        """
        # Reconstrucion
        ssl_loss, embedded, _ = self.neural_network.loss(batch, self.ssl_loss_fn, self.device)
        # Density loss
        if self.dc_tree is None:
            # Batch-wise DCTree
            dc_distances = DCTree(X[batch[0]], min_points=self.min_points).dc_distances()
            batch_dc_dists = torch.tensor(dc_distances, device=self.device)
        else:
            # DCTree of all data points X
            if self.use_matrix_dc_distance:
                idxs = batch[0].to(self.device)
                batch_dc_dists = matrix_dc_distance_torch[idxs[:, None], idxs[None, :]]
            else:
                dc_distances = self.dc_tree.dc_distances(batch[0], batch[0])
                batch_dc_dists = torch.tensor(dc_distances, device=self.device)
        batch_eucl_dists = squared_euclidean_distance(embedded, embedded)
        loss_dens = (batch_eucl_dists - batch_dc_dists).pow(2).mean()
        loss = self.ssl_loss_weight * ssl_loss + self.density_loss_weight * loss_dens
        return loss

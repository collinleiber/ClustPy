"""
@authors:
Collin Leiber
"""

import torch
import numpy as np
from clustpy.deep._utils import detect_device, mean_squared_error
from clustpy.deep._encoding_utils import encode_batchwise
from clustpy.deep._data_utils import get_train_and_test_dataloader
from clustpy.deep._train_utils import get_neural_network
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections.abc import Callable
from pathlib import Path
from clustpy.deep.neural_networks._abstract_neural_network import _AbstractNeuralNetwork


class DEN(_AbstractDeepClusteringAlgo):
    """
    The Deep Embedding Network (DEN) algorithm.
    It trains a neural network by optimizing a loss functions consisting of three components.
    These are (1) the standrad loss function of the neural netork (e.g. reconstruction loss for autoencoders), (2) the locality-preserving constraint and (3) the group sparsity constraint.
    Finally, k-Means is excuted in the resulting embedding.

    Parameters
    ----------
    n_clusters : int
        number of clusters (default: 8)
    group_size : int | list | None
        the number of features in each group. Can also be a list, specifying the size of each group separately. Can be None if embedding_size is specified (default: 2)
    n_neighbors : int
        the number of nearest-neighbors (including itself) for the locality-preserving constraint. Nearest-neighbors will be calculated by using the Euclidean distance.
        If another distance should be used to define the nearest-neighbors, the neighbors can be included in the custom_dataloader as additional_inputs.
        In this case, it is expected that the trainloader is composed of: (sample_ids, original_samples, 1st-NNs, 2nd-NNs, ..., (n_neighbors-1)-NNs) (default: 5)
    weight_locality_constraint : float
        weight alpha for the locality-preserving constraint (default: 0.5)
    weight_sparsity_constraint : float
        weight beta for the group sparsity constraint (default: 1.)
    heat_kernel_t_parameter : float
        the t parameter for the heat kernel included in the locality-preserving constraint (default: 1.)
    group_lasso_lambda_parameter : float
        the lambda parameter for the group lasso included in the group sparsity constraint (default: 1.)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict | None
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate. If None, it will be set to {"lr": 1e-3} (default: None)
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    optimizer_class : type[torch.optim.Optimizer]
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: mean_squared_error)
    neural_network : _AbstractNeuralNetwork | tuple[type[_AbstractNeuralNetwork], dict] | None
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str | Path | None
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network. If it is set to zero, it will be automatically updated using the group_size parameter (default: 0)
    custom_dataloaders : tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader] | tuple[str | Path, str | Path] | None
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    device : torch.device | int | str | None
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int | None
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by KMeans)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by KMeans)
    neural_network_trained_ : _AbstractNeuralNetwork
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DEN
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> den = DEN(n_clusters=3, pretrain_epochs=3)
    >>> den.fit(data)

    References
    ----------
    Huang, Peihao, et al. "Deep embedding network for clustering."
    2014 22nd International conference on pattern recognition. IEEE, 2014.
    """

    def __init__(self, n_clusters: int = 8, group_size : int | list | None = 2, n_neighbors: int = 5, weight_locality_constraint: float = 0.5, 
                 weight_sparsity_constraint: float = 1., heat_kernel_t_parameter: float = 1., group_lasso_lambda_parameter: float = 1.,
                 batch_size: int = 256, pretrain_optimizer_params: dict | None = None,
                 pretrain_epochs: int = 100, optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
                 ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
                 neural_network: _AbstractNeuralNetwork | tuple[type[_AbstractNeuralNetwork], dict] | None = None,
                 neural_network_weights: str | Path | None = None, embedding_size: int = 0,
                 custom_dataloaders: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader] | tuple[str | Path, str | Path] | None = None,
                 device: torch.device | int | str | None = None, random_state: np.random.RandomState | int | None = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters = n_clusters
        self.group_size = group_size
        self.n_neighbors = n_neighbors
        self.weight_locality_constraint = weight_locality_constraint
        self.weight_sparsity_constraint = weight_sparsity_constraint
        self.heat_kernel_t_parameter = heat_kernel_t_parameter
        self.group_lasso_lambda_parameter = group_lasso_lambda_parameter
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.custom_dataloaders = custom_dataloaders


    def _check_group_size_and_embedding_size(self) -> tuple[list, int]:
        """
        Check if the values for group_size and embedding_size match.

        Returns
        -------
        tuple : tuple[list, int]
            the size of each group,
            the embedding size
        """
        assert (type(self.group_size) is list and np.sum(self.group_size) == self.embedding_size) or (type(self.group_size) is int and self.group_size * self.n_clusters == self.embedding_size) or (self.group_size is None and self.embedding_size != 0) or (self.embedding_size == 0 and self.group_size is not None), "Either group_size must be None, embedding_size == 0 or group_size must be set in accordance to the embedding size. You set group_size = {0} and embedding_size = {1}".format(self.group_size, self.embedding_size)
        if self.embedding_size == 0:
            group_size = self.group_size
            if isinstance(group_size, int):
                group_size = [group_size] * self.n_clusters
            assert type(group_size) is list, "group_size must be of type int or list. Your input: {0} / type: {1}".format(group_size, type(group_size))
            embedding_size = np.sum(group_size)
        else:
            assert self.embedding_size >= self.n_clusters, "embedding_size can not be smaller than n_clusters"
            embedding_size = self.embedding_size
            group_size_np = np.array([embedding_size // self.n_clusters] * self.n_clusters)
            group_size_np[: embedding_size % self.n_clusters] += 1
            group_size = group_size_np.tolist()
        assert isinstance(group_size, list) and len(group_size) == self.n_clusters, "group_size must have n_clusters entries"
        return group_size, embedding_size


    def _locality_preserving_loss(self, batch: list, embedded: torch.Tensor, neural_network: _AbstractNeuralNetwork, device: torch.device) -> torch.Tensor:
        """
        Calculate the DEN locality preserving loss of given embedded samples.

        Parameters
        ----------
        batch : list
            the minibatch
        embedded : torch.Tensor
            the embedded samples
        neural_network : _AbstractNeuralNetwork
            the neural network
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the DEN locality preserving loss
        """
        locality_preserving_loss = torch.tensor(0.)
        samples = batch[1].to(device)
        for i in range(self.n_neighbors - 1):  # TODO: Maybe use functorch.vmap in the future for vectorization
            neighbors = batch[2 + i].to(device)
            embedded_neighbor = neural_network.encode(neighbors)
            embedded_diff = (embedded - embedded_neighbor).pow(2).sum(1)
            orig_diff = (samples - neighbors).pow(2).sum(1)
            heat_kernel = torch.exp(-orig_diff / self.heat_kernel_t_parameter)
            locality_preserving_loss = locality_preserving_loss + (heat_kernel * embedded_diff).sum()
        return locality_preserving_loss / embedded.shape[0]


    def _group_sparsity_loss(self, embedded: torch.Tensor, group_size: list) -> torch.Tensor:
        """
        Calculate the DEN group sparsity loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        group_size : list
            the size of each group

        Returns
        -------
        loss : torch.Tensor
            the DEN group sparsity loss
        """
        group_sparsity_loss = torch.tensor(0.)
        group_index = 0
        for g in range(self.n_clusters):
            group_units = embedded[:, group_index:group_index+group_size[g]]
            group_units_length = (group_units.pow(2) + 1e-10).sum(1).sqrt()
            group_lasso_loss = self.group_lasso_lambda_parameter * torch.sqrt(torch.tensor(group_size[g])) * group_units_length
            group_sparsity_loss = group_sparsity_loss + group_lasso_loss.sum()
            # raise group index
            group_index += group_size[g]
        return group_sparsity_loss / embedded.shape[0]
    

    def _loss(self, batch: list, group_size: list, neural_network: _AbstractNeuralNetwork, device: torch.device):
        """
        Calculate the complete DEN + neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        group_size : list
            the size of each group
        neural_network : _AbstractNeuralNetwork
            the neural network
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DEN loss
        """
        # Calculate ssl loss
        ssl_loss, embedded = neural_network.loss(batch, self.ssl_loss_fn, device)
        # Calculate locality-preserving constraint
        locality_preserving_loss = self._locality_preserving_loss(batch, embedded, neural_network, device)
        # Calculate group sparsity constraint
        group_sparsity_loss = self._group_sparsity_loss(embedded, group_size)
        loss = ssl_loss + self.weight_locality_constraint * locality_preserving_loss + self.weight_sparsity_constraint * group_sparsity_loss
        return loss


    def _get_nearest_neighbors(self, X: np.ndarray) -> list:
        """
        Get a list containing the nearest neighbors of each entry in X.
        The list contains the actual data points, not the data indices.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        nearest_neigbors : list
            list containing the nearest neighbors of each entry in X
        """
        nearest_neigbors = []
        neighbors = NearestNeighbors(n_neighbors=self.n_neighbors)
        neighbors.fit(X)
        nearest_neighbors_ids = neighbors.kneighbors(n_neighbors=self.n_neighbors - 1, return_distance=False)
        for i in range(self.n_neighbors - 1):
            nearest_neigbors.append(X[nearest_neighbors_ids[:, i]])
        return nearest_neigbors


    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'DEN':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray | None
            the labels (can be ignored)

        Returns
        -------
        self : DEN
            this instance of the DEN algorithm
        """
        assert self.n_neighbors > 0, "n_neigbors must be larger than 0"
        X, _, random_state, pretrain_optimizer_params, _, _ = self._check_parameters(X, y=y)
        group_size, embedding_size = self._check_group_size_and_embedding_size()
        # Get the device to train on and the dataloaders
        device = detect_device(self.device)
        if self.custom_dataloaders is None:
            nearest_neighbors = self._get_nearest_neighbors(X)
        trainloader, testloader, _ = get_train_and_test_dataloader(X, self.batch_size, self.custom_dataloaders, 
                                                                   additional_inputs_trainloader=nearest_neighbors if self.custom_dataloaders is None else None)
        # Check that the trainloader includes neighbors -> must contain n_neighbors + 1 (the ids) entries
        assert len(next(iter(trainloader))) >= self.n_neighbors + 1, "Trainloader does not appear to include any neighbors."
        # Get AE
        neural_network = get_neural_network(input_dim=X.shape[1], embedding_size=embedding_size, 
                                            neural_network=self.neural_network, neural_network_weights=self.neural_network_weights, 
                                            device=device, random_state=random_state)
        optimizer = self.optimizer_class(neural_network.parameters(), **pretrain_optimizer_params)
        # DEN training loop
        tbar = tqdm.trange(self.pretrain_epochs, desc="DEN training")
        for _ in tbar:
            # Update Network
            total_loss = 0
            for batch in trainloader:
                loss = self._loss(batch, group_size, neural_network, device)
                total_loss += loss.item()
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Loss": total_loss}
            tbar.set_postfix(postfix_str)
        # Execute clustering with Kmeans
        embedded_data = encode_batchwise(testloader, neural_network)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        kmeans.fit(embedded_data)
        # Save parameters
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.neural_network_trained_ = neural_network
        self.set_n_features_in(X)
        return self

"""
@authors:
Collin Leiber,
Julian Schilcher
"""

import numpy as np
import torch
from clustpy.deep._utils import squared_euclidean_distance, encode_batchwise, predict_batchwise, \
    embedded_kmeans_prediction
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
from sklearn.cluster import KMeans
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.hierarchical._cluster_tree import BinaryClusterTree, _ClusterTreeNode
import tqdm
import copy


class _DeepECT_ClusterTreeNode(_ClusterTreeNode):

    def set_center_weight_and_torch_labels(self, center: np.ndarray, weight: float, optimizer: torch.optim.Optimizer,
                                           device: torch.device) -> None:
        """
        Set the cluster center and cluster weight for this node.
        Furthermore, create a copy of the labels as torch tensor that is saved on the specified device.

        Parameters
        ----------
        center : np.ndarray
            The cluster center
        weight : float
            The cluster weight
        optimizer : torch.optim.Optimizer
            Optimizer for training
        device : torch.device
            device to be trained on
        """
        self.center = torch.nn.Parameter(torch.tensor(center).to(device), requires_grad=True)
        self.weight = weight
        optimizer.add_param_group({"params": self.center})
        self.torch_labels = torch.tensor(self.labels, dtype=torch.int32).to(device)

    def update_parents_torch_labels(self, device: torch.device) -> None:
        """
        Update the torch_labels parameter of parent nodes.
        Has to be called when a new node has been added or a node has been deleted.

        Parameters
        ----------
        device : torch.device
            device to be trained on
        """
        parent_node_to_update = self.parent_node
        while parent_node_to_update is not None:
            new_torch_labels = torch.tensor(parent_node_to_update.labels, dtype=torch.int32).to(device)
            if hasattr(parent_node_to_update, "torch_labels") and torch.equal(parent_node_to_update.torch_labels,
                                                                              new_torch_labels):
                # Torch labels were already updated
                break
            parent_node_to_update.torch_labels = new_torch_labels
            parent_node_to_update = parent_node_to_update.parent_node


class _DeepECT_Module(torch.nn.Module):
    """
    The _DeepECT_Module. Contains most of the algorithm specific procedures like the loss and tree-grow functions.

    Parameters
    ----------
    cluster_tree: BinaryClusterTree
        The cluster tree
    max_n_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree
    grow_interval : int
        Number of epochs after which the the tree is grown
    pruning_threshold : float
        The threshold for pruning the tree
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    """

    def __init__(self, cluster_tree: BinaryClusterTree, max_n_leaf_nodes: int, grow_interval: int,
                 pruning_threshold: float, augmentation_invariance: bool = False):
        super().__init__()
        # Create initial cluster tree
        self.cluster_tree = cluster_tree
        self.max_n_leaf_nodes = max_n_leaf_nodes
        self.grow_interval = grow_interval
        self.pruning_threshold = pruning_threshold
        self.augmentation_invariance = augmentation_invariance

    def predict_hard(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the minimum squared Euclidean distance to the cluster centers of the leaf nodes to get the labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples

        Returns
        -------
        labels : torch.Tensor
            the final labels
        """
        leaf_nodes, _ = self.cluster_tree.get_leaf_and_split_nodes()
        device = leaf_nodes[0].center.device
        _, _, labels = self._get_labels_from_leafs(embedded.to(device), leaf_nodes)
        labels = labels.detach().cpu()
        return labels

    def _get_labels_from_leafs(self, embedded: torch.Tensor, leaf_nodes: list) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Get the cluster assignments of the current batch by considering the distance to the closest center of a leaf node.
        The assignment of a sample to a cluster center is represented by the index of the center and by the actual label of the the assigned leaf node.
        These values usually differ and both values are returned.

        Parameters
        ----------
        embedded : torch.Tensor
            The embedded batch of data
        leaf_nodes : list
            list containing all leaf nodes within the cluster tree

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor)
            The centers of the leaf nodes,
            The index of the cluster center assigned to each sample,
            The labels of the samples
        """
        leaf_centers = torch.stack([leaf.center for leaf in leaf_nodes], dim=0)
        leaf_labels = torch.stack([leaf.torch_labels[0] for leaf in leaf_nodes])
        # Get distances between points and centers. Get nearest center
        squared_diffs = squared_euclidean_distance(embedded, leaf_centers)
        cluster_center_assignments = (squared_diffs.min(dim=1)[1]).int()
        labels = leaf_labels[cluster_center_assignments]
        return leaf_centers, cluster_center_assignments, labels

    def _grow_tree(self, testloader: torch.utils.data.DataLoader, neural_network: torch.nn.Module, leaf_nodes: list,
                   new_cluster_id: int, optimizer: torch.optim.Optimizer, device: torch.device,
                   random_state: np.random.RandomState) -> None:
        """
        Grows the tree at the leaf node with the highest squared distances between its assigned samples and the center.
        The distance is not normalized, so larger clusters will be weighted higher.
        After the leaf node with highest squared distances has been identified, it will be split into two leaf nodes by performing bisecting KMeans.

        Parameters
        ----------
        testloader : torch.utils.data.DataLoader
            dataloader to be used for updating the clustering parameters
        neural_network : torch.nn.Module
            the neural network
        leaf_nodes : list
            list containing all leaf nodes within the cluster tree
        new_cluster_id : int
            the new cluster ID that should be added to the tree
        optimizer : torch.optim.Optimizer
            Optimizer for training
        device : torch.device
            device to be trained on
        random_state : np.random.RandomState
            use a fixed random state to get a repeatable solution
        """
        leaf_to_split = None
        max_sum_of_squared = 0
        embedded = encode_batchwise(testloader, neural_network)
        embedded_torch = torch.from_numpy(embedded).to(device)
        leaf_centers, cluster_center_assignments, labels = self._get_labels_from_leafs(embedded_torch, leaf_nodes)
        # Search leaf node with max distances
        squared_distances = (embedded_torch - leaf_centers[cluster_center_assignments]).pow(2).sum(1)
        for leaf_id in range(leaf_centers.shape[0]):
            squared_distances_clust = squared_distances[cluster_center_assignments == leaf_id]
            # Check that cluster has more than 1 sample
            if squared_distances_clust.shape[0] > 1:
                sum_of_squared_clust = squared_distances_clust.sum()
                if sum_of_squared_clust > max_sum_of_squared:
                    max_sum_of_squared = sum_of_squared_clust
                    leaf_to_split = leaf_id
        # Split node
        new_left_node, new_right_node = self.cluster_tree.split_cluster(
            leaf_nodes[leaf_to_split].labels[0], new_cluster_id)
        km = KMeans(n_clusters=2, n_init=20, random_state=random_state).fit(
            embedded[cluster_center_assignments.detach().cpu().numpy() == leaf_to_split])
        new_left_node.set_center_weight_and_torch_labels(km.cluster_centers_[0], 1, optimizer, device)
        new_right_node.set_center_weight_and_torch_labels(km.cluster_centers_[1], 1, optimizer, device)
        new_left_node.update_parents_torch_labels(
            device)  # Has to be called only once as the parent is the same for the left and right node
        # Change old center from torch.nn.Parameter to regular Tensor
        leaf_nodes[leaf_to_split].center = leaf_nodes[leaf_to_split].center.data

    def _update_split_node_centers(self, split_nodes: list, leaf_nodes: list, labels: torch.Tensor) -> list:
        """
        Update the centers and the weights of the split nods analytically as described in the paper.
        Returns a list containing all split nodes whose weight is below the pruning threshold (can be empty).

        Parameters
        ----------
        split_nodes : list
            list containing all split nodes within the cluster tree
        leaf_nodes : list
            list containing all leaf nodes within the cluster tree
        labels : torch.Tensor
            labels of the samples

        Returns
        -------
        nodes_to_prune : list
            list containing all split nodes whose weight is now below the pruning threshold
        """
        nodes_to_prune = []
        for node in split_nodes + leaf_nodes:
            if not node.is_leaf_node():
                # Update center of split nodes
                left_child = node.left_node_
                right_child = node.right_node_
                node.center = (left_child.weight * left_child.center + right_child.weight * right_child.center) / (
                        left_child.weight + right_child.weight)
            # Update weight of all nodes except root node
            if node.parent_node is not None:
                n_samples_in_node = torch.isin(labels, node.torch_labels).sum()
                node.weight = 0.5 * node.weight + 0.5 * n_samples_in_node
                if node.weight < self.pruning_threshold:
                    nodes_to_prune.append(node)
        return nodes_to_prune

    def _prune_tree(self, nodes_to_prune: list, device: torch.device) -> None:
        """
        Delete all nodes within nodes_to_prune from the cluster tree.

        Parameters
        ----------
        nodes_to_prune : list
            Contains all nodes that should be deleted. Can also be empty.
        device : torch.device
            device to be trained on
        """
        for node in nodes_to_prune:
            sibling = node.delete_node()
            if sibling is not None:
                sibling.update_parents_torch_labels(device)

    def _node_center_loss(self, embedded: torch.Tensor, leaf_centers: torch.Tensor,
                          cluster_center_assignments: torch.Tensor, embedded_aug: torch.Tensor) -> torch.Tensor:
        """
        Calculate the node center loss L_nc.

        Parameters
        ----------
        embedded : torch.Tensor
            The embedded batch of data
        leaf_centers : torch.Tensor
            The centers of the leaf nodes
        cluster_center_assignments : torch.Tensor
            The index of the cluster center assigned to each sample
        embedded_aug : torch.Tensor
            the embedded augmented batch of data

        Returns
        -------
        nc_loss : torch.Tensor
            The node center loss
        """
        unique_assignments = torch.unique(cluster_center_assignments)
        # Note that batch must not contain samples from all leaf nodes
        is_cluster_in_batch = [assign in unique_assignments for assign in range(leaf_centers.shape[0])]
        leaf_centers_in_batch = leaf_centers[is_cluster_in_batch]
        centers = torch.stack(
            [torch.mean(embedded[cluster_center_assignments == assign], dim=0) for assign in unique_assignments], dim=0)
        if self.augmentation_invariance:
            centers_aug = torch.stack(
                [torch.mean(embedded_aug[cluster_center_assignments == assign], dim=0) for assign in
                 unique_assignments], dim=0)
            centers = (centers + centers_aug) / 2
        # Calculate loss
        sum_centers_dist = torch.linalg.vector_norm(leaf_centers_in_batch - centers.detach(), dim=1).sum()
        nc_loss = sum_centers_dist / leaf_centers.shape[0]
        return nc_loss

    def _data_compression_loss(self, embedded: torch.Tensor, split_nodes: list, labels: torch.Tensor,
                               device: torch.device, embedded_aug: torch.Tensor) -> torch.Tensor:
        """
        Calculate the data compression loss L_dc.

        Parameters
        ----------
        embedded : torch.Tensor
            The embedded batch of data
        split_nodes : list
            list containing all split nodes within the cluster tree
        labels : torch.Tensor
            labels of the samples
        device : torch.device
            device to be trained on
        embedded_aug : torch.Tensor
            the embedded augmented batch of data

        Returns
        -------
        dc_loss : torch.Tensor
            The data compression loss
        """
        dc_loss = torch.tensor(0.).to(device)
        for node in split_nodes:
            samples_in_left = torch.isin(labels, node.left_node_.torch_labels)
            samples_in_right = torch.isin(labels, node.right_node_.torch_labels)
            # Check if samples are contained in subtree
            if torch.any(samples_in_left) or torch.any(samples_in_right):
                proj = (node.left_node_.center - node.right_node_.center) / torch.linalg.vector_norm(
                    node.left_node_.center - node.right_node_.center).detach()
            if torch.any(samples_in_left):
                # Loss on left side
                left_center = node.left_node_.center.detach()
                dc_loss += torch.abs(torch.matmul(left_center - embedded[samples_in_left], proj)).sum()
                if self.augmentation_invariance:
                    dc_loss += torch.abs(torch.matmul(left_center - embedded_aug[samples_in_left], proj)).sum()
            if torch.any(samples_in_right):
                # Loss on right side
                right_center = node.right_node_.center.detach()
                dc_loss += torch.abs(torch.matmul(right_center - embedded[samples_in_right], proj)).sum()
                if self.augmentation_invariance:
                    dc_loss += torch.abs(torch.matmul(right_center - embedded_aug[samples_in_right], proj)).sum()
        dc_loss = dc_loss / (2 * len(split_nodes) * embedded.shape[0])
        if self.augmentation_invariance:
            dc_loss /= 2
        return dc_loss

    def _loss(self, batch: list, neural_network: torch.nn.Module, ssl_loss_fn: torch.nn.modules.loss._Loss,
              clustering_loss_weight: float, ssl_loss_weight: float, leaf_nodes: list, split_nodes: list,
              device: torch.device) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the complete DeepECT + neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        neural_network : torch.nn.Module
            the neural network
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        leaf_nodes : list
            list containing all leaf nodes within the cluster tree
        split_nodes : list
            list containing all split nodes within the cluster tree
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : (torch.Tensor, torch.Tensor)
            the final DeepECT loss,
            the labels of the samples
        """
        # compute self-supervised loss
        if self.augmentation_invariance:
            ssl_loss, embedded, _, embedded_aug, _ = neural_network.loss_augmentation(batch, ssl_loss_fn, device)
        else:
            ssl_loss, embedded, _ = neural_network.loss(batch, ssl_loss_fn, device)
            embedded_aug = None
        # calculate cluster loss
        leaf_centers, cluster_center_assignments, labels = self._get_labels_from_leafs(embedded, leaf_nodes)
        nc_loss = self._node_center_loss(embedded, leaf_centers, cluster_center_assignments, embedded_aug)
        dc_loss = self._data_compression_loss(embedded, split_nodes, labels, device, embedded_aug)
        # Combine losses
        loss = clustering_loss_weight * (nc_loss + dc_loss) + ssl_loss_weight * ssl_loss
        return loss, labels

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader, n_epochs: int, device: torch.device,
            optimizer: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss, clustering_loss_weight: float,
            ssl_loss_weight: float, random_state: np.random.RandomState) -> "_DeepECT_Module":
        """
        Trains the _DeepECT_Module in place.

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
            Optimizer for training
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        random_state : np.random.RandomState
            use a fixed random state to get a repeatable solution

        Returns
        -------
        self : _DeepECT_Module
            This instance of the _DeepECT_Module
        """
        cluster_id = 2  # Two clusters were created during the initialization of the algorithm
        leaf_nodes, split_nodes = self.cluster_tree.get_leaf_and_split_nodes()
        tbar = tqdm.trange(n_epochs, desc="DeepECT training")
        for epoch in tbar:
            # Update Network
            total_loss = 0
            with torch.no_grad():
                # Grow tree
                if (epoch % self.grow_interval == 0 or self.cluster_tree.n_leaf_nodes_ < 2) and len(
                        leaf_nodes) < self.max_n_leaf_nodes:
                    self._grow_tree(testloader, neural_network, leaf_nodes, cluster_id, optimizer, device, random_state)
                    cluster_id += 1
                    leaf_nodes, split_nodes = self.cluster_tree.get_leaf_and_split_nodes()
            for batch in trainloader:
                # Calculate loss
                loss, labels = self._loss(batch, neural_network, ssl_loss_fn, clustering_loss_weight, ssl_loss_weight,
                                          leaf_nodes, split_nodes, device)
                total_loss += loss.item()
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Adapt centers and weights of split nodes analytically
                with torch.no_grad():
                    nodes_to_prune = self._update_split_node_centers(split_nodes, leaf_nodes, labels)
                    # Prune Tree
                    if len(nodes_to_prune) > 0:
                        self._prune_tree(nodes_to_prune, device)
                        leaf_nodes, split_nodes = self.cluster_tree.get_leaf_and_split_nodes()
            postfix_str = {"Loss": total_loss}
            tbar.set_postfix(postfix_str)
        return self


def _deep_ect(X: np.ndarray, max_n_leaf_nodes: int, batch_size: int, pretrain_optimizer_params: dict,
              clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int, grow_interval: int,
              pruning_threshold: float, optimizer_class: torch.optim.Optimizer,
              ssl_loss_fn: torch.nn.modules.loss._Loss, neural_network: torch.nn.Module | tuple,
              neural_network_weights: str, embedding_size: int, clustering_loss_weight: float, ssl_loss_weight: float,
              custom_dataloaders: tuple, augmentation_invariance: bool, device: torch.device,
              random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DeepECT clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        The given data set. Can be a np.ndarray or a torch.Tensor
    max_n_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree
    batch_size : int
        Size of the data batches
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the neural network, includes the learning rate
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        Number of epochs for the pretraining of the neural network
    clustering_epochs : int
        Number of epochs for the actual clustering procedure
    grow_interval : int
        Number of epochs after which the the tree is grown
    pruning_threshold : float
        The threshold for pruning the tree
    optimizer_class : torch.optim.Optimizer
        The optimizer class
    ssl_loss_fn : torch.nn.modules.loss._Loss
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
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations
    device : torch.device
        The device on which to perform the computations
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, torch.nn.Module)
        The tree as identified DeepECT,
        The labels as identified by DeepECT,
        The final neural network
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, _, neural_network, _, _, _, init_leafnode_centers, _ = get_default_deep_clustering_initialization(
        X, 2, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, ssl_loss_fn,
        neural_network, embedding_size, custom_dataloaders, KMeans, {"n_init": 20}, device,
        random_state, neural_network_weights=neural_network_weights)
    cluster_tree = BinaryClusterTree(_DeepECT_ClusterTreeNode)
    # Setup DeepECT Module
    deepect_module = _DeepECT_Module(cluster_tree, max_n_leaf_nodes, grow_interval, pruning_threshold,
                                     augmentation_invariance).to(device)
    # Use DeepECT optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(neural_network.parameters()), **clustering_optimizer_params)
    # DeepECT Training loop
    left_node, right_node = cluster_tree.split_cluster(0, 1)
    left_node.set_center_weight_and_torch_labels(init_leafnode_centers[0], 1, optimizer, device)
    right_node.set_center_weight_and_torch_labels(init_leafnode_centers[1], 1, optimizer, device)
    left_node.update_parents_torch_labels(
        device)  # Has to be called only once as the parent is the same for the left and right node
    # Change old center from torch.nn.Parameter to regular Tensor
    # Start fit
    deepect_module.fit(neural_network, trainloader, testloader, clustering_epochs, device, optimizer, ssl_loss_fn,
                       clustering_loss_weight, ssl_loss_weight, random_state)
    # Get labels
    labels = predict_batchwise(testloader, neural_network, deepect_module)
    return cluster_tree, labels, neural_network


class DeepECT(_AbstractDeepClusteringAlgo):
    """
    The Deep Embedded Cluster Tree (DeepECT) algorithm.
    First, a neural network will be trained (will be skipped if input neural network is given).
    Afterward, a cluster tree will be grown and the network will be optimized using the DeepECT loss function.

    Parameters
    ----------
    max_n_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree (default: 20)
    batch_size : int
        Size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 50)
    clustering_epochs : int
        Number of epochs for the actual clustering procedure (default: 200)
    grow_interval : int
        Number of epochs after which the the tree is grown (default: 2)
    pruning_threshold : float
        The threshold for pruning the tree (default: 0.1)
    optimizer_class : torch.optim.Optimizer
        The optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        Size of the embedding within the neural network (default: 10)
    clustering_loss_weight : float
        weight of the clustering loss (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations (default: False)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    tree_ : PredictionClusterTree
        The prediction cluster tree after training
    neural_network : torch.nn.Module
        The final neural network
    """

    def __init__(self, max_n_leaf_nodes: int = 20, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 50, clustering_epochs: int = 200,
                 grow_interval: int = 2, pruning_threshold: float = 0.1,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 1., ssl_loss_weight: float = 1.,
                 custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 device: torch.device = None, random_state: np.random.RandomState | int = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.max_n_leaf_nodes = max_n_leaf_nodes
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {
            "lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.grow_interval = grow_interval
        self.pruning_threshold = pruning_threshold
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "DeepECT":
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
        self : DeepECT
            This instance of the DeepECT algorithm
        """
        super().fit(X, y)
        tree, labels, neural_network = _deep_ect(X, self.max_n_leaf_nodes, self.batch_size,
                                                 self.pretrain_optimizer_params, self.clustering_optimizer_params,
                                                 self.pretrain_epochs, self.clustering_epochs, self.grow_interval,
                                                 self.pruning_threshold, self.optimizer_class, self.ssl_loss_fn,
                                                 self.neural_network, self.neural_network_weights, self.embedding_size,
                                                 self.clustering_loss_weight, self.ssl_loss_weight,
                                                 self.custom_dataloaders, self.augmentation_invariance, self.device,
                                                 self.random_state)
        self.tree_ = tree
        self.labels_ = labels
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
        leaf_nodes, _ = self.tree_.get_leaf_and_split_nodes()
        leaf_centers = np.array([leaf.center.data.detach().cpu().numpy() for leaf in leaf_nodes])
        leaf_labels = np.array([leaf.labels[0] for leaf in leaf_nodes])
        cluster_center_assignments = embedded_kmeans_prediction(X_embed, leaf_centers)
        predicted_labels = leaf_labels[cluster_center_assignments]
        return predicted_labels

    def flat_clustering(self, n_leaf_nodes_to_keep: int) -> np.ndarray:
        """
        Transform the predicted labels into a flat clustering result by only keeping n_leaf_nodes_to_keep leaf nodes in the tree.
        Returns labels as if the clustering procedure would have stopped at the specified number of nodes.
        Note that each leaf node corresponds to a cluster.

        Parameters
        ----------
        n_leaf_nodes_to_keep : int
            The number of leaf nodes to keep in the cluster tree

        Returns
        -------
        labels_pruned : np.ndarray
            The new cluster labels
        """
        assert self.labels_ is not None, "The DeepECT algorithm has not run yet. Use the fit() function first."
        tree_copy = copy.deepcopy(self.tree_)
        labels_pruned = tree_copy.prune_to_n_leaf_nodes(n_leaf_nodes_to_keep, self.labels_)
        return labels_pruned

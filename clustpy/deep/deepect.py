import os
import sys
from typing import Union

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch
import torch.utils
import torch.utils.data
from clustpy.data.real_torchvision_data import load_mnist
from clustpy.deep._data_utils import augmentation_invariance_check
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from clustpy.deep._utils import set_torch_seed
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from sklearn.cluster import KMeans
from tqdm import tqdm

from clustpy.metrics.hierarchical_metrics import PredictionClusterTree
from clustpy.deep._deepect_utils import (
    Cluster_Tree,
    transform_cluster_tree_to_pred_tree,
)


class _DeepECT_Module(torch.nn.Module):
    """
    The _DeepECT_Module. Contains most of the algorithm specific procedures like the loss and tree-grow functions.

    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    cluster_tree: Cluster_Node
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(
        self,
        init_leafnode_centers: np.ndarray,
        device: torch.device,
        random_state: np.random.RandomState,
        augmentation_invariance: bool = False,
    ):
        super().__init__()
        self.augmentation_invariance = augmentation_invariance
        # Create initial cluster tree
        self.cluster_tree = Cluster_Tree(
            init_leafnode_centers,
            device,
        )
        self.device = device
        self.random_state = random_state

    def fit(
        self,
        autoencoder: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        max_iterations: int,
        pruning_threshold: float,
        grow_interval: int,
        max_leaf_nodes: int,
        optimizer: torch.optim.Optimizer,
        rec_loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device | str],
    ) -> "_DeepECT_Module":
        """
        Trains the _DeepECT_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            The autoencoder used for training
        trainloader : torch.utils.data.DataLoader
            DataLoader for training data
        testloader : torch.utils.data.DataLoader
            DataLoader for testing data
        max_iterations : int
            Maximum number of iterations for training
        pruning_threshold : float
            Threshold for pruning the cluster tree
        grow_interval : int
            Interval for growing the cluster tree
        max_leaf_nodes : int
            Maximum number of leaf nodes in the cluster tree
        optimizer : torch.optim.Optimizer
            Optimizer for training
        rec_loss_fn : torch.nn.modules.loss._Loss
            Loss function for reconstruction
        device : Union[torch.device, str]
            Device for training (e.g., "cuda" or "cpu")

        Returns
        -------
        self : _DeepECT_Module
            This instance of the _DeepECT_Module
        """
        mov_dc_loss = 0.0
        mov_nc_loss = 0.0
        mov_rec_loss = 0.0
        mov_rec_loss_aug = 0.0
        mov_loss = 0.0

        optimizer.add_param_group({"params": self.cluster_tree.root.left_child.center})
        optimizer.add_param_group({"params": self.cluster_tree.root.right_child.center})

        with tqdm(
            range(max_iterations), desc="Fit", total=max_iterations
        ) as progress_bar:
            while True:
                for batch in trainloader:
                    optimizer.zero_grad()
                    if progress_bar.n > max_iterations:
                        break
                    if (
                        progress_bar.n > 0 and progress_bar.n % grow_interval == 0
                    ) or self.cluster_tree.number_nodes < 3:
                        if len(self.cluster_tree.leaf_nodes) < max_leaf_nodes:
                            self.cluster_tree.grow_tree(
                                testloader,
                                autoencoder,
                                optimizer,
                                device,
                            )

                    if self.augmentation_invariance:
                        idxs, M, M_aug = batch
                    else:
                        idxs, M = batch

                    # calculate autoencoder loss
                    rec_loss, embedded, reconstructed = autoencoder.loss(
                        [idxs, M], rec_loss_fn, self.device
                    )
                    if self.augmentation_invariance:
                        rec_loss_aug, embedded_aug, reconstructed_aug = (
                            autoencoder.loss([idxs, M_aug], rec_loss_fn, self.device)
                        )

                    self.cluster_tree.assign_to_nodes(embedded)

                    # calculate cluster loss
                    nc_loss = self.cluster_tree.nc_loss(
                        augmented_batch=(
                            embedded_aug if self.augmentation_invariance else None
                        )
                    )
                    dc_loss = self.cluster_tree.dc_loss(
                        len(M),
                        encoded_augmented_batch=(
                            embedded_aug if self.augmentation_invariance else None
                        ),
                    )

                    # adapt centers of split nodes analytically
                    self.cluster_tree.adapt_inner_nodes(self.cluster_tree.root)
                    self.cluster_tree.clear_node_assignments()

                    if self.cluster_tree.prune_tree(pruning_threshold):
                        nc_loss = torch.tensor([0.0], dtype=torch.float, device=device)
                        dc_loss = torch.tensor([0.0], dtype=torch.float, device=device)

                    if self.augmentation_invariance:
                        loss = nc_loss + dc_loss + rec_loss + rec_loss_aug
                        mov_rec_loss_aug += rec_loss_aug.item()
                    else:
                        loss = nc_loss + dc_loss + rec_loss

                    mov_nc_loss += nc_loss.item()
                    mov_dc_loss += dc_loss.item()
                    mov_rec_loss += rec_loss.item()
                    mov_loss += loss.item()

                    if (
                        progress_bar.n <= 10 or progress_bar.n % 100 == 0
                    ) and progress_bar.n > 0:
                        logging.info(
                            f"{progress_bar.n} - moving averages: dc_loss: {mov_dc_loss/progress_bar.n} "
                            f"nc_loss: {mov_nc_loss/progress_bar.n} rec_loss: {mov_rec_loss/progress_bar.n} "
                            f"{f'rec_loss_aug: {mov_rec_loss_aug/progress_bar.n}' if self.augmentation_invariance else ''} "
                            f"total_loss: {mov_loss/progress_bar.n}"
                        )

                    loss.backward()
                    optimizer.step()
                    progress_bar.update()
                else:
                    continue
                break
        return self

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
    ) -> PredictionClusterTree:
        """
        Batchwise prediction of the given samples in the dataloader for a
        given number of classes.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for the samples to be predicted
        autoencoder: torch.nn.Module
            Autoencoder model for calculating the embeddings

        Returns
        -------
        pred_tree : PredictionClusterTree
            The prediction cluster tree with assigned samples
        """
        # get prediction tree
        pred_tree = transform_cluster_tree_to_pred_tree(self.cluster_tree)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predict"):
                # calculate embeddings of the samples which should be predicted
                batch_data = batch[1].to(self.device)
                indices = batch[0].to(self.device)
                embeddings = autoencoder.encode(batch_data)
                # assign the embeddings to the cluster tree
                self.cluster_tree.assign_to_nodes(embeddings)
                # use assignment indices for prediction tree
                for node in self.cluster_tree.leaf_nodes:
                    pred_tree[node.id].assign_batch(indices, node.assignment_indices)
                self.cluster_tree.clear_node_assignments()

        return pred_tree


def _deep_ect(
    X: np.ndarray,
    batch_size: int,
    pretrain_optimizer_params: dict,
    clustering_optimizer_params: dict,
    pretrain_epochs: int,
    max_iterations: int,
    pruning_threshold: float,
    grow_interval: int,
    optimizer_class: torch.optim.Optimizer,
    rec_loss_fn: torch.nn.modules.loss._Loss,
    autoencoder: _AbstractAutoencoder,
    embedding_size: int,
    max_leaf_nodes: int,
    custom_dataloaders: tuple,
    augmentation_invariance: bool,
    random_state: np.random.RandomState,
    autoencoder_save_param_path: str = "pretrained_ae.pth",
):
    """
    Start the actual DeepECT clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        The given data set. Can be a np.ndarray or a torch.Tensor
    batch_size : int
        Size of the data batches
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        Number of epochs for the pretraining of the autoencoder
    max_iterations : int
        Number of iterations for the actual clustering procedure
    pruning_threshold : float
        The threshold for pruning the tree
    grow_interval : int
        Interval for growing the tree
    optimizer_class : torch.optim.Optimizer
        The optimizer class
    rec_loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction
    autoencoder : torch.nn.Module
        The input autoencoder
    embedding_size : int
        Size of the embedding within the autoencoder
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution
    autoencoder_save_param_path : str
        Path to save the autoencoder parameters

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DeepECT after the training terminated,
        The cluster centers as identified by DeepECT after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    save_ae_state_dict = not hasattr(autoencoder, "fitted") or not autoencoder.fitted
    set_torch_seed(random_state)

    (
        device,
        trainloader,
        testloader,
        autoencoder,
        _,
        _,
        _,
        init_leafnode_centers,
        _,
    ) = get_standard_initial_deep_clustering_setting(
        X,
        2,
        batch_size,
        pretrain_optimizer_params,
        pretrain_epochs,
        optimizer_class,
        rec_loss_fn,
        autoencoder,
        embedding_size,
        custom_dataloaders,
        KMeans,
        {"n_init": 20, "random_state": random_state},
        random_state,
    )

    print(device)
    if save_ae_state_dict:
        autoencoder.save_parameters(autoencoder_save_param_path)
    # Setup DeepECT Module
    deepect_module = _DeepECT_Module(
        init_leafnode_centers,
        device,
        random_state,
        augmentation_invariance,
    ).to(device)
    # Use DeepECT optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(
        list(autoencoder.parameters()), **clustering_optimizer_params
    )
    # DeepECT Training loop
    deepect_module.fit(
        autoencoder.to(device),
        trainloader,
        testloader,
        max_iterations,
        pruning_threshold,
        grow_interval,
        max_leaf_nodes,
        optimizer,
        rec_loss_fn,
        device,
    )
    # Get labels
    deepect_tree: PredictionClusterTree = deepect_module.predict(
        testloader, autoencoder
    )
    return deepect_tree, autoencoder


class DeepECT:
    """
    The Deep Embedded Cluster Tree (DeepECT) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, a cluster tree will be grown and the AE will be optimized using the DeepECT loss function.

    Parameters
    ----------
    batch_size : int
        Size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        Number of epochs for the pretraining of the autoencoder (default: 50)
    max_iterations : int
        Number of iterations for the actual clustering procedure (default: 50000)
    grow_interval : int
        Interval for growing the tree (default: 500)
    pruning_threshold : float
        The threshold for pruning the tree (default: 0.1)
    optimizer_class : torch.optim.Optimizer
        The optimizer class (default: torch.optim.Adam)
    rec_loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        The input autoencoder. If None, a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        Size of the embedding within the autoencoder (default: 10)
    max_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree (default: 20)
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position. If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations (default: False)
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    autoencoder_param_path : str
        Path to save the autoencoder parameters (default: None)

    Attributes
    ----------
    tree_ : PredictionClusterTree
        The prediction cluster tree after training
    autoencoder : torch.nn.Module
        The final autoencoder
    """

    def __init__(
        self,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = None,
        clustering_optimizer_params: dict = None,
        pretrain_epochs: int = 50,
        max_iterations: int = 40000,
        grow_interval: int = 500,
        pruning_threshold: float = 0.1,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        rec_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: _AbstractAutoencoder = None,
        embedding_size: int = 10,
        max_leaf_nodes: int = 20,
        custom_dataloaders: tuple = None,
        augmentation_invariance: bool = False,
        random_state: np.random.RandomState = np.random.RandomState(42),
        autoencoder_param_path: str = None,
    ):
        self.batch_size = batch_size
        self.pretrain_optimizer_params = (
            {"lr": 1e-3}
            if pretrain_optimizer_params is None
            else pretrain_optimizer_params
        )
        self.clustering_optimizer_params = (
            {"lr": 1e-4}
            if clustering_optimizer_params is None
            else clustering_optimizer_params
        )
        self.pretrain_epochs = pretrain_epochs
        self.max_iterations = max_iterations
        self.grow_interval = grow_interval
        self.pruning_threshold = pruning_threshold
        self.optimizer_class = optimizer_class
        self.rec_loss_fn = rec_loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.autoencoder_param_path = autoencoder_param_path

    def fit_predict(self, X: np.ndarray) -> "DeepECT":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set as a 2D-array of shape (#samples, #features)

        Returns
        -------
        self : DeepECT
            This instance of the DeepECT algorithm
        """
        augmentation_invariance_check(
            self.augmentation_invariance, self.custom_dataloaders
        )
        tree, autoencoder = _deep_ect(
            X,
            self.batch_size,
            self.pretrain_optimizer_params,
            self.clustering_optimizer_params,
            self.pretrain_epochs,
            self.max_iterations,
            self.pruning_threshold,
            self.grow_interval,
            self.optimizer_class,
            self.rec_loss_fn,
            self.autoencoder,
            self.embedding_size,
            self.max_leaf_nodes,
            self.custom_dataloaders,
            self.augmentation_invariance,
            self.random_state,
            self.autoencoder_param_path,
        )
        self.tree_ = tree
        self.autoencoder = autoencoder
        return self

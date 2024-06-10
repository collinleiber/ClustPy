"""
@authors:
Collin Leiber
"""

import torch
import numpy as np
from clustpy.deep._utils import detect_device, encode_batchwise, run_initial_clustering
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._train_utils import get_trained_autoencoder
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin, BaseEstimator, ClusterMixin
from sklearn.mixture import GaussianMixture as GMM
import inspect


def _manifold_based_sequential_dc(X: np.ndarray, n_clusters: int, batch_size: int, pretrain_optimizer_params: dict,
                                  pretrain_epochs: int, optimizer_class: torch.optim.Optimizer,
                                  loss_fn: torch.nn.modules.loss._Loss, autoencoder: torch.nn.Module,
                                  embedding_size: int, custom_dataloaders: tuple, manifold_class: TransformerMixin,
                                  manifold_params: dict, clustering_class: ClusterMixin, clustering_params: dict,
                                  random_state: np.random.RandomState) -> (
        int, np.ndarray, np.ndarray, torch.nn.Module, TransformerMixin):
    """
    Execute a manifold-based sequential deep clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters (can be None)
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    optimizer_class : torch.optim.Optimizer
        the optimizer class
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    manifold_class : TransformerMixin
        the manifold technique class
    manifold_params : dict
        Parameters for the manifold technique. Check out e.g. sklearn.manifold.TSNE for more information
    clustering_class : ClusterMixin
        clustering class to obtain the cluster labels after pretraining the autoencoder and learning the manifold
    clustering_params : dict
        parameters for the clustering class
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray, torch.nn.Module, TransformerMixin)
        The number of clusters,
        The cluster labels,
        The cluster centers,
        The final autoencoder,
        The Manifold object
    """
    # Get the device to train on
    device = detect_device()
    # sample random mini-batches from the data -> shuffle = True
    if custom_dataloaders is None:
        trainloader = get_dataloader(X, batch_size, True, False)
        testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
    # Get initial AE
    autoencoder = get_trained_autoencoder(trainloader, n_epochs=pretrain_epochs,
                                          optimizer_params=pretrain_optimizer_params, optimizer_class=optimizer_class,
                                          device=device, loss_fn=loss_fn, embedding_size=embedding_size,
                                          autoencoder=autoencoder)
    # Encode data
    X_embed = encode_batchwise(testloader, autoencoder)
    # Get possible input parameters of the manifold class
    manifold_class_parameters = inspect.getfullargspec(manifold_class).args + inspect.getfullargspec(
        manifold_class).kwonlyargs
    if "random_state" not in manifold_params.keys() and "random_state" in manifold_class_parameters:
        manifold_params["random_state"] = random_state
    # Execute Manifold
    manifold = manifold_class(**manifold_params)
    X_manifold = manifold.fit_transform(X_embed)
    # Execute Clustering Algorithm
    n_clusters, labels, centers, clustering_algo = run_initial_clustering(X_manifold, n_clusters, clustering_class,
                                                                          clustering_params, random_state)
    return n_clusters, labels, centers, autoencoder, manifold


class DDC_density_peak_clustering(BaseEstimator, ClusterMixin):
    """
    A variant of the Density Peak Algorithm as proposed in the DDC paper.

    Parameters
    ----------
    ratio : float
        The ratio parameter, defining the cutoff distance d_c by calculating: average pairwise distance * ratio

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels

    References
    ----------
    Ren, Yazhou, et al. "Deep density-based image clustering."
    Knowledge-Based Systems 197 (2020): 105841.
    """

    def __init__(self, ratio: float):
        self.ratio = ratio

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DDC_density_peak_clustering':
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
        self : DDC_density_peak_clustering
            this instance of the DDC variant of the Density Peak Clsutering algorithm
        """
        n_clusters, labels = _density_peak_clustering(X, self.ratio)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        return self


def _density_peak_clustering(X: np.ndarray, ratio: float) -> (int, np.ndarray):
    """
    Execute the variant of the Density Peak Algorithm as proposed in the paper.

    Parameters
    ----------
    X : np.ndarray
        The given data set
    ratio : float
        The ratio parameter, defining the cutoff distance d_c by calculating: average pairwise distance * ratio

    Returns
    -------
    tuple : (int,np.ndarray)
        The number of clusters,
        The cluster labels
    """
    distances = pdist(X)
    max_dist = np.max(distances)
    d_c = np.mean(distances) * ratio
    if d_c >= max_dist:
        print(
            "[WARNING] ratio parameter was chosen too large (ratio={0}). It is recommended to set ratio smaller than 1. d_c will be set to the maximum possible value".format(
                ratio))
        d_c = max_dist - 1e-8  # d_c can not be larger than the max distance
    # Calculate rho_i
    adj_distancse = np.exp(-((distances / d_c) ** 2))  # Equation 7
    rhos = np.sum(squareform(adj_distancse), axis=1)
    avg_rho = np.mean(rhos)  # Below Equation 9
    # Calculate delta_i and search for local cluster centers
    distances = squareform(distances)  # Convert distances to symmetric matrix
    deltas = np.zeros(X.shape[0])
    labels = np.full(X.shape[0], -1, np.int32)
    cluster_rhos = np.zeros((0, 2))
    cluster_id = 0
    chain_of_ids = []
    queue = list(range(X.shape[0]))
    while len(queue) > 0:
        i = queue.pop(0)
        if labels[i] == -1:
            chain_of_ids.append(i)
            distances_i = distances[i].copy()
            distances_i[rhos <= rhos[i]] = max_dist  # Equation 8
            nn_with_higher_dens = np.argmin(distances_i)  # Equation 8
            deltas[i] = distances_i[nn_with_higher_dens]  # Equation 8
            # Check if i is local cluster center
            if deltas[i] > d_c and rhos[i] > avg_rho:  # Equation 9
                labels[chain_of_ids] = cluster_id
                cluster_rhos = np.r_[cluster_rhos, [[np.sum(rhos[chain_of_ids]), len(chain_of_ids)]]]
                cluster_id += 1
                chain_of_ids = []
            elif labels[nn_with_higher_dens] != -1:
                labels[chain_of_ids] = labels[nn_with_higher_dens]
                cluster_rhos[labels[nn_with_higher_dens], 0] += np.sum(rhos[chain_of_ids])
                cluster_rhos[labels[nn_with_higher_dens], 1] += len(chain_of_ids)
                chain_of_ids = []
            else:
                queue.insert(0, nn_with_higher_dens)
    # ==> Start Merging of clusters
    # Average rho of clusters
    avg_cluster_rho = cluster_rhos[:, 0] / cluster_rhos[:, 1]
    # Get core points
    ids_core_points = np.where(rhos > avg_cluster_rho[labels])[0]  # Equation 10
    # Are clusters density connected?
    for i in range(len(ids_core_points) - 1):
        core_point_i = ids_core_points[i]
        for j in range(i + 1, len(ids_core_points)):
            core_point_j = ids_core_points[j]
            if distances[core_point_i, core_point_j] < d_c and labels[core_point_i] != labels[
                core_point_j]:  # Equation 11
                min_label = min(labels[core_point_i], labels[core_point_j])
                max_label = max(labels[core_point_i], labels[core_point_j])
                labels[labels == max_label] = min_label
                labels[labels > max_label] -= 1
                cluster_id -= 1
    return cluster_id, labels


class DDC(_AbstractDeepClusteringAlgo):
    """
    The Deep Density-based Image Clustering (DDC) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, t-SNE is executed on the embedded data and a variant of the Density Peak Clustering algorithm is executed.

    Parameters
    ----------
    ratio : float
        The ratio parameter, defining the cutoff distance d_c by calculating: average pairwise distance * ratio (default: 0.1)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    tsne_params : dict
        Parameters for the t-SNE execution. For example, perplexity can be changed by setting tsne_params to {"n_components": 2, "perplexity": 25}.
        Check out sklearn.manifold.TSNE for more information (default: {"n_components": 2})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels (obtained by a variant of Density Peak Clustering)
    autoencoder : torch.nn.Module
        The final autoencoder
    tsne_ : TSNE
        The t-SNE object

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DDC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> ddc = DDC(pretrain_epochs=3, clustering_epochs=3)
    >>> ddc.fit(data)

    References
    ----------
    Ren, Yazhou, et al. "Deep density-based image clustering."
    Knowledge-Based Systems 197 (2020): 105841.
    """

    def __init__(self, ratio: float = 0.1, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, custom_dataloaders: tuple = None, tsne_params: dict = None,
                 random_state: np.random.RandomState = None):
        super().__init__(batch_size, autoencoder, embedding_size, random_state)
        self.ratio = ratio
        if ratio > 1:
            print("[WARNING] ratio for DDC algorithm has been set to a value > 1 which can cause poor results")
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.custom_dataloaders = custom_dataloaders
        self.tsne_params = {"n_components": 2} if tsne_params is None else tsne_params

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DDC':
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
        self : DDC
            this instance of the DDC algorithm
        """
        n_clusters, labels, _, autoencoder, tsne = _manifold_based_sequential_dc(X, None, self.batch_size,
                                                                                 self.pretrain_optimizer_params,
                                                                                 self.pretrain_epochs,
                                                                                 self.optimizer_class, self.loss_fn,
                                                                                 self.autoencoder,
                                                                                 self.embedding_size,
                                                                                 self.custom_dataloaders, TSNE,
                                                                                 self.tsne_params,
                                                                                 DDC_density_peak_clustering,
                                                                                 {"ratio": self.ratio},
                                                                                 self.random_state)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.autoencoder = autoencoder
        self.tsne_ = tsne
        return self


class N2D(_AbstractDeepClusteringAlgo):
    """
    The Not 2 Deep (N2D) clustering algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, t-SNE/UMAP/ISOMAP is executed on the embedded data and the EM algorithm is executed.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    manifold_class : TransformerMixin
        the manifold technique class (default: TSNE)
    manifold_params : dict
        Parameters for the manifold execution. For example, perplexity can be changed for TSNE by setting manifold_params to {"n_components": 2, "perplexity": 25}.
        Check out e.g. sklearn.manifold.TSNE for more information (default: {"n_components": 2})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder
    manifold_ : TransformerMixin
        The manifold object

    References
    ----------
    McConville, Ryan, et al. "N2d:(not too) deep clustering via clustering the local manifold of an autoencoded embedding."
    2020 25th international conference on pattern recognition (ICPR). IEEE, 2021.
    """

    def __init__(self, n_clusters: int, batch_size: int = 256, pretrain_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, custom_dataloaders: tuple = None, manifold_class: TransformerMixin = TSNE,
                 manifold_params: dict = None, random_state: np.random.RandomState = None):
        super().__init__(batch_size, autoencoder, embedding_size, random_state)
        self.n_clusters = n_clusters
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.custom_dataloaders = custom_dataloaders
        self.manifold_class = manifold_class
        self.manifold_params = {"n_components": 2} if manifold_params is None else manifold_params

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'N2D':
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
        self : N2D
            this instance of the N2D algorithm
        """
        n_clusters, labels, centers, autoencoder, manifold = _manifold_based_sequential_dc(X, self.n_clusters,
                                                                                           self.batch_size,
                                                                                           self.pretrain_optimizer_params,
                                                                                           self.pretrain_epochs,
                                                                                           self.optimizer_class,
                                                                                           self.loss_fn,
                                                                                           self.autoencoder,
                                                                                           self.embedding_size,
                                                                                           self.custom_dataloaders,
                                                                                           self.manifold_class,
                                                                                           self.manifold_params,
                                                                                           GMM, {},
                                                                                           self.random_state)
        self.labels_ = labels.astype(np.int32)
        self.cluster_centers_ = centers
        self.autoencoder = autoencoder
        self.manifold_ = manifold
        return self

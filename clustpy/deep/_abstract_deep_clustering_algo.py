from clustpy.deep._utils import set_torch_seed
from sklearn.base import TransformerMixin, BaseEstimator, ClusterMixin
import numpy as np
import torch
from clustpy.deep._data_utils import augmentation_invariance_check
from clustpy.utils.checks import check_parameters
from sklearn.utils.validation import check_is_fitted


class _AbstractDeepClusteringAlgo(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    An abstract deep clustering algorithm class that can be used by other deep clustering implementations.

    Parameters
    ----------
    batch_size : int
        size of the data batches
    neural_network : torch.nn.Module | tuple
        the neural network used for the computations.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict).
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network.
    embedding_size : int
        size of the embedding within the autoencoder
    device : torch.device
        The device on which to perform the computations
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int
    """

    def __init__(self, batch_size: int, neural_network: torch.nn.Module | tuple, neural_network_weights: str,
                 embedding_size: int, device: torch.device, random_state: np.random.RandomState | int):
        self.batch_size = batch_size
        self.neural_network = neural_network
        self.neural_network_weights = neural_network_weights
        self.embedding_size = embedding_size
        self.device = device
        self.random_state = random_state

    def _check_parameters(self, X: np.ndarray, *, y: np.ndarray=None) -> (np.ndarray, np.ndarray, np.random.RandomState, dict, dict, dict):
        """
        Check if parameters for X, y and random_state are defined in accordance with the sklearn standard.
        Furthermore, it checks the deep clustering specific settings for augmentation_invariance, pretrain_optimizer_params, clustering_optimizer_params and initial_clustering_params.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can usually be ignored) (default: None)

        Returns
        -------
        tuple : (np.ndarray, np.ndarray, np.random.RandomState, dict, dict, dict)
            the checked data set,
            the checked labels,
            the checked random_state,
            the checked pretrain_optimizer_params,
            the checked clustering_optimizer_params,
            the checked initial_clustering_params
        """
        X, y, random_state = check_parameters(X=X, y=y, random_state=self.random_state, allow_nd=True)
        set_torch_seed(random_state)
        if hasattr(self, "augmentation_invariance"):
            assert hasattr(self,
                           "custom_dataloaders"), "If class uses augmentation_invariance it also requires the attribute custom_dataloaders"
            augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        if hasattr(self, "pretrain_optimizer_params"):
            pretrain_optimizer_params = {"lr": 1e-3} if self.pretrain_optimizer_params is None else self.pretrain_optimizer_params
        else:
            pretrain_optimizer_params = None
        if hasattr(self, "clustering_optimizer_params"):
            clustering_optimizer_params = {"lr": 1e-4} if self.clustering_optimizer_params is None else self.clustering_optimizer_params
        else:
            clustering_optimizer_params = None
        if hasattr(self, "initial_clustering_params"):
            initial_clustering_params = {} if self.initial_clustering_params is None else self.initial_clustering_params
        else:
            initial_clustering_params = None
        return X, y, random_state, pretrain_optimizer_params, clustering_optimizer_params, initial_clustering_params

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Embed the given data set using the trained neural network.

        Parameters
        ----------
        X: np.ndarray
            The given data set

        Returns
        -------
        X_embed : np.ndarray
            The embedded data set
        """
        check_is_fitted(self, ["labels_", "neural_network_trained_"])
        X, _, _ = check_parameters(X)
        X_embed = self.neural_network_trained_.transform(X, self.batch_size)
        return X_embed.astype(X.dtype)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> '_AbstractDeepClusteringAlgo':
        """
        Placeholder for the fit function of deep clustering algorithms.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : _AbstractDeepClusteringAlgo
            this instance of the _AbstractDeepClusteringAlgo
        """
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray=None):
        """
        Train the deep clusterin algorithm on the given data set and return the final embedded version of the data using the trained neural network.

        Parameters
        ----------
        X: np.ndarray
            The given data set
        y : np.ndarray
            the labels (can usually be ignored)

        Returns
        -------
        X_embed : np.ndarray
            The embedded data set
        """
        self.fit(X, y)
        X_embed = self.transform(X)
        return X_embed

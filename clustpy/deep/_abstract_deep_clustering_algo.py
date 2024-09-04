from clustpy.deep._utils import set_torch_seed
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
import numpy as np
import torch
from clustpy.deep._data_utils import augmentation_invariance_check


class _AbstractDeepClusteringAlgo(BaseEstimator, ClusterMixin):
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
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> '_AbstractDeepClusteringAlgo':
        """
        Checks if augmentation invariance is correctly applied and sets the seed for the execution.

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
        if hasattr(self, "augmentation_invariance"):
            assert hasattr(self,
                           "custom_dataloaders"), "If class uses augmentation_invariance it also requires the attribute custom_dataloaders"
            augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        set_torch_seed(self.random_state)

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
        X_embed = self.neural_network.transform(X, self.batch_size)
        return X_embed

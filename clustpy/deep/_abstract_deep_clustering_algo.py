from clustpy.deep._utils import set_torch_seed
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
import numpy as np
import torch


class _AbstractDeepClusteringAlgo(BaseEstimator, ClusterMixin):
    """
    An abstract deep clustering algorithm class that can be used by other deep clustering implementations.

    Parameters
    ----------
    batch_size : int
        size of the data batches
    neural_network : torch.nn.Module
        the neural network used for the computations
    embedding_size : int
        size of the embedding within the autoencoder
    device : torch.device
        The device on which to perform the computations
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int
    """

    def __init__(self, batch_size: int, neural_network: torch.nn.Module, embedding_size: int,
                 device: torch.device, random_state: np.random.RandomState | int):
        self.batch_size = batch_size
        self.neural_network = neural_network
        self.embedding_size = embedding_size
        self.device = device
        self.random_state = check_random_state(random_state)
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

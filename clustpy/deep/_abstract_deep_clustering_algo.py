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
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int
    """

    def __init__(self, batch_size: int, autoencoder: torch.nn.Module, embedding_size: int,
                 random_state: np.random.RandomState):
        self.batch_size = batch_size
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Embed the given data set using the trained autoencoder.

        Parameters
        ----------
        X: np.ndarray
            The given data set

        Returns
        -------
        X_embed : np.ndarray
            The embedded data set
        """
        X_embed = self.autoencoder.transform(X, self.batch_size)
        return X_embed

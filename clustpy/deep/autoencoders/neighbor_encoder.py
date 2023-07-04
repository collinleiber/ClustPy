"""
@authors:
Collin Leiber
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist
from clustpy.deep.autoencoders.feedforward_autoencoder import FeedforwardAutoencoder
from clustpy.deep.autoencoders._abstract_autoencoder import FullyConnectedBlock


def get_neighbors_batchwise(X: np.ndarray, n_neighbors: int, metric: str = "sqeuclidean",
                            batch_size: int = 10000) -> list:
    """
    For large datasets it is often not possible to determine the nearest neighbors in a trivial manner.
    Therefore, here is an implementation that calculates the nearest neighbors in batches.
    Ignores the objects themselves (with distance of 0) as nearest neighbors.
    It reduces the memory consumption of a trivial nearest neighbor implementation from (data_size x data_size) to (batch_size x data_size).
    A list is returned, which can be given as additional input into a DataLoader and is therefore directly compatible with the NeighborEncoder.
    Due to runtime concerns it is still recommended to use a more complex nearest neighbor retrieval implementation (e.g. from sklearn.neighbor)!

    Parameters
    ----------
    X : np.ndarray
        The given data set
    n_neighbors : int
        The number of nearest neighbors to identify
    metric : str
        The distance metric to be used. See scipy.spatial.distance.cdist for more information (default: sqeuclidean)
    batch_size : int
        The size of the batches (default: 10000)

    Returns
    -------
    nearest_neigbors : list
        A list containing the nearest neighbors as torch.Tensors, i.e. [1-nearest-neighbor tensor, 2-nearest-neighbor tensor, ...]

    Examples
    --------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import get_dataloader
    >>> X, L = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> n_neighbors = 3
    >>> neighbors = get_neighbors_batchwise(X, n_neighbors)
    >>> dataloader = get_dataloader(X, 256, True, additional_inputs=neighbors)
    >>> neighbor_encoder = NeighborEncoder(layers=[X.shape[1], 512, 256, 10], n_neighbors=n_neighbors)
    >>> neighbor_encoder.fit(dataloader=dataloader, n_epochs=5, lr=1e-3)
    """
    # batch_size should not be larger than the dataset
    batch_size = min(X.shape[0], batch_size)
    # Create list containing the nearest neighbors
    nearest_neigbors = [np.zeros(X.shape) for _ in range(n_neighbors)]
    # Check if last batch is a complete batch
    add_one_batch = 0 if X.shape[0] % batch_size == 0 else 1
    n_iterations = X.shape[0] // batch_size + add_one_batch
    for i in range(n_iterations):
        index_0 = i * batch_size
        if i != n_iterations - 1 or add_one_batch == 0:
            index_1 = index_0 + batch_size
        else:
            index_1 = index_0 + X.shape[0] % batch_size
        distances = cdist(X[index_0:index_1], X, metric=metric)
        arg_distances = np.argsort(distances, axis=1)
        for k in range(n_neighbors):
            nearest_neigbors[k][index_0:index_1] = X[arg_distances[:, k + 1]]
    return nearest_neigbors


class NeighborEncoder(FeedforwardAutoencoder):
    """
    A NeighborEncoder. Does not compare the reconstruction of an object to itself but to its nearest neighbors.
    For more information see the stated reference.
    If n_neighbors is 0 and decode_self is true, the NeighborEncoder will work as a regular FeedforwardAutoencoder.

    Parameters
    ----------
    layers : list
        list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the embedding dimension.
        If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    n_neighbors : int
        the number of nearest neighbors to be considered
    decode_self : bool
        specifies whether a point itself should also be decoded (default: False)
    batch_norm : bool
        Set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float
        Set the amount of dropout you want to use (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    decoder_layers : list
        list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers (default: None)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)

    Attributes
    ----------
    encoder : FullyConnectedBlock
        encoder part of the autoencoder, responsible for embedding data points (class is FullyConnectedBlock)
    decoder : FullyConnectedBlock
        decoder part of the autoencoder, responsible for reconstructing the data point itself (class is FullyConnectedBlock).
        Only used if decode_self is true.
    neighbor_decoders : list
        list containing one decoder network (class is FullyConnectedBlock) for each nearest neighbor
    fitted  : bool
        boolean value indicating whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by multiple deep clustering algorithms

    Examples
    --------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import get_dataloader
    >>> from clustpy.deep._utils import detect_device
    >>> from scipy.spatial.distance import pdist, squareform

    >>> X, L = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> device = detect_device()
    >>> n_neighbors = 3

    >>> dist_matrix = squareform(pdist(X))
    >>> neighbor_ids = np.argsort(dist_matrix, axis=1)
    >>> neighbors = [X[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    >>> # Alternatively: neighbors = get_neighbors_batchwise(X, n_neighbors)

    >>> dataloader = get_dataloader(X, 256, True, additional_inputs=neighbors)
    >>> neighbor_encoder = NeighborEncoder(layers=[X.shape[1], 512, 256, 10], n_neighbors=n_neighbors, decode_self=False)
    >>> neighbor_encoder.fit(dataloader=dataloader, device=device, n_epochs=5, lr=1e-3)

    References
    ----------
    Yeh, Chin-Chia Michael, et al. "Representation Learning by Reconstructing Neighborhoods." arXiv preprint arXiv:1811.01557 (2018).
    """

    def __init__(self, layers: list, n_neighbors: int, decode_self: bool = False, batch_norm: bool = False,
                 dropout: float = None, activation_fn: torch.nn.Module = torch.nn.LeakyReLU, bias: bool = True,
                 decoder_layers: list = None, decoder_output_fn: torch.nn.Module = None, reusable: bool = True):
        assert n_neighbors > 0 or decode_self, "n_neighbors must be an integer larger than 0 or decode_self must be true"
        super(NeighborEncoder, self).__init__(layers, batch_norm, dropout, activation_fn, bias,
                                              decoder_layers, decoder_output_fn, reusable)
        self.n_neighbors = n_neighbors
        self.decode_self = decode_self
        neighbor_decoders = torch.nn.ModuleList([FullyConnectedBlock(layers=self.decoder.layers, batch_norm=batch_norm,
                                                                     dropout=dropout, activation_fn=activation_fn,
                                                                     bias=bias,
                                                                     output_fn=decoder_output_fn) for _ in
                                                 range(n_neighbors)])
        self.neighbor_decoders = neighbor_decoders

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function of each neighbor network to embedded.
        Returns a (n_neighbors x batch_size x dimensionality) tensor if decode_self is false, else a (n_neighbors + 1 x batch_size x dimensionality) tensor

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        decoded_neighbors : torch.Tensor
            returns the reconstructions of the embedded sample concerning its neighbor decoders
        """
        assert embedded.shape[1] == self.decoder.layers[0], "Input layer of the decoder does not match input sample"
        n_decoded_objects = self.n_neighbors + 1 if self.decode_self else self.n_neighbors
        decoded_neighbors = torch.zeros((n_decoded_objects, embedded.shape[0], self.encoder.layers[0]))
        for i in range(self.n_neighbors):  # TODO: Maybe use functorch.vmap in the future for vectorization
            decoded_neighbors[i] = self.neighbor_decoders[i](embedded)
        if self.decode_self:
            decoded_neighbors[-1] = self.decoder(embedded)
        return decoded_neighbors

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the loss of a single batch of data.
        Corresponds to the sum of losses concerning each neighbor.
        batch must contain the data object at the first position and the neighbors at the following positions.

        Parameters
        ----------
        batch: list
            the different parts of a dataloader (id, samples, 1-nearest-neighbor, 2-nearest-neighbor, ...)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : (torch.Tensor, torch.Tensor, torch.Tensor)
            the sum of the reconstruction losses of the input sample,
            the embedded input sample,
            the reconstructions of the embedded sample concerning its neighbor decoders
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        batch_data = batch[1].to(device)
        embedded = self.encode(batch_data)
        decoded_neighbors = self.decode(embedded)
        loss = torch.tensor(0.)
        for i in range(self.n_neighbors):  # TODO: Maybe use functorch.vmap in the future for vectorization
            neighbors = batch[2 + i].to(device)
            loss = loss + loss_fn(decoded_neighbors[i], neighbors)
        if self.decode_self:
            reconstruction = decoded_neighbors[-1]
            loss = loss + loss_fn(reconstruction, batch_data)
        return loss, embedded, decoded_neighbors

    def fit(self, n_epochs: int, optimizer_params: dict, batch_size: int = 128,
            dataloader: torch.utils.data.DataLoader = None, evalloader: torch.utils.data.DataLoader = None,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = None,
            device: torch.device = torch.device("cpu"), model_path: str = None,
            print_step: int = 0) -> 'NeighborEncoder':
        """
        Trains the NeighborEncoder in place.
        Equal to fit function of the FeedforwardAutoencoder but does only work with a dataloader (not with a regular data array).
        This is because the dataloader must contain the nearest neighbors of each point at the positions 2, 3, ....

        Parameters
        ----------
        n_epochs : int
            number of epochs for training
        optimizer_params : dict
            parameters of the optimizer, includes the learning rate
        batch_size : int
            size of the data batches (default: 128)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training (default: default=None)
        evalloader : torch.utils.data.DataLoader
            dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau (default: None)
        optimizer_class : torch.optim.Optimizer
            optimizer to be used (default: torch.optim.Adam)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction (default: torch.nn.MSELoss())
        patience : int
            patience parameter for EarlyStopping (default: 5)
        scheduler : torch.optim.lr_scheduler
            learning rate scheduler that should be used.
            If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader (default: None)
        scheduler_params : dict
            dictionary of the parameters of the scheduler object (default: None)
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        model_path : str
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)
        print_step : int
            specifies how often the losses are printed. If 0, no prints will occur (default: 0)

        Returns
        -------
        self : NeighborEncoder
            this instance of the NeighborEncoder
        """
        super().fit(n_epochs, optimizer_params, batch_size, None, None, dataloader, evalloader, optimizer_class,
                    loss_fn, patience,
                    scheduler, scheduler_params, device, model_path, print_step)
        return self

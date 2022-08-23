"""
@authors:
Collin Leiber
"""

import torch
from cluspy.deep.flexible_autoencoder import FullyConnectedBlock, FlexibleAutoencoder


class NeighborEncoder(FlexibleAutoencoder):
    """
    A NeighborEncoder. Does not compare the reconstruction of an object to itself but to its nearest neighbors.
    For more information see the stated reference.
    If n_neighbors is 0 and decode_self is true, the NeighborEncoder will work as a regular FlexibleAutoencoder.

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
        e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)

    Attributes
    ----------
    encoder : FullyConnectedBlock
        encoder part of the autoencoder, responsible for embedding data points (class is FullyConnectedBlock)
    decoder : FullyConnectedBlock
        decoder part of the autoencoder, responsible for reconstructing the data point itself (class is FullyConnectedBlock).
        Only used if decode_self is true.
    fitted  : bool
        boolean value indicating whether the autoencoder is already fitted.
    neighbor_decoders : list
        list containing one decoder network (class is FullyConnectedBlock) for each nearest neighbor

    Examples
    --------
    from cluspy.data import load_optdigits
    from cluspy.deep import get_dataloader
    from scipy.spatial.distance import pdist, squareform
    from cluspy.deep._utils import detect_device
    import numpy as np

    X, L = load_optdigits()
    dist_matrix = squareform(pdist(X))
    device = detect_device()
    n_neighbors = 3
    neighbor_ids = np.argsort(dist_matrix, axis=1)
    neighbors = [X[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    dataloader = get_dataloader(X, 256, True, additional_inputs=neighbors)
    neighbor_encoder = NeighborEncoder(layers=[X.shape[1], 512, 256, 10], n_neighbors=n_neighbors, decode_self=False)
    neighbor_encoder.fit(dataloader=dataloader, device=device, n_epochs=100, lr=1e-3)

    References
    ----------
    Yeh, Chin-Chia Michael, et al. "Representation Learning by Reconstructing Neighborhoods." arXiv preprint arXiv:1811.01557 (2018).
    """

    def __init__(self, layers: list, n_neighbors: int, decode_self: bool = False, batch_norm: bool = False,
                 dropout: float = None, activation_fn: torch.nn.Module = torch.nn.LeakyReLU,
                 bias: bool = True, decoder_layers: list = None, decoder_output_fn: torch.nn.Module = None):
        assert n_neighbors > 0 or decode_self, "n_neighbors must be an integer larger than 0 or decode_self must be true"
        super(NeighborEncoder, self).__init__(layers, batch_norm, dropout, activation_fn, bias,
                                              decoder_layers, decoder_output_fn)
        self.n_neighbors = n_neighbors
        self.decode_self = decode_self
        neighbor_decoders = []
        for i in range(n_neighbors):
            neighbor_decoders.append(FullyConnectedBlock(layers=self.decoder.layers, batch_norm=batch_norm,
                                                         dropout=dropout, activation_fn=activation_fn, bias=bias,
                                                         output_fn=decoder_output_fn))
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
            returns the reconstruction of embedded concerning each neighbor
        """
        assert embedded.shape[1] == self.decoder.layers[0], "Input layer of the decoder does not match input sample"
        n_decoded_objects = self.n_neighbors + 1 if self.decode_self else self.n_neighbors
        decoded_neighbors = torch.zeros((n_decoded_objects, embedded.shape[0], self.encoder.layers[0]))
        for i in range(self.n_neighbors):
            decoded_neighbors[i] = self.neighbor_decoders[i](embedded)
        if self.decode_self:
            decoded_neighbors[-1] = self.decoder(embedded)
        return decoded_neighbors

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss) -> torch.Tensor:
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

        Returns
        -------
        loss : torch.Tensor
            returns the sum of the reconstruction losses of the input sample
        """
        batch_data = batch[1]
        decoded_neighbors = self.forward(batch_data)
        loss = 0
        for i in range(self.n_neighbors):
            neighbors = batch[2 + i]
            loss = loss + loss_fn(decoded_neighbors[i], neighbors)
        if self.decode_self:
            reconstruction = decoded_neighbors[-1]
            loss = loss + loss_fn(reconstruction, batch_data)
        return loss

    def fit(self, n_epochs: int, lr: float, batch_size: int = 128,
            dataloader: torch.utils.data.DataLoader = None, evalloader: torch.utils.data.DataLoader = None,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = None,
            device: torch.device = torch.device("cpu"), model_path: str = None,
            print_step: int = 0) -> 'NeighborEncoder':
        """
        Trains the NeighborEncoder in place.
        Equal to fit function of the FelxibleAutoencoder but does only work with a dataloader (not with a regular data array).
        This is because the dataloader must contain the nearest neighbors of each point at the positions 2, 3, ....

        Parameters
        ----------
        n_epochs : int
            number of epochs for training
        lr : float
            learning rate to be used for the optimizer_class
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
        super().fit(n_epochs, lr, batch_size, None, None, dataloader, evalloader, optimizer_class, loss_fn, patience,
                    scheduler, scheduler_params, device, model_path, print_step)
        return self

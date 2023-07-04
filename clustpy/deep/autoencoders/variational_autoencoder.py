"""
@authors:
Lukas Miklautz,
Donatella Novakovic,
Collin Leiber
"""

import torch
from clustpy.deep.autoencoders.feedforward_autoencoder import FullyConnectedBlock, FeedforwardAutoencoder


def _vae_sampling(q_mean: torch.Tensor, q_logvar: torch.Tensor) -> torch.Tensor:
    """
    Sample from the central layer of the variational autoencoder.

    Parameters
    ----------
    q_mean : torch.Tensor
        mean value of the central layer
    q_logvar : torch.Tensor
        logarithmic variance of the central layer (use logarithm of variance - numerical purposes)

    Returns
    -------
    z : torch.Tensor
        The new sample
    """
    std = torch.exp(0.5 * q_logvar)
    eps = torch.randn_like(std)
    z = q_mean + eps * std
    return z


class VariationalAutoencoder(FeedforwardAutoencoder):
    """
    A variational autoencoder (VAE).

    Parameters
    ----------
    layers : list
        list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the dimension of the mean and variance value in the central layer.
             If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    batch_norm : bool
        set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float
        set the amount of dropout you want to use (default: None)
    activation: torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    decoder_layers : list
        list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers (default: None)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: torch.nn.Sigmoid)
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)

    Attributes
    ----------
    encoder : FullyConnectedBlock
        encoder part of the autoencoder, responsible for embedding data points (class is FullyConnectedBlock)
    decoder : FullyConnectedBlock
        decoder part of the autoencoder, responsible for reconstructing data points from the embedding (class is FullyConnectedBlock)
    mean : torch.nn.Linear
        mean value of the central layer
    log_variance : torch.nn.Linear
        logarithmic variance of the central layer (use logarithm of variance - numerical purposes)
    fitted  : bool
        boolean value indicating whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by mutliple deep clustering algorithms

    References
    ----------
    Kingma, Diederik P., and Max Welling. "Auto-encoding variational Bayes." Int. Conf. on Learning Representations.
    """

    def __init__(self, layers: list, batch_norm: bool = False, dropout: float = None,
                 activation_fn: torch.nn.Module = torch.nn.LeakyReLU, bias: bool = True, decoder_layers: list = None,
                 decoder_output_fn: torch.nn.Module = torch.nn.Sigmoid, reusable: bool = True):
        super(VariationalAutoencoder, self).__init__(layers, batch_norm, dropout, activation_fn, bias,
                                                     decoder_layers, decoder_output_fn, reusable)
        # Get size of embedding from last dimension of layers
        embedding_size = layers[-1]
        # Overwrite encoder from FeedforwardAutoencoder, leave out the last layer
        self.encoder = FullyConnectedBlock(layers=layers[:-1], batch_norm=batch_norm, dropout=dropout,
                                           activation_fn=activation_fn, bias=bias, output_fn=activation_fn)
        self.mean = torch.nn.Linear(layers[-2], embedding_size)
        self.log_variance = torch.nn.Linear(layers[-2], embedding_size)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Apply the encoder function to x.
        Overwrites function from FeedforwardAutoencoder.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor)
            mean value of the central VAE layer,
            logarithmic variance value of the central VAE layer (use logarithm of variance - numerical purposes)
        """
        assert x.shape[1] == self.encoder.layers[0], "Input layer of the encoder does not match input sample."
        embedded = self.encoder(x)
        q_mean = self.mean(embedded)
        q_logvar = self.log_variance(embedded)
        return q_mean, q_logvar

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Applies both the encode and decode function.
        The forward function is automatically called if we call self(x).
        Overwrites function from FeedforwardAutoencoder.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of embedded points

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            sampling using q_mean and q_logvar,
            mean value of the central VAE layer,
            logarithmic variance value of the central VAE layer (use logarithm of variance - numerical purposes),
            the reconstruction of the data point
        """
        q_mean, q_logvar = self.encode(x)
        z = _vae_sampling(q_mean, q_logvar)
        reconstruction = self.decode(z)
        return z, q_mean, q_logvar, reconstruction

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
             beta: float = 1) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the loss of a single batch of data.

        Parameters
        ----------
        batch: list
            the different parts of a dataloader (id, samples, ...)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on
        beta : float
            weighting of the KL loss (default: 1)

        Returns
        -------
        total_loss: (torch.Tensor, torch.Tensor, torch.Tensor)
            the reconstruction loss of the input sample,
            the sampling,
            the reconstruction of the data point
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        batch_data = batch[1].to(device)
        z, q_mean, q_logvar, reconstruction = self.forward(batch_data)
        rec_loss = loss_fn(reconstruction, batch_data)

        kl_loss = 0.5 * torch.sum(q_mean.pow(2) + torch.exp(q_logvar) - 1.0 - q_logvar)
        kl_loss /= batch_data.shape[0]

        total_loss = rec_loss + beta * kl_loss
        return total_loss, z, reconstruction

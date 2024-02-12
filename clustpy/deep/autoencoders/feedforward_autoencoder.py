"""
@authors:
Lukas Miklautz
"""

import torch
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder, FullyConnectedBlock


class FeedforwardAutoencoder(_AbstractAutoencoder):
    """
    A flexible feedforward autoencoder.

    Parameters
    ----------
    layers : list
        list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the embedding dimension.
        If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
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
        decoder part of the autoencoder, responsible for reconstructing data points from the embedding (class is FullyConnectedBlock)
    fitted  : bool
        indicates whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by multiple deep clustering algorithms

    References
    ----------
    E.g.:
    Ballard, Dana H. "Modular learning in neural networks." Aaai. Vol. 647. 1987.

    or

    Kramer, Mark A. "Nonlinear principal component analysis using autoassociative neural networks."
    AIChE journal 37.2 (1991): 233-243.
    """

    def __init__(self, layers: list, batch_norm: bool = False, dropout: float = None,
                 activation_fn: torch.nn.Module = torch.nn.LeakyReLU, bias: bool = True, decoder_layers: list = None,
                 decoder_output_fn: torch.nn.Module = None, reusable: bool = True):
        super(FeedforwardAutoencoder, self).__init__(reusable)
        if decoder_layers is None:
            decoder_layers = layers[::-1]
        if (layers[-1] != decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {layers[-1]} and {decoder_layers[0]} respectively.")
        if (layers[0] != decoder_layers[-1]):
            raise ValueError(
                f"Output and input dimension do not match, they are {layers[0]} and {decoder_layers[-1]} respectively.")
        # Initialize encoder
        self.encoder = FullyConnectedBlock(layers=layers, batch_norm=batch_norm, dropout=dropout,
                                           activation_fn=activation_fn, bias=bias, output_fn=None)

        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(layers=decoder_layers, batch_norm=batch_norm, dropout=dropout,
                                           activation_fn=activation_fn, bias=bias,
                                           output_fn=decoder_output_fn)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points
        
        Returns
        -------
        embedded : torch.Tensor
            the embedded data point with dimensionality embedding_size
        """
        assert x.shape[1] == self.encoder.layers[
            0], "Input layer of the encoder ({0}) does not match input sample ({1})".format(self.encoder.layers[0],
                                                                                            x.shape[1])
        embedded = self.encoder(x)
        return embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points
        
        Returns
        -------
        decoded : torch.Tensor
            returns the reconstruction of embedded
        """
        assert embedded.shape[1] == self.decoder.layers[0], "Input layer of the decoder does not match input sample"
        decoded = self.decoder(embedded)
        return decoded

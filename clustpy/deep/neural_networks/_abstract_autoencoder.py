"""
@authors:
Lukas Miklautz
"""

import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import get_device_from_module, mean_squared_error, set_torch_seed
import tqdm
from collections.abc import Callable
from clustpy.utils.checks import check_random_state
from clustpy.deep.neural_networks._abstract_neural_network import _AbstractNeuralNetwork


class FullyConnectedBlock(torch.nn.Module):
    """
    Feed Forward Neural Network Block

    Parameters
    ----------
    layers : list
        list of the different layer sizes
    batch_norm : bool
        set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float | None
        set the amount of dropout you want to use (default: None)
    activation_fn : type[torch.nn.Module] | None
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: None)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    output_fn : type[torch.nn.Module] | None
        activation function from torch.nn, set the activation function for the last layer, if None then it will be linear (default: None)

    Attributes
    ----------
    block: torch.nn.Sequential
        feed forward neural network
    layer_positions : list
        ids specifying at which positions the input layers are contained within block
    """

    def __init__(self, layers: list, batch_norm: bool = False, dropout: float | None = None,
                 activation_fn: type[torch.nn.Module] | None = None, bias: bool = True, output_fn: type[torch.nn.Module] | None = None):
        super(FullyConnectedBlock, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bias = bias
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        layer_positions = []
        fc_block_list : list[torch.nn.Module] = []
        for i in range(len(layers) - 1):
            layer_positions.append(len(fc_block_list))
            fc_block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=self.bias))
            if self.batch_norm:
                fc_block_list.append(torch.nn.BatchNorm1d(layers[i + 1]))
            if self.dropout is not None:
                fc_block_list.append(torch.nn.Dropout(self.dropout))
            if self.activation_fn is not None:
                # last layer is handled differently
                if (i != len(layers) - 2):
                    fc_block_list.append(self.activation_fn())
                else:
                    if self.output_fn is not None:
                        fc_block_list.append(self.output_fn())

        self.block = torch.nn.Sequential(*fc_block_list)
        self.layer_positions = layer_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass a sample through the FullyConnectedBlock.

        Parameters
        ----------
        x : torch.Tensor
            the sample

        Returns
        -------
        forwarded : torch.Tensor
            The passed sample.
        """
        forwarded = self.block(x)
        return forwarded


class _AbstractAutoencoder(_AbstractNeuralNetwork):


    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for a decode function of an autoencoder.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        embedded : torch.Tensor
            should return the reconstruction of embedded
        """
        return embedded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies both the encode and decode function.
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of embedded points

        Returns
        -------
        reconstruction : torch.Tensor
            returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def loss(self, batch: list, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss, device: torch.device,
             corruption_fn: Callable | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss of a single batch of data.

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, samples, ...)
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on
        corruption_fn : Callable | None
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)

        Returns
        -------
        loss : tuple[torch.Tensor, torch.Tensor]
            the reconstruction loss of the input sample,
            the embedded input sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        batch_data = batch[1].to(device)
        batch_data_adj = batch_data if corruption_fn is None else corruption_fn(batch_data)
        embedded = self.encode(batch_data_adj)
        reconstructed = self.decode(embedded)
        loss = ssl_loss_fn(reconstructed, batch_data)
        return loss, embedded

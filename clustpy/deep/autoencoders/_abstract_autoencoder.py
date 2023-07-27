"""
@authors:
Lukas Miklautz
"""

import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from clustpy.deep._data_utils import get_dataloader
import os


class FullyConnectedBlock(torch.nn.Module):
    """
    Feed Forward Neural Network Block

    Parameters
    ----------
    layers : list
        list of the different layer sizes
    batch_norm : bool
        set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float
        set the amount of dropout you want to use (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: None)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the last layer, if None then it will be linear (default: None)

    Attributes
    ----------
    block: torch.nn.Sequential
        feed forward neural network
    """

    def __init__(self, layers: list, batch_norm: bool = False, dropout: float = None,
                 activation_fn: torch.nn.Module = None, bias: bool = True, output_fn: torch.nn.Module = None):
        super(FullyConnectedBlock, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bias = bias
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        fc_block_list = []
        for i in range(len(layers) - 1):
            fc_block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=self.bias))
            if self.batch_norm:
                fc_block_list.append(torch.nn.BatchNorm1d(layers[i + 1]))
            if self.dropout is not None:
                fc_block_list.append(torch.nn.Dropout(self.dropout))
            if self.activation_fn is not None:
                # last layer is handled differently
                if (i != len(layers) - 2):
                    fc_block_list.append(activation_fn())
                else:
                    if self.output_fn is not None:
                        fc_block_list.append(self.output_fn())

        self.block = torch.nn.Sequential(*fc_block_list)

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


class _AbstractAutoencoder(torch.nn.Module):
    """
    An abstract autoencoder class that can be used by other autoencoder implementations.

    Parameters
    ----------
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)

    Attributes
    ----------
    fitted  : bool
        indicates whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by multiple deep clustering algorithms
    """

    def __init__(self, reusable: bool = True):
        super(_AbstractAutoencoder, self).__init__()
        self.fitted = False
        self.reusable = reusable

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for an encode function of an autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        x : torch.Tensor
            should return the embedded data point
        """
        return x

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

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the loss of a single batch of data.

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, samples, ...)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : (torch.Tensor, torch.Tensor, torch.Tensor)
            the reconstruction loss of the input sample,
            the embedded input sample,
            the reconstructed input sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        batch_data = batch[1].to(device)
        embedded = self.encode(batch_data)
        reconstructed = self.decode(embedded)
        loss = loss_fn(reconstructed, batch_data)
        return loss, embedded, reconstructed

    def evaluate(self, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.modules.loss._Loss,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Evaluates the autoencoder.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on (default: torch.device('cpu'))

        Returns
        -------
        loss: torch.Tensor
            returns the reconstruction loss of all samples in dataloader
        """
        with torch.no_grad():
            self.eval()
            loss = torch.tensor(0.)
            for batch in dataloader:
                new_loss, _, _ = self.loss(batch, loss_fn, device)
                loss += new_loss
            loss /= len(dataloader)
        return loss

    def fit(self, n_epochs: int, optimizer_params: dict, batch_size: int = 128, data: np.ndarray = None,
            data_eval: np.ndarray = None, dataloader: torch.utils.data.DataLoader = None,
            evalloader: torch.utils.data.DataLoader = None, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = {},
            device: torch.device = torch.device("cpu"), model_path: str = None,
            print_step: int = 0) -> '_AbstractAutoencoder':
        """
        Trains the autoencoder in place.

        Parameters
        ----------
        n_epochs : int
            number of epochs for training
        optimizer_params : dict
            parameters of the optimizer, includes the learning rate
        batch_size : int
            size of the data batches (default: 128)
        data : np.ndarray
            train data set. If data is passed then dataloader can remain empty (default: None)
        data_eval : np.ndarray
            evaluation data set. If data_eval is passed then evalloader can remain empty (default: None)
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
            dictionary of the parameters of the scheduler object (default: {})
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        model_path : str
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)
        print_step : int
            specifies how often the losses are printed. If 0, no prints will occur (default: 0)

        Returns
        -------
        self : _AbstractAutoencoder
            this instance of the autoencoder

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        # evalloader has priority over data_eval
        if evalloader is None:
            if data_eval is not None:
                evalloader = get_dataloader(data_eval, batch_size, False)
        optimizer = optimizer_class(params=self.parameters(), **optimizer_params)

        early_stopping = EarlyStopping(patience=patience)
        if scheduler is not None:
            scheduler = scheduler(optimizer=optimizer, **scheduler_params)
            # Depending on the scheduler type we need a different step function call.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_step_scheduler = True
                if evalloader is None:
                    raise ValueError(
                        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, but evalloader is None. Specify evalloader such that validation loss can be computed.")
            else:
                eval_step_scheduler = False
        best_loss = np.inf
        # training loop
        for epoch_i in range(n_epochs):
            self.train()
            for batch in dataloader:
                loss, _, _ = self.loss(batch, loss_fn, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                print(f"Epoch {epoch_i}/{n_epochs - 1} - Batch Reconstruction loss: {loss.item():.6f}")

            if scheduler is not None and not eval_step_scheduler:
                scheduler.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, loss_fn=loss_fn, device=device)
                if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                    print(f"Epoch {epoch_i} EVAL loss total: {val_loss.item():.6f}")
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        # Check if directory exists
                        parent_directory = os.path.dirname(model_path)
                        if parent_directory != "" and not os.path.isdir(parent_directory):
                            os.makedirs(parent_directory)
                        torch.save(self.state_dict(), model_path)

                if early_stopping.early_stop:
                    if print_step > 0:
                        print(f"Stop training at epoch {best_epoch}")
                        print(f"Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                    break
                if scheduler is not None and eval_step_scheduler:
                    scheduler.step(val_loss)
        # change to eval mode after training
        self.eval()
        # Save last version of model
        if evalloader is None and model_path is not None:
            # Check if directory exists
            parent_directory = os.path.dirname(model_path)
            if parent_directory != "" and not os.path.isdir(parent_directory):
                os.makedirs(parent_directory)
            torch.save(self.state_dict(), model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self

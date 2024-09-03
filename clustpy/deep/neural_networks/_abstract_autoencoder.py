"""
@authors:
Lukas Miklautz
"""

import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import encode_batchwise, get_device_from_module
import os
import tqdm
from collections.abc import Callable
from sklearn.utils import check_random_state
from clustpy.deep._utils import set_torch_seed


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
    layer_positions : list
        ids specifying at which positions the input layers are contained within block
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

        layer_positions = []
        fc_block_list = []
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
                    fc_block_list.append(activation_fn())
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


class _AbstractAutoencoder(torch.nn.Module):
    """
    An abstract autoencoder class that can be used by other autoencoder implementations.

    Parameters
    ----------
    work_on_copy : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)


    Attributes
    ----------
    fitted  : bool
        indicates whether the autoencoder is already fitted
    work_on_copy : bool
        indicates whether deep clustering algorithms should work on a copy of the original autoencoder
    """

    def __init__(self, work_on_copy: bool = True, random_state: np.random.RandomState | int = None):
        super(_AbstractAutoencoder, self).__init__()
        self.fitted = False
        self.work_on_copy = work_on_copy
        self.random_state = check_random_state(random_state)

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

    def loss(self, batch: list, ssl_loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
             corruption_fn: Callable = None) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the loss of a single batch of data.

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, samples, ...)
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on
        corruption_fn : Callable
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)

        Returns
        -------
        loss : (torch.Tensor, torch.Tensor, torch.Tensor)
            the reconstruction loss of the input sample,
            the embedded input sample,
            the reconstructed input sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        batch_data = batch[1].to(device)
        batch_data_adj = batch_data if corruption_fn is None else corruption_fn(batch_data)
        embedded = self.encode(batch_data_adj)
        reconstructed = self.decode(embedded)
        loss = ssl_loss_fn(reconstructed, batch_data)
        return loss, embedded, reconstructed

    def loss_augmentation(self, batch: list, ssl_loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
                          corruption_fn: Callable = None) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the loss of a single batch of data and an augmented version of the data.
        Note that the augmented samples come at position batch[1] and the original samples at batch[2].

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, augmented samples, original samples, ...)
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on
        corruption_fn : Callable
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)

        Returns
        -------
        loss : (torch.Tensor, torch.Tensor, torch.Tensor)
            the combined network loss of the sample and the augmented sample,
            the embedded input sample,
            the reconstructed input sample,
            the embedded augmented sample,
            the reconstructed augmented sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        # First entry (batch[0]) are the indices, second entry (batch[1]) are the augmented samples, third entry (batch[2]) are the original samples
        # If additional inputs are used in the dataloader, entries at an uneven position (batch[3], batch[5], ...) are augmented and entries at even positions (batch[4], batch[6], ...) original
        # Considering also the additional inputs can be relevant, e.g., when using a NeighborEncoder
        batches_orig = [batch[i] for i in range(2, len(batch), 2)]
        batches_aug = [batch[i] for i in range(1, len(batch), 2)]
        loss_orig, embedded, reconstructed = self.loss([batch[0]] + batches_orig, ssl_loss_fn, device, corruption_fn)
        loss_augmented, embedded_aug, reconstructed_aug = self.loss([batch[0]] + batches_aug, ssl_loss_fn, device,
                                                                    corruption_fn)
        loss_total = (loss_orig + loss_augmented) / 2
        return loss_total, embedded, reconstructed, embedded_aug, reconstructed_aug

    def evaluate(self, dataloader: torch.utils.data.DataLoader, ssl_loss_fn: torch.nn.modules.loss._Loss,
                 device: torch.device) -> torch.Tensor:
        """
        Evaluates the autoencoder.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on

        Returns
        -------
        loss: torch.Tensor
            returns the reconstruction loss of all samples in dataloader
        """
        with torch.no_grad():
            self.eval()
            loss = torch.tensor(0.)
            for batch in dataloader:
                new_loss, _, _ = self.loss(batch, ssl_loss_fn, device)
                loss += new_loss
            loss /= len(dataloader)
        return loss

    def fit(self, n_epochs: int = 100, optimizer_params: dict = None, batch_size: int = 128,
            data: np.ndarray | torch.Tensor = None, data_eval: np.ndarray | torch.Tensor = None,
            dataloader: torch.utils.data.DataLoader = None, evalloader: torch.utils.data.DataLoader = None,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = {},
            corruption_fn: Callable = None, model_path: str = None) -> '_AbstractAutoencoder':
        """
        Trains the autoencoder in place.

        Parameters
        ----------
        n_epochs : int
            number of epochs for training (default: 100)
        optimizer_params : dict
            parameters of the optimizer, includes the learning rate (default: {"lr": 1e-3})
        batch_size : int
            size of the data batches (default: 128)
        data : np.ndarray | torch.Tensor
            train data set. If data is passed then dataloader can remain empty (default: None)
        data_eval : np.ndarray | torch.Tensor
            evaluation data set. If data_eval is passed then evalloader can remain empty (default: None)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training (default: default=None)
        evalloader : torch.utils.data.DataLoader
            dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau (default: None)
        optimizer_class : torch.optim.Optimizer
            optimizer to be used (default: torch.optim.Adam)
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss (default: torch.nn.MSELoss())
        patience : int
            patience parameter for EarlyStopping (default: 5)
        scheduler : torch.optim.lr_scheduler
            learning rate scheduler that should be used.
            If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader (default: None)
        scheduler_params : dict
            dictionary of the parameters of the scheduler object (default: {})
        corruption_fn : Callable
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)
        model_path : str
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)

        Returns
        -------
        self : _AbstractAutoencoder
            this instance of the autoencoder

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        set_torch_seed(self.random_state)
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        # evalloader has priority over data_eval
        if evalloader is None:
            if data_eval is not None:
                evalloader = get_dataloader(data_eval, batch_size, False)
        optimizer_params = {"lr": 1e-3} if optimizer_params is None else optimizer_params
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
        device = get_device_from_module(self)
        tbar = tqdm.trange(n_epochs, desc="AE training")
        for epoch_i in tbar:
            self.train()
            total_loss = 0
            for batch in dataloader:
                loss, _, _ = self.loss(batch, ssl_loss_fn, device, corruption_fn)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Training Loss": total_loss}

            if scheduler is not None and not eval_step_scheduler:
                scheduler.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, ssl_loss_fn=ssl_loss_fn, device=device)
                postfix_str["Eval Loss"] = val_loss.item()
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        self.save_parameters(model_path)
                if early_stopping.early_stop:
                    print(f"Stop training at epoch {best_epoch}. Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                if scheduler is not None and eval_step_scheduler:
                    scheduler.step(val_loss)
            tbar.set_postfix(postfix_str)
        # change to eval mode after training
        self.eval()
        # Save last version of model
        if evalloader is None and model_path is not None:
            self.save_parameters(model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self

    def save_parameters(self, path: str) -> None:
        """
        Save the current state_dict of the model.

        Parameters
        ----------
        path : str
            Path where the state_dict should be stored
        """
        # Check if directory exists
        parent_directory = os.path.dirname(path)
        if parent_directory != "" and not os.path.isdir(parent_directory):
            os.makedirs(parent_directory)
        torch.save(self.state_dict(), path)

    def load_parameters(self, path: str) -> '_AbstractAutoencoder':
        """
        Load a state_dict into the current model to set its parameters.

        Parameters
        ----------
        path : str
            Path from where the state_dict should be loaded

        Returns
        -------
        self : _AbstractAutoencoder
            this instance of the autoencoder
        """
        self.load_state_dict(torch.load(path))
        self.fitted = True
        return self

    def transform(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Embed the given data set using the trained autoencoder.

        Parameters
        ----------
        X: np.ndarray
            The given data set
        batch_size : int
            size of the data batches

        Returns
        -------
        X_embed : np.ndarray
            The embedded data set
        """
        dataloader = get_dataloader(X, batch_size, False, False)
        X_embed = encode_batchwise(dataloader, self)
        return X_embed

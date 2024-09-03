"""
@authors:
Collin Leiber
"""

import torch
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from clustpy.deep._utils import get_device_from_module
from clustpy.deep._data_utils import get_dataloader
import numpy as np
import tqdm
from collections.abc import Callable
from clustpy.deep._utils import set_torch_seed


class StackedAutoencoder(FeedforwardAutoencoder):
    """
    A stacked autoencoder.
    Regarding its architecture, it corresponds to a standard FeedforwardAutoencoder but uses a different training strategy.
    First, each layer is trained separately in a greedy manner, referred to as layer-wise training.
    Afterward, all layers are trained at once to finetune the autoencoder.

    Parameters
    ----------
    layers : list
        list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the embedding dimension.
        Note that in case of a StackedAutoencoder the decoder requires the reversed structure of the encoder.
    batch_norm : bool
        Set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float
        Set the amount of dropout you want to use (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)
    work_on_copy : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    encoder : FullyConnectedBlock
        encoder part of the autoencoder, responsible for embedding data points (class is FullyConnectedBlock)
    decoder : FullyConnectedBlock
        decoder part of the autoencoder, responsible for reconstructing data points from the embedding (class is FullyConnectedBlock)
    fitted  : bool
        indicates whether the autoencoder is already fitted
    work_on_copy : bool
        indicates whether deep clustering algorithms should work on a copy of the original autoencoder

    References
    ----------
    E.g.:
    Bengio, Yoshua, et al. "Greedy layer-wise training of deep networks."
    Advances in neural information processing systems 19 (2006).
    
    or

    Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion."
    Journal of machine learning research 11.12 (2010).
    """

    def __init__(self, layers: list, batch_norm: bool = False, dropout: float = None,
                 activation_fn: torch.nn.Module = torch.nn.LeakyReLU, bias: bool = True,
                 decoder_output_fn: torch.nn.Module = None, work_on_copy: bool = True,
                 random_state: np.random.RandomState | int = None):
        super().__init__(layers, batch_norm, dropout, activation_fn, bias, None, decoder_output_fn, work_on_copy,
                         random_state)

    def layerwise_training(self, n_epochs_per_layer: int = 20, optimizer_params: dict = None, batch_size: int = 128,
                           data: np.ndarray | torch.Tensor = None, dataloader: torch.utils.data.DataLoader = None,
                           optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                           ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                           corruption_fn: Callable = None) -> 'StackedAutoencoder':
        """
        Trains the autoencoder in a greedy layer-wise fashion.

        Parameters
        ----------
        n_epochs_per_layer : int
            number of epochs for training each layer separately (default: 20)
        optimizer_params : dict
            parameters of the optimizer, includes the learning rate (default: {"lr": 1e-3})
        batch_size : int
            size of the data batches (default: 128)
        data : np.ndarray | torch.Tensor
            train data set. If data is passed then dataloader can remain empty (default: None)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training (default: default=None)
        optimizer_class : torch.optim.Optimizer
            optimizer to be used (default: torch.optim.Adam)
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss (default: torch.nn.MSELoss())
        corruption_fn : Callable
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)

        Returns
        -------
        self : StackedAutoencoder
            this instance of the autoencoder

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        optimizer_params = {"lr": 1e-3} if optimizer_params is None else optimizer_params
        optimizer = optimizer_class(params=self.parameters(), **optimizer_params)
        device = get_device_from_module(self)
        encoder_linear_layer_ids = self.encoder.layer_positions
        decoder_linear_layer_ids = self.decoder.layer_positions
        assert len(encoder_linear_layer_ids) == len(
            decoder_linear_layer_ids), "The decoder must be a reversed version of the encoder"
        # Start training
        tbar = tqdm.tqdm(total=n_epochs_per_layer * len(encoder_linear_layer_ids), desc="Stacked AE training")
        for layer_nr, encoder_layer_to_train in enumerate(encoder_linear_layer_ids):
            # Train this specific layer for a certain amount of epochs
            for _ in range(n_epochs_per_layer):
                total_loss = 0
                for batch in dataloader:
                    input_data = batch[1].to(device)
                    with torch.no_grad():
                        # encode batch using already trained layers including non-linearity functions etc
                        for encode_layer in range(encoder_layer_to_train):
                            input_data = self.encoder.block[encode_layer](input_data)
                    # Calculate loss regarding current layer
                    input_data_adj = input_data if corruption_fn is None else corruption_fn(input_data)
                    encoded = self.encoder.block[encoder_layer_to_train](input_data_adj)
                    decoded = self.decoder.block[decoder_linear_layer_ids[-(layer_nr + 1)]](encoded)
                    loss = ssl_loss_fn(decoded, input_data)
                    # Update network
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                postfix_str = {"Loss": total_loss, "LayerID": layer_nr}
                tbar.set_postfix(postfix_str)
                tbar.update()
        return self

    def fit(self, n_epochs_per_layer: int = 20, n_epochs: int = 100, optimizer_params: dict = None,
            batch_size: int = 128, data: np.ndarray | torch.Tensor = None, data_eval: np.ndarray | torch.Tensor = None,
            dataloader: torch.utils.data.DataLoader = None, evalloader: torch.utils.data.DataLoader = None,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = {},
            corruption_fn: Callable = None, model_path: str = None) -> 'StackedAutoencoder':
        """
        Trains the autoencoder in place.
        First, a greedy layer-wise training is performed. Afterward, the weights are finetuned by training all layer simultaneously.

        Parameters
        ----------
        n_epochs_per_layer : int
            number of epochs for training each layer separately (default: 20)
        n_epochs: int
            number of epochs for the final finetuning (default: 100)
        optimizer_params : dict
            parameters of the optimizer, includes the learning rate (default: {"lr": 1e-3})
        batch_size : int
            size of the data batches (default: 128)
        data : np.ndarray | torch.Tensor
            train data set. If data is passed then dataloader can remain empty (default: None)
        data_eval : np.ndarray | torch.Tensor
            evaluation data set. If data_eval is passed then evalloader can remain empty.
            Only used for finetuning (default: None)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training (default: default=None)
        evalloader : torch.utils.data.DataLoader
            dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau.
            Only used for finetuning (default: None)
        optimizer_class : torch.optim.Optimizer
            optimizer to be used (default: torch.optim.Adam)
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss (default: torch.nn.MSELoss())
        patience : int
            patience parameter for EarlyStopping.
            Only used for finetuning (default: 5)
        scheduler : torch.optim.lr_scheduler
            learning rate scheduler that should be used.
            If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader.
            Only used for finetuning (default: None)
        scheduler_params : dict
            dictionary of the parameters of the scheduler object.
            Only used for finetuning (default: {})
        corruption_fn : Callable
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)
        model_path : str
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)

        Returns
        -------
        self : StackedAutoencoder
            this instance of the autoencoder

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        set_torch_seed(self.random_state)
        self.layerwise_training(n_epochs_per_layer, optimizer_params, batch_size, data, dataloader,
                                optimizer_class, ssl_loss_fn, corruption_fn)
        super().fit(n_epochs, optimizer_params, batch_size, data, data_eval, dataloader, evalloader,
                    optimizer_class, ssl_loss_fn, patience, scheduler, scheduler_params, corruption_fn, model_path)
        return self

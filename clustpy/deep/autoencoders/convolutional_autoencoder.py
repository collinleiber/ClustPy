import torch
import numpy as np
from .._early_stopping import EarlyStopping
from .._data_utils import get_dataloader
from .resnet_ae_modules import resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder
from .flexible_autoencoder import FullyConnectedBlock
from torchvision.models._api import Weights

_VALID_CONV_MODULES = {
    "resnet18": {
        "enc": resnet18_encoder,
        "dec": resnet18_decoder,
    },
    "resnet50": {
        "enc": resnet50_encoder,
        "dec": resnet50_decoder,
    },
}

class ConvolutionalAutoencoder(torch.nn.Module):
    """
    A convolutional autoencoder based on the ResNet architecture.
    
    Parameters
    ----------
    input_height: int
        height of the images for the decoder (assume square images)
    fc_layers : list
        list of the different layer sizes from flattened convolutional layer input to embedding. First input needs to be 512 if conv_encoder="resnet18" and 2048 if conv_encoder="resnet50" .
        If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    fc_decoder_layers : list
        list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers. (default=None)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)
    pretrained_encoder_weights : torchvision.models._api.Weights
        weights from torch.vision.models, indicates whether pretrained resnet weights should be used for the encoder. (default: None)
    pretrained_decoder_weights : torchvision.models._api.Weights
        weights from torch.vision.models, indicates whether pretrained resnet weights should be used for the decoder (not implemented yet). (default: None)
    fc_kwargs : dict
        additional parameters for FullyConnectedBlock
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    
    Attributes
    ----------
    conv_encoder: convolutional resnet encoder part of the autoencoder
    conv_decoder: convolutional resnet decoder part of the autoencoder
    fc_encoder : FullyConnectedBlock
        fully connected encoder part of the convolutional autoencoder, together with conv_encoder is responsible for embedding data points
    fc_decoder : FullyConnectedBlock
        fully connected decoder part of the convolutional autoencoder, together with conv_decoder is responsible for reconstructing data points from the embedding
    fitted  : bool
        indicates whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by mutliple deep clustering algorithms
        
    References
    ----------
    Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>
    E.g. Ballard, Dana H. "Modular learning in neural networks." Aaai. Vol. 647. 1987.
    """

    def __init__(self, input_height: int, fc_layers: list, conv_encoder: str = "resnet18", conv_decoder: str = None, activation_fn: torch.nn.Module = torch.nn.ReLU, 
                 fc_decoder_layers: list=None, decoder_output_fn: torch.nn.Module = None, pretrained_encoder_weights: Weights = None, pretrained_decoder_weights: Weights = None, reusable: bool = True, **fc_kwargs):
        super().__init__()
        self.fitted = False
        self.input_height = input_height
        self.reusable = reusable

        # Check if layers match
        if fc_layers[0] not in [512, 2048]:
            raise ValueError(f"First input in fc_layers needs to be 512 or 2048, but is fc_layers[0] = {fc_layers[0]}")
        if fc_decoder_layers is None:
            fc_decoder_layers = fc_layers[::-1]
        if (fc_layers[-1] != fc_decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {layers[-1]} and {fc_decoder_layers[0]} respectively.")
        if (fc_layers[0] != fc_decoder_layers[-1]):
            raise ValueError(
                f"Output and input dimension do not match, they are {fc_layers[0]} and {fc_decoder_layers[-1]} respectively.")

        # Setup convolutional encoder and decoder
        if conv_encoder in _VALID_CONV_MODULES:
            self.conv_encoder = _VALID_CONV_MODULES[conv_encoder]["enc"](first_conv=True, maxpool1=True, pretrained_weights=pretrained_encoder_weights)
            if conv_decoder is None:
                conv_decoder = conv_encoder
            if conv_decoder in _VALID_CONV_MODULES:
                self.conv_decoder = _VALID_CONV_MODULES[conv_decoder]["dec"](latent_dim=fc_decoder_layers[-1], input_height=self.input_height, first_conv=True, maxpool1=True, pretrained_weights=pretrained_decoder_weights)
            else:
                raise ValueError(f"value for conv_decoder={conv_decoder} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}")
        else:
            raise ValueError(f"value for conv_encoder={conv_encoder} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}")

        # Initialize encoder
        self.fc_encoder = FullyConnectedBlock(layers=fc_layers, activation_fn=activation_fn, output_fn=None, **fc_kwargs)
        # Inverts the list of layers to make symmetric version of the encoder
        self.fc_decoder = FullyConnectedBlock(layers=fc_decoder_layers, activation_fn=activation_fn, output_fn=decoder_output_fn, **fc_kwargs)

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
        embedded = self.conv_encoder(x)
        embedded =  self.fc_encoder(embedded)
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
        decoded = self.fc_decoder(embedded)
        decoded =  self.conv_decoder(decoded)
        return decoded

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

    def loss(self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
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

        Returns
        -------
        loss : torch.Tensor
            returns the reconstruction loss of the input sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        batch_data = batch[1].to(device)
        reconstruction = self.forward(batch_data)
        loss = loss_fn(reconstruction, batch_data)
        return loss

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
            loss = torch.tensor(0)
            for batch in dataloader:
                loss += self.loss(batch, loss_fn, device)
            loss /= len(dataloader)
        return loss
    
    def fit(self, n_epochs: int, lr: float, batch_size: int = 128, data: np.ndarray = None,
            data_eval: np.ndarray = None,
            dataloader: torch.utils.data.DataLoader = None, evalloader: torch.utils.data.DataLoader = None,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), patience: int = 5,
            scheduler: torch.optim.lr_scheduler = None, scheduler_params: dict = None,
            device: torch.device = torch.device("cpu"), model_path: str = None,
            print_step: int = 0) -> 'ConvolutionalAutoencoder':
        """
        Trains the autoencoder in place.
        
        Parameters
        ----------
        n_epochs : int
            number of epochs for training
        lr : float
            learning rate to be used for the optimizer_class
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
            dictionary of the parameters of the scheduler object (default: None)
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        model_path : str
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)
        print_step : int
            specifies how often the losses are printed. If 0, no prints will occur (default: 0)

        Returns
        -------
        self : FlexibleAutoencoder
            this instance of the FlexibleAutoencoder

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
        params_dict = {'params': self.parameters(), 'lr': lr}
        optimizer = optimizer_class(**params_dict)

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
                loss = self.loss(batch, loss_fn, device)
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
            torch.save(self.state_dict(), model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self
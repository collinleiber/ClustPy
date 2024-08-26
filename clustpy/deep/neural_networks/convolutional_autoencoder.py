import torch
from clustpy.deep.neural_networks._resnet_ae_modules import resnet18_encoder, resnet18_decoder, resnet50_encoder, \
    resnet50_decoder, ResNetEncoder
from clustpy.deep.neural_networks._abstract_autoencoder import FullyConnectedBlock, _AbstractAutoencoder
from torchvision.models._api import Weights
import numpy as np

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

_CONV_MODULES_INPUT_DIM = {"resnet18": 512, "resnet50": 2048}


class ConvolutionalAutoencoder(_AbstractAutoencoder):
    """
    A convolutional autoencoder based on the ResNet architecture.
    
    Parameters
    ----------
    input_height: int
        height of the images for the decoder (assume square images)
    fc_layers : list
        list of the different layer sizes from flattened convolutional layer input to embedding. First input needs to be 512 if conv_encoder_name="resnet18" and 2048 if conv_encoder_name="resnet50".
        If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    conv_encoder_name : str
        name of convolutional resnet encoder part of the autoencoder. Can be 'resnet18' or 'resnet50' (default: 'resnet18')
    conv_decoder_name : str
        name of convolutional resnet decoder part of the autoencoder. Can be 'resnet18' or 'resnet50'. If None it will be the same as conv_encoder_name (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    fc_decoder_layers : list
        list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers (default: None)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)
    pretrained_encoder_weights : torchvision.models._api.Weights
        weights from torch.vision.models, indicates whether pretrained resnet weights should be used for the encoder. (default: None)
    pretrained_decoder_weights : torchvision.models._api.Weights
        weights from torch.vision.models, indicates whether pretrained resnet weights should be used for the decoder (not implemented yet). (default: None)
    work_on_copy : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    fc_kwargs : dict
        additional parameters for FullyConnectedBlock
    
    Attributes
    ----------
    conv_encoder : ResNetEncoder
        convolutional resnet encoder part of the autoencoder
    conv_decoder : ResNetEncoder
        convolutional resnet decoder part of the autoencoder
    fc_encoder : FullyConnectedBlock
        fully connected encoder part of the convolutional autoencoder, together with conv_encoder is responsible for embedding data points
    fc_decoder : FullyConnectedBlock
        fully connected decoder part of the convolutional autoencoder, together with conv_decoder is responsible for reconstructing data points from the embedding
    fitted  : bool
        indicates whether the autoencoder is already fitted
    work_on_copy : bool
        indicates whether deep clustering algorithms should work on a copy of the original autoencoder
        
    References
    ----------
    He, Kaiming, et al. "Deep residual learning for image recognition."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

    and

    LeCun, Yann, et al. "Backpropagation applied to handwritten zip code recognition."
    Neural computation 1.4 (1989): 541-551.
    """

    def __init__(self, input_height: int, fc_layers: list, conv_encoder_name: str = "resnet18",
                 conv_decoder_name: str = None, activation_fn: torch.nn.Module = torch.nn.ReLU,
                 fc_decoder_layers: list = None, decoder_output_fn: torch.nn.Module = None,
                 pretrained_encoder_weights: Weights = None, pretrained_decoder_weights: Weights = None,
                 work_on_copy: bool = True, random_state: np.random.RandomState | int = None, **fc_kwargs):
        super().__init__(work_on_copy, random_state)
        self.input_height = input_height

        # Check if layers match
        if fc_decoder_layers is None:
            fc_decoder_layers = fc_layers[::-1]
        if (fc_layers[-1] != fc_decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {fc_layers[-1]} and {fc_decoder_layers[0]} respectively.")

        # Setup convolutional encoder and decoder
        if conv_decoder_name is None:
            conv_decoder_name = conv_encoder_name
        if conv_encoder_name in _VALID_CONV_MODULES:
            if fc_layers[0] != _CONV_MODULES_INPUT_DIM[conv_encoder_name]:
                raise ValueError(
                    f"First input in fc_layers needs to be {_CONV_MODULES_INPUT_DIM[conv_encoder_name]} for {conv_encoder_name} architecture, but is fc_layers[0] = {fc_layers[0]}")
            self.conv_encoder = _VALID_CONV_MODULES[conv_encoder_name]["enc"](first_conv=True, maxpool1=True,
                                                                              pretrained_weights=pretrained_encoder_weights)
        else:
            raise ValueError(
                f"value for conv_encoder_name={conv_encoder_name} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}")
        if conv_decoder_name in _VALID_CONV_MODULES:
            # if fc_decoder_layers[-1] != _CONV_MODULES_INPUT_DIM[conv_decoder_name]:
            #     raise ValueError(
            #         f"Last input in fc_decoder_layers needs to be {_CONV_MODULES_INPUT_DIM[conv_decoder_name]} for {conv_decoder_name} architecture, but is fc_decoder_layers[0] = {fc_decoder_layers[-1]}")
            self.conv_decoder = _VALID_CONV_MODULES[conv_decoder_name]["dec"](latent_dim=fc_decoder_layers[-1],
                                                                              input_height=self.input_height,
                                                                              first_conv=True, maxpool1=True,
                                                                              pretrained_weights=pretrained_decoder_weights)
        else:
            raise ValueError(
                f"value for conv_decoder_name={conv_decoder_name} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}")

        # Initialize encoder
        self.fc_encoder = FullyConnectedBlock(layers=fc_layers, activation_fn=activation_fn, output_fn=None,
                                              **fc_kwargs)
        # Inverts the list of layers to make symmetric version of the encoder
        self.fc_decoder = FullyConnectedBlock(layers=fc_decoder_layers, activation_fn=activation_fn,
                                              output_fn=decoder_output_fn, **fc_kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x. Runs x through the conv_encoder and then the fc_encoder.

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
        embedded = self.fc_encoder(embedded)
        return embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded. Runs x through the conv_decoder and then the fc_decoder.

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
        decoded = self.conv_decoder(decoded)
        return decoded

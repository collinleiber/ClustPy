import torch
import numpy as np
from clustpy.deep._utils import get_device_from_module
from clustpy.deep.neural_networks._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.neural_networks._abstract_neural_network import _AbstractNeuralNetwork


def encode_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: _AbstractNeuralNetwork) -> np.ndarray:
    """
    Utility function for embedding the whole data set in a mini-batch fashion

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        data to embed
    neural_network : _AbstractNeuralNetwork
        the neural network that is used for the encoding (e.g. an autoencoder)

    Returns
    -------
    embeddings_numpy : np.ndarray
        The embedded data set
    """
    device = get_device_from_module(neural_network)
    embeddings_numpy = None
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedded_data = neural_network.encode(batch_data)
        # In case encode() returns more than one value (e.g., for a variational autoencoder), we will pick the first
        if type(embedded_data) is tuple:
            embedded_data = embedded_data[0]
        if embeddings_numpy is None:
            assert hasattr(dataloader.dataset, '__len__'), "The dataloader must have a dataset attribute to determine the size of the data set."
            embeddings_numpy = np.zeros([len(dataloader.dataset)] + list(embedded_data.shape[1:]), dtype=float)
        embeddings_numpy[batch[0]] = embedded_data.detach().cpu().numpy()
    assert embeddings_numpy is not None, "The dataloader must have at least one batch to embed the data set."
    return embeddings_numpy


def decode_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: _AbstractAutoencoder) -> np.ndarray:
    """
    Utility function for decoding the whole data set in a mini-batch fashion, e.g., with an autoencoder.
    Note: Assumes an implemented decode function

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        data to decode
    neural_network : _AbstractAutoencoder
        the neural network that is used for the decoding (e.g. an autoencoder)

    Returns
    -------
    decodings_numpy : np.ndarray
        The decoded data set
    """
    device = get_device_from_module(neural_network)
    decodings_numpy = None
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedded_data = neural_network.encode(batch_data)
        # In case encode() returns more than one value (e.g., for a variational autoencoder), we all of them will be used for decoding
        if type(embedded_data) is tuple:
            decoded_data = neural_network.decode(*embedded_data)
        else:
            decoded_data = neural_network.decode(embedded_data)
        if decodings_numpy is None:
            assert hasattr(dataloader.dataset, '__len__'), "The dataloader must have a dataset attribute to determine the size of the data set."
            decodings_numpy = np.zeros([len(dataloader.dataset)] + list(decoded_data.shape[1:]), dtype=float)
        decodings_numpy[batch[0]] = decoded_data.detach().cpu().numpy()
    assert decodings_numpy is not None, "The dataloader must have at least one batch to decode the data set."
    return decodings_numpy


def encode_decode_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: _AbstractAutoencoder) -> tuple[
        np.ndarray, np.ndarray]:
    """
    Utility function for encoding and decoding the whole data set in a mini-batch fashion, e.g., with an autoencoder.
    Note: Assumes an implemented decode function

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    neural_network : _AbstractAutoencoder
        the neural network that is used for the encoding and decoding (e.g. an autoencoder)

    Returns
    -------
    tuple : tuple[np.ndarray, np.ndarray]
        The embedded data set,
        The decoded data set
    """
    device = get_device_from_module(neural_network)
    embeddings_numpy = None
    decodings_numpy = None
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedded_data = neural_network.encode(batch_data)
        # In case encode() returns more than one value (e.g., for a variational autoencoder), we all of them will be used for decoding
        if isinstance(embedded_data, tuple):
            decoded_data = neural_network.decode(*embedded_data)
            embedded_data = embedded_data[0]
        else:
            decoded_data = neural_network.decode(embedded_data)
        if embeddings_numpy is None or decodings_numpy is None:
            assert hasattr(dataloader.dataset, '__len__'), "The dataloader must have a dataset attribute to determine the size of the data set."
            embeddings_numpy = np.zeros([len(dataloader.dataset)] + list(embedded_data.shape[1:]), dtype=float)
            decodings_numpy = np.zeros([len(dataloader.dataset)] + list(decoded_data.shape[1:]), dtype=float)
        embeddings_numpy[batch[0]] = embedded_data.detach().cpu().numpy()
        decodings_numpy[batch[0]] = decoded_data.detach().cpu().numpy()
    assert embeddings_numpy is not None and decodings_numpy is not None, "The dataloader must have at least one batch to embed the data set."
    return embeddings_numpy, decodings_numpy


def predict_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: _AbstractNeuralNetwork,
                      cluster_module: torch.nn.Module) -> np.ndarray:
    """
    Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion.
    Method calls the predict_hard method of the cluster_module for each batch of data.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    neural_network : _AbstractNeuralNetwork
        the neural network that is used for the encoding (e.g. an autoencoder)
    cluster_module : torch.nn.Module
        the cluster module that is used for the encoding (e.g. DEC). Usually contains the predict method.

    Returns
    -------
    predictions_numpy : np.ndarray
        The predictions of the cluster_module for the data set
    """
    device = get_device_from_module(neural_network)
    assert hasattr(dataloader.dataset, '__len__'), "The dataloader must have a dataset attribute to determine the size of the data set."
    predictions_numpy = np.zeros(len(dataloader.dataset), dtype=np.int32)
    for batch in dataloader:
        batch_data = batch[1].to(device)
        encoded_data = neural_network.encode(batch_data)
        if isinstance(encoded_data, tuple):
            encoded_data = encoded_data[0]
        predict_hard = getattr(cluster_module, 'predict_hard', None)
        assert callable(predict_hard), "The cluster_module must have a callable predict_hard method to predict the labels of the input data."
        prediction = predict_hard(encoded_data).detach().cpu()
        predictions_numpy[batch[0]] = prediction
    return predictions_numpy

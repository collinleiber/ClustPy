from sklearn.base import ClusterMixin
import inspect
import torch
from itertools import islice
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import os
import subprocess


def set_torch_seed(random_state: np.random.RandomState | int) -> None:
    """
    Set the random state for torch applications.

    Parameters
    ----------
    random_state : np.random.RandomState | int
        use a fixed random state or an integer to get a repeatable solution
    """
    if type(random_state) is int:
        seed = random_state
    elif type(random_state) is np.random.RandomState:
        seed = random_state.randint(np.iinfo(np.int32).max)
    else:
        raise ValueError("random_state must be of type int or np.random.RandomState")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def squared_euclidean_distance(tensor1: torch.Tensor, tensor2: torch.Tensor,
                               weights: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the pairwise squared Euclidean distance between two tensors.
    Each row in the tensors is interpreted as a separate object, while each column represents its features.
    Therefore, the result of an (4x3) and (12x3) tensor will be a (4x12) tensor.
    Optionally, features can be individually weighted.
    The default behavior is that all features are weighted by 1.

    Parameters
    ----------
    tensor1 : torch.Tensor
        the first tensor
    tensor2 : torch.Tensor
        the second tensor
    weights : torch.Tensor
        tensor containing the weights of the features (default: None)

    Returns
    -------
    squared_diffs : torch.Tensor
        the pairwise squared Euclidean distances
    """
    assert tensor1.shape[1] == tensor2.shape[1], "The number of features of the two input tensors must match."
    ta = tensor1.unsqueeze(1)
    tb = tensor2.unsqueeze(0)
    squared_diffs = (ta - tb)
    if weights is not None:
        assert tensor1.shape[1] == weights.shape[0]
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).sum(2)
    return squared_diffs


def detect_device(device: torch.device | int | str = None) -> torch.device:
    """
    Automatically detects if you have a cuda enabled GPU.
    Device can also be read from environment variable "CLUSTPY_DEVICE".
    It can be set using, e.g., os.environ["CLUSTPY_DEVICE"] = "cuda:1"

    Parameters
    ----------
    device : torch.device | int | str
        the input device. Will be returned if it is not None (default: None)

    Returns
    -------
    device : torch.device
        device on which the prediction should take place
    """
    if device == -1:
        # Special case
        device = torch.device('cpu')
    elif device is None:
        env_device = os.environ.get("CLUSTPY_DEVICE", None)
        # Check if environment device is None - in that case CLUSTPY_DEVICE has not been specified
        if env_device is None:
            if torch.cuda.is_available():
                # Try to automatically identify best GPU
                try:
                    shell_output = (subprocess.check_output("nvidia-smi -q -d Utilization |grep Memory", shell=True)).decode('utf-8')[:-1]
                    entries = shell_output.split("\n")[::2]
                    used_memory = [int(e.split(":")[1].replace(" %", "")) for e in entries]
                    device = torch.device("cuda:{0}".format(np.argmin(used_memory)))
                    print(device, "was automatically chosen as device for the computation.")
                except Exception:
                    # Default: Use first available GPU
                    device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(env_device)
    elif type(device) is int or type(device) is str:
        device = torch.device(device)
    return device


def get_device_from_module(neural_network: torch.nn.Module) -> torch.device:
    """
    Get the device from a given module.

    Parameters
    ----------
    neural_network : torch.nn.Module
        the neural network that is used for the encoding (e.g. an autoencoder)

    Returns
    -------
    device : torch.device
        device of the module
    """
    example_param = next(neural_network.parameters())
    if example_param.is_cuda:
        device = torch.device('cuda:' + str(example_param.get_device()))
    else:
        device = torch.device('cpu')
    return device


def encode_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: torch.nn.Module) -> np.ndarray:
    """
    Utility function for embedding the whole data set in a mini-batch fashion

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    neural_network : torch.nn.Module
        the neural network that is used for the encoding (e.g. an autoencoder)

    Returns
    -------
    embeddings_numpy : np.ndarray
        The embedded data set
    """
    device = get_device_from_module(neural_network)
    embeddings = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedded_data = neural_network.encode(batch_data)
        # In case encode() returns more than one value (e.g., for a variational autoencoder), we will pick the first
        if type(embedded_data) is tuple:
            embedded_data = embedded_data[0]
        embeddings.append(embedded_data.detach().cpu())
    embeddings_numpy = torch.cat(embeddings, dim=0).numpy()
    return embeddings_numpy


def decode_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: torch.nn.Module) -> np.ndarray:
    """
    Utility function for decoding the whole data set in a mini-batch fashion, e.g., with an autoencoder.
    Note: Assumes an implemented decode function

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    neural_network : torch.nn.Module
        the neural network that is used for the decoding (e.g. an autoencoder)
    device : torch.device
        device to be trained on

    Returns
    -------
    reconstructions_numpy : np.ndarray
        The reconstructed data set
    """
    device = get_device_from_module(neural_network)
    reconstructions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedded_data = neural_network.encode(batch_data)
        # In case encode() returns more than one value (e.g., for a variational autoencoder), we all of them will be used for decoding
        if type(embedded_data) is tuple:
            decoded_data = neural_network.decode(*embedded_data)
        else:
            decoded_data = neural_network.decode(embedded_data)
        reconstructions.append(decoded_data.detach().cpu())
    reconstructions_numpy = torch.cat(reconstructions, dim=0).numpy()
    return reconstructions_numpy


def encode_decode_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: torch.nn.Module) -> (
        np.ndarray, np.ndarray):
    """
    Utility function for encoding and decoding the whole data set in a mini-batch fashion, e.g., with an autoencoder.
    Note: Assumes an implemented decode function

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    neural_network : torch.nn.Module
        the neural network that is used for the encoding and decoding (e.g. an autoencoder)

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The embedded data set,
        The reconstructed data set
    """
    device = get_device_from_module(neural_network)
    embeddings = []
    reconstructions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedding = neural_network.encode(batch_data)
        embeddings.append(embedding.detach().cpu())
        reconstructions.append(neural_network.decode(embedding).detach().cpu())
    embeddings_numpy = torch.cat(embeddings, dim=0).numpy()
    reconstructions_numpy = torch.cat(reconstructions, dim=0).numpy()
    return embeddings_numpy, reconstructions_numpy


def predict_batchwise(dataloader: torch.utils.data.DataLoader, neural_network: torch.nn.Module,
                      cluster_module: torch.nn.Module) -> np.ndarray:
    """
    Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion.
    Method calls the predict_hard method of the cluster_module for each batch of data.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    neural_network : torch.nn.Module
        the neural network that is used for the encoding (e.g. an autoencoder)
    cluster_module : torch.nn.Module
        the cluster module that is used for the encoding (e.g. DEC). Usually contains the predict method.

    Returns
    -------
    predictions_numpy : np.ndarray
        The predictions of the cluster_module for the data set
    """
    device = get_device_from_module(neural_network)
    predictions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        prediction = cluster_module.predict_hard(neural_network.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    predictions_numpy = torch.cat(predictions, dim=0).numpy()
    return predictions_numpy


# def add_noise(batch):
#     mask = torch.empty(
#         batch.shape, device=batch.device).bernoulli_(0.8)
#     return batch * mask


def int_to_one_hot(int_tensor: torch.Tensor, n_integers: int) -> torch.Tensor:
    """
    Convert a tensor containing integers (e.g. labels) to an one hot encoding.
    Here, each integer gets its own features in the resulting tensor, where only the values 0 or 1 are accepted.
    E.g. [0,0,1,2,1] gets
    [[1,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,1,0]]

    Parameters
    ----------
    int_tensor : torch.Tensor
        The original tensor containing integers
    n_integers : int
        The number of different integers within int_tensor

    Returns
    -------
    onehot : torch.Tensor
        The final one hot encoding tensor
    """
    onehot = torch.zeros([int_tensor.shape[0], n_integers], dtype=torch.float, device=int_tensor.device)
    onehot.scatter_(1, int_tensor.unsqueeze(1).long(), 1)
    return onehot


def embedded_kmeans_prediction(X_embed: np.ndarray, cluster_centers: np.ndarray) -> np.ndarray:
    """
    Predicts the labels of the given embedded data.
    Labels correspond to the id of the closest cluster center.

    Parameters
    ----------
    X_embed : np.ndarray
        dataloader to be used
    cluster_centers : np.ndarray
        input cluster centers

    Returns
    -------
    predicted_labels : np.ndarray
        The predicted labels
    """
    predicted_labels, _ = pairwise_distances_argmin_min(X=X_embed, Y=cluster_centers, metric='euclidean',
                                                        metric_kwargs={'squared': True})
    predicted_labels = predicted_labels.astype(np.int32)
    return predicted_labels


def run_initial_clustering(X: np.ndarray, n_clusters: int, clustering_class: ClusterMixin, clustering_params: dict,
                           random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray, ClusterMixin):
    """
    Get an initial clustering result for a deep clustering algorithm.
    This result can then be refined by the optimization of the neural network.

    Parameters
    ----------
    X : np.ndarray
        the embedded data set
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    clustering_class : ClusterMixin
        the class of the initial clustering algorithm.
        If it is None, random labels will be chosen
    clustering_params : dict
        the parameters for the initial clustering algorithm
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray, ClusterMixin)
        The number of clusters (can change if e.g. DBSCAN is used),
        The initial cluster labels,
        The initial cluster centers,
        The clustering object
    """
    if clustering_class is None:
        clustering_algo = ClusterMixin()
        clustering_algo.labels_ = np.random.randint(n_clusters, size=X.shape[0])
    else:
        # Get possible input parameters of the clustering algorithm
        clustering_class_parameters = inspect.getfullargspec(clustering_class).args + inspect.getfullargspec(
            clustering_class).kwonlyargs
        # Check if n_clusters or n_components is contained in the possible parameters
        if "n_clusters" in clustering_class_parameters:
            if "random_state" in clustering_class_parameters and "random_state" not in clustering_params.keys():
                clustering_algo = clustering_class(n_clusters=n_clusters, random_state=random_state, **clustering_params)
            else:
                clustering_algo = clustering_class(n_clusters=n_clusters, **clustering_params)
        elif "n_components" in clustering_class_parameters:  # in case of GMM
            if "random_state" in clustering_class_parameters and "random_state" not in clustering_params.keys():
                clustering_algo = clustering_class(n_components=n_clusters, random_state=random_state, **clustering_params)
            else:
                clustering_algo = clustering_class(n_components=n_clusters, **clustering_params)
        else:  # in case of e.g., DBSCAN
            if "random_state" in clustering_class_parameters and "random_state" not in clustering_params.keys():
                clustering_algo = clustering_class(random_state=random_state, **clustering_params)
            else:
                clustering_algo = clustering_class(**clustering_params)
        # Run algorithm
        clustering_algo.fit(X)
    # Check if clustering algorithm return cluster centers
    if hasattr(clustering_algo, "cluster_centers_"):
        labels = clustering_algo.labels_
        centers = clustering_algo.cluster_centers_
    elif hasattr(clustering_algo, "means_"):  # in case of GMM
        labels = clustering_algo.predict(X)
        centers = clustering_algo.means_
    else:  # in case of e.g., DBSCAN
        labels = clustering_algo.labels_
        centers = np.array([np.mean(X[labels == i], axis=0) for i in np.unique(labels) if i >= 0])
    n_clusters = np.sum(np.unique(labels) >= 0)  # Needed for DBSCAN, XMeans, GMeans, ...
    return n_clusters, labels, centers, clustering_algo

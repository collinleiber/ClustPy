from clustpy.deep.autoencoders import FeedforwardAutoencoder
import torch
import copy
import numpy as np
from sklearn.base import ClusterMixin
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import run_initial_clustering, detect_device, encode_batchwise


def _get_default_layers(input_dim: int, embedding_size: int) -> list:
    """
    Get the default layers for a feedforward autoencoder.
    Default layers are [input_dim, 500, 500, 2000, embedding_size]

    Parameters
    ----------
    input_dim : int
        size of the first layer
    embedding_size : int
        size of the last layer

    Returns
    -------
    layers : list
        list containing the layers
    """
    layers = [input_dim, 500, 500, 2000, embedding_size]
    return layers


def get_trained_network(trainloader: torch.utils.data.DataLoader = None, data: np.ndarray = None,
                        n_epochs: int = 100, batch_size: int = 128, optimizer_params: dict = None,
                        optimizer_class: torch.optim.Optimizer = torch.optim.Adam, device=None,
                        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), embedding_size: int = 10,
                        neural_network: torch.nn.Module = None,
                        neural_network_class: torch.nn.Module = FeedforwardAutoencoder) -> torch.nn.Module:
    """This function returns a trained neural network. The following cases are considered
       - If the neural network is initialized and trained (neural_network.fitted==True), then return input neural network without training it again.
       - If the neural network is initialized and not trained (neural_network.fitted==False), it will be fitted (neural_network.fitted will be set to True) using default parameters.
       - If the neural network is None, a new neural network is created using neural_network_class, and it will be fitted as described above.
       Beware the input neural_network_class or neural_network object needs both a fit() function and the fitted attribute. See clustpy.deep.feedforward_autoencoder.FeedforwardAutoencoder for an example.

    Parameters
    ----------
    trainloader : torch.utils.data.DataLoader
        dataloader used to train neural_network (default: None)
    data : np.ndarray
        train data set. If data is passed then trainloader can remain empty (default: None)
    n_epochs : int
        number of training epochs (default: 100)
    batch_size : int
        size of the data batches (default: 128)
    optimizer_params : dict
        parameters of the optimizer for the neural network training, includes the learning rate (default: {"lr": 1e-3})
    optimizer_class : torch.optim.Optimizer
        optimizer for training (default: torch.optim.Adam)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.MSELoss())
    embedding_size : int
        dimension of the innermost layer of the neural network (default: 10)
    neural_network : torch.nn.Module
        neural network object to be trained (optional) (default: None)
    neural_network_class : torch.nn.Module
        The neural network class that should be used (default: FeedforwardAutoencoder)
    
    Returns
    -------
    neural_network : torch.nn.Module
        The fitted neural network
    """
    device = detect_device(device)
    optimizer_params = {"lr": 1e-3} if optimizer_params is None else optimizer_params
    if trainloader is None:
        if data is None:
            raise ValueError("data must be specified if trainloader is None")
        trainloader = get_dataloader(data, batch_size, True)
    if neural_network is None:
        input_dim = torch.numel(next(iter(trainloader))[1][0])  # Get input dimensions from first batch
        if embedding_size > input_dim:
            print(
                "WARNING: embedding_size is larger than the dimensionality of the input dataset. embedding_size: {0} / input dimensionality: {1}".format(
                    embedding_size, input_dim))
        # Init neural network parameters
        layers = _get_default_layers(input_dim, embedding_size)
        neural_network = neural_network_class(layers=layers)
    assert hasattr(neural_network,
                   "fitted"), "Neural network has no attribute 'fitted' and is therefore not compatible. Check documentation of fitted, e.g., at clustpy.deep.autoencoders._abstract_autoencoder._AbstractAutoencoder"
    # Save neural network to device
    neural_network.to(device)
    if not neural_network.fitted:
        print("Neural network is not fitted yet, will be pretrained.")
        # Pretrain neural network
        neural_network.fit(n_epochs=n_epochs, optimizer_params=optimizer_params, dataloader=trainloader,
                        optimizer_class=optimizer_class, loss_fn=loss_fn)
    if neural_network.reusable:
        # If neural network is used by multiple deep clustering algorithms, create a deep copy of the object
        neural_network = copy.deepcopy(neural_network)
    return neural_network


def get_default_deep_clustering_initialization(X: np.ndarray | torch.Tensor, n_clusters: int, batch_size: int,
                                               pretrain_optimizer_params: dict, pretrain_epochs: int,
                                               optimizer_class: torch.optim.Optimizer,
                                               loss_fn: torch.nn.modules.loss._Loss,
                                               neural_network: torch.nn.Module, embedding_size: int,
                                               custom_dataloaders: tuple, initial_clustering_class: ClusterMixin,
                                               initial_clustering_params: dict, device: torch.device,
                                               random_state: np.random.RandomState,
                                               neural_network_class: torch.nn.Module = FeedforwardAutoencoder) -> (
        torch.device, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.nn.Module, np.ndarray, int,
        np.ndarray, np.ndarray, ClusterMixin):
    """
    Get the initial setting for most deep clustering algorithms by pretraining a neural network and obtaining an initial clustering result.
    This function further returns the device, where the optimization should take place (e.g., CPU or GPU), and the dataloaders.

    Parameters
    ----------
    X : np.ndarray | torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network
    optimizer_class : torch.optim.Optimizer
        the optimizer
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    neural_network : torch.nn.Module
        the input neural network. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the neural network
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining.
        If it is None, random labels will be chosen
    initial_clustering_params : dict
        parameters for the initial clustering class
    device : torch.device
        The device on which to perform the computations
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    neural_network_class : torch.nn.Module
        The neural network class that should be used (default: FeedforwardAutoencoder)

    Returns
    -------
    tuple : (torch.device, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.nn.Module, np.ndarray, int, np.ndarray, np.ndarray, ClusterMixin)
        The device,
        The trainloader,
        The testloader,
        The pretrained neural network,
        The embedded data,
        The number of clusters (can change if e.g. DBSCAN is used),
        The initial cluster labels,
        The initial cluster centers,
        The clustering object
    """
    device = detect_device(device)
    if custom_dataloaders is None:
        trainloader = get_dataloader(X, batch_size, True, False)
        testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
    neural_network = get_trained_network(trainloader, n_epochs=pretrain_epochs,
                                      optimizer_params=pretrain_optimizer_params, optimizer_class=optimizer_class,
                                      device=device, loss_fn=loss_fn, embedding_size=embedding_size,
                                      neural_network=neural_network, neural_network_class=neural_network_class)
    # Execute initial clustering in embedded space
    embedded_data = encode_batchwise(testloader, neural_network)
    n_clusters, init_labels, init_centers, init_cluster_obj = run_initial_clustering(embedded_data, n_clusters,
                                                                                     initial_clustering_class,
                                                                                     initial_clustering_params,
                                                                                     random_state)
    return device, trainloader, testloader, neural_network, embedded_data, n_clusters, init_labels, init_centers, init_cluster_obj

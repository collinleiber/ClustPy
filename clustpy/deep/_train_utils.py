from clustpy.deep.neural_networks import FeedforwardAutoencoder
import torch
import copy
import numpy as np
from sklearn.base import ClusterMixin
from clustpy.deep._data_utils import get_dataloader, get_train_and_test_dataloader, get_data_dim_from_dataloader
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


def _get_neural_network(input_dim: int, embedding_size: int = 10, neural_network: torch.nn.Module | tuple = None,
                        neural_network_class: torch.nn.Module = FeedforwardAutoencoder,
                        neural_network_params: dict = None, neural_network_weights: str = None,
                        random_state: np.random.RandomState | int = None) -> torch.nn.Module:
    """This function returns a new neural_network.
    - If neural_network is already a torch.nn.module, nothing will happen.
    - If neural_network is None, a new neural_network will be created using the neural_network_class and the parameters from neural_network_params.
    Optionally, the weights contained in the state_dict file referenced by neural_network_weights will be loaded.

    Parameters
    ----------
    input_dim : int
        The input number of features
    embedding_size : int
        dimension of the innermost layer of the neural network (default: 10)
    neural_network : torch.nn.Module | tuple
        the neural network used for the computations.
        Can also be None. In this case a new neural network will be created using neural_network_class and neural_network_params (default: None)
    neural_network_class : torch.nn.Module
        The neural network class that should be used (default: FeedforwardAutoencoder)
    neural_network_params : dict
        Parameters to be used when creating a new neural network using the neural_network_class (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Returns
    -------
    neural_network : torch.nn.Module
        The created neural network
    """
    if neural_network is None:
        if embedding_size > input_dim:
            print(
                "WARNING: embedding_size is larger than the dimensionality of the input dataset. embedding_size: {0} / input dimensionality: {1}".format(
                    embedding_size, input_dim))
        # Init neural network parameters
        if neural_network_params is None:
            neural_network_params = dict()
        if "layers" not in neural_network_params.keys():
            layers = _get_default_layers(input_dim, embedding_size)
            neural_network_params["layers"] = layers
        if "random_state" not in neural_network_params.keys():
            neural_network_params["random_state"] = random_state
        if neural_network_params["layers"][-1] != embedding_size:
            print(
                "WARNING: embedding_size ({0}) in _get_neural_network does not correspond to the layers used to create the neural network. In the following an embedding size of {1} as specified in the layers will be used".format(
                    embedding_size, neural_network_params["layers"][-1]))
        neural_network = neural_network_class(**neural_network_params)
    assert hasattr(neural_network,
                   "fitted"), "Neural network has no attribute 'fitted' and is therefore not compatible. Check documentation of fitted, e.g., at clustpy.deep.neural_networks._abstract_autoencoder._AbstractAutoencoder"
    if neural_network_weights is not None:
        neural_network.load_parameters(neural_network_weights)
    return neural_network


def get_trained_network(trainloader: torch.utils.data.DataLoader = None, data: np.ndarray = None,
                        n_epochs: int = 100, batch_size: int = 128, optimizer_params: dict = None,
                        optimizer_class: torch.optim.Optimizer = torch.optim.Adam, device=None,
                        ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), embedding_size: int = 10,
                        neural_network: torch.nn.Module | tuple = None,
                        neural_network_class: torch.nn.Module = FeedforwardAutoencoder,
                        neural_network_params: dict = None, neural_network_weights: str = None,
                        random_state: np.random.RandomState | int = None) -> torch.nn.Module:
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
    ssl_loss_fn : torch.nn.modules.loss._Loss
        self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    embedding_size : int
        dimension of the innermost layer of the neural network (default: 10)
    neural_network : torch.nn.Module | tuple
        neural network object to be trained (optional)
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_class : torch.nn.Module
        The neural network class that should be used (default: FeedforwardAutoencoder)
    neural_network_params : dict
        Parameters to be used when creating a new neural network using the neural_network_class (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    
    Returns
    -------
    neural_network : torch.nn.Module
        The fitted neural network
    """
    if trainloader is None:
        if data is None:
            raise ValueError("data must be specified if trainloader is None")
        trainloader = get_dataloader(data, batch_size, True)
    # Get neural network object
    input_dim = get_data_dim_from_dataloader(trainloader)
    if neural_network is not None and type(neural_network) is tuple:
        assert len(
            neural_network) == 2, "If neural_network is a tuple, it has to contain two entries: the neural network class (torch.nn.Module) and the initialization parameters (dict)"
        neural_network_class = neural_network[0]
        neural_network_params = neural_network[1]
        neural_network = None
    neural_network = _get_neural_network(input_dim, embedding_size, neural_network, neural_network_class,
                                         neural_network_params, neural_network_weights, random_state)
    # Move neural network to device
    device = detect_device(device)
    neural_network.to(device)
    if not neural_network.fitted:
        print("Neural network is not fitted yet, will be pretrained.")
        # Pretrain neural network
        optimizer_params = {"lr": 1e-3} if optimizer_params is None else optimizer_params
        neural_network.fit(n_epochs=n_epochs, optimizer_params=optimizer_params, dataloader=trainloader,
                           optimizer_class=optimizer_class, ssl_loss_fn=ssl_loss_fn)
    if neural_network.work_on_copy:
        # If neural network is used by multiple deep clustering algorithms, create a deep copy of the object
        neural_network = copy.deepcopy(neural_network)
    return neural_network


def get_default_deep_clustering_initialization(X: np.ndarray | torch.Tensor, n_clusters: int, batch_size: int,
                                               pretrain_optimizer_params: dict, pretrain_epochs: int,
                                               optimizer_class: torch.optim.Optimizer,
                                               ssl_loss_fn: torch.nn.modules.loss._Loss,
                                               neural_network: torch.nn.Module | tuple, embedding_size: int,
                                               custom_dataloaders: tuple, initial_clustering_class: ClusterMixin,
                                               initial_clustering_params: dict, device: torch.device,
                                               random_state: np.random.RandomState,
                                               neural_network_class: torch.nn.Module = FeedforwardAutoencoder,
                                               neural_network_params: dict = None,
                                               neural_network_weights: str = None) -> (
        torch.device, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, torch.nn.Module, np.ndarray, int,
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
    ssl_loss_fn : torch.nn.modules.loss._Loss
        self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict)
    embedding_size : int
        size of the embedding within the neural network
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
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
    neural_network_params : dict
        Parameters to be used when creating a new neural network using the neural_network_class (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)

    Returns
    -------
    tuple : (torch.device, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, torch.nn.Module, np.ndarray, int, np.ndarray, np.ndarray, ClusterMixin)
        The device,
        The trainloader,
        The testloader,
        The batch size (can be different from input if another value is used within custom_dataloader),
        The pretrained neural network,
        The embedded data,
        The number of clusters (can change if e.g. DBSCAN is used),
        The initial cluster labels,
        The initial cluster centers,
        The clustering object
    """
    device = detect_device(device)
    trainloader, testloader, batch_size = get_train_and_test_dataloader(X, batch_size, custom_dataloaders)
    neural_network = get_trained_network(trainloader, n_epochs=pretrain_epochs,
                                         optimizer_params=pretrain_optimizer_params, optimizer_class=optimizer_class,
                                         device=device, ssl_loss_fn=ssl_loss_fn, embedding_size=embedding_size,
                                         neural_network=neural_network, neural_network_class=neural_network_class,
                                         neural_network_params=neural_network_params,
                                         neural_network_weights=neural_network_weights,
                                         random_state=random_state)
    # Execute initial clustering in embedded space
    embedded_data = encode_batchwise(testloader, neural_network)
    n_clusters, init_labels, init_centers, init_cluster_obj = run_initial_clustering(embedded_data, n_clusters,
                                                                                     initial_clustering_class,
                                                                                     initial_clustering_params,
                                                                                     random_state)
    return device, trainloader, testloader, batch_size, neural_network, embedded_data, n_clusters, init_labels, init_centers, init_cluster_obj

from clustpy.deep.autoencoders import FeedforwardAutoencoder
import torch
import copy
import numpy as np
from sklearn.base import ClusterMixin
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import run_initial_clustering, detect_device, encode_batchwise


def _get_default_layers(input_dim: int, embedding_size: int) -> list:
    """
    Get the default layers for an autoencoder.
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


def get_trained_autoencoder(trainloader: torch.utils.data.DataLoader, optimizer_params: dict, n_epochs: int, device,
                            optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
                            embedding_size: int, autoencoder: torch.nn.Module = None,
                            autoencoder_class: torch.nn.Module = FeedforwardAutoencoder) -> torch.nn.Module:
    """This function returns a trained autoencoder. The following cases are considered
       - If the autoencoder is initialized and trained (autoencoder.fitted==True), then return input autoencoder without training it again.
       - If the autoencoder is initialized and not trained (autoencoder.fitted==False), it will be fitted (autoencoder.fitted will be set to True) using default parameters.
       - If the autoencoder is None, a new autoencoder is created using autoencoder_class, and it will be fitted as described above.
       Beware the input autoencoder_class or autoencoder object needs both a fit() function and the fitted attribute. See clustpy.deep.feedforward_autoencoder.FeedforwardAutoencoder for an example.

    Parameters
    ----------
    trainloader : torch.utils.data.DataLoader
        dataloader used to train autoencoder
    optimizer_params : dict
        parameters of the optimizer for the autoencoder training, includes the learning rate
    n_epochs : int
        number of training epochs
    device : torch.device
        device to be trained on
    optimizer_class : torch.optim.Optimizer
        optimizer for training.
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction.
    embedding_size : int
        dimension of the innermost layer of the autoencoder
    autoencoder : torch.nn.Module
        autoencoder object to be trained (optional) (default: None)
    autoencoder_class : torch.nn.Module
        The autoencoder class that should be used (default: FeedforwardAutoencoder)
    
    Returns
    -------
    autoencoder : torch.nn.Module
        The fitted autoencoder
    """
    if autoencoder is None:
        input_dim = torch.numel(next(iter(trainloader))[1][0])  # Get input dimensions from first batch
        if embedding_size > input_dim:
            print(
                "WARNING: embedding_size is larger than the dimensionality of the input dataset. embedding_size: {0} / input dimensionality: {1}".format(
                    embedding_size, input_dim))
        # Init Autoencoder parameters
        layers = _get_default_layers(input_dim, embedding_size)
        autoencoder = autoencoder_class(layers=layers)
    assert hasattr(autoencoder,
                   "fitted"), "Autoencoder has no attribute 'fitted' and is therefore not compatible. Check documentation of fitted clustpy.deep.autoencoders._abstract_autoencoder._AbstractAutoencoder"
    # Save autoencoder to device
    autoencoder.to(device)
    if not autoencoder.fitted:
        print("Autoencoder is not fitted yet, will be pretrained.")
        # Pretrain Autoencoder
        autoencoder.fit(n_epochs=n_epochs, optimizer_params=optimizer_params, dataloader=trainloader,
                        device=device, optimizer_class=optimizer_class, loss_fn=loss_fn)
    if autoencoder.reusable:
        # If autoencoder is used by multiple deep clustering algorithms, create a deep copy of the object
        autoencoder = copy.deepcopy(autoencoder)

    return autoencoder


def get_standard_initial_deep_clustering_setting(X: np.ndarray, n_clusters: int, batch_size: int,
                                                 pretrain_optimizer_params: dict, pretrain_epochs: int,
                                                 optimizer_class: torch.optim.Optimizer,
                                                 loss_fn: torch.nn.modules.loss._Loss,
                                                 autoencoder: torch.nn.Module, embedding_size: int,
                                                 custom_dataloaders: tuple, initial_clustering_class: ClusterMixin,
                                                 initial_clustering_params: dict,
                                                 random_state: np.random.RandomState,
                                                 autoencoder_class: torch.nn.Module = FeedforwardAutoencoder) -> (
        torch.device, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.nn.Module, np.ndarray, int,
        np.ndarray, np.ndarray, ClusterMixin):
    """
    Get the initial setting for most deep clustering algorithms by pretraining an autoencoder and obtraining an initial clustering result.
    This function further returns the device, where the optimization should take place (e.g., CPU or GPU), and the dataloaders.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    optimizer_class : torch.optim.Optimizer
        the optimizer
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    autoencoder_class : torch.nn.Module
        The autoencoder class that should be used (default: FeedforwardAutoencoder)

    Returns
    -------
    tuple : (torch.device, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.nn.Module, np.ndarray, int, np.ndarray, np.ndarray, ClusterMixin)
        The device,
        The trainloader,
        The testloader,
        The pretrained autoencoder,
        The embedded data,
        The number of clusters (can change if e.g. DBSCAN is used),
        The initial cluster labels,
        The initial cluster centers,
        The clustering object
    """
    device = detect_device()
    if custom_dataloaders is None:
        trainloader = get_dataloader(X, batch_size, True, False)
        testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
    autoencoder = get_trained_autoencoder(trainloader, pretrain_optimizer_params, pretrain_epochs, device,
                                          optimizer_class, loss_fn, embedding_size, autoencoder, autoencoder_class)
    # Execute initial clustering in embedded space
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    n_clusters, init_labels, init_centers, init_cluster_obj = run_initial_clustering(embedded_data, n_clusters,
                                                                                     initial_clustering_class,
                                                                                     initial_clustering_params,
                                                                                     random_state)
    return device, trainloader, testloader, autoencoder, embedded_data, n_clusters, init_labels, init_centers, init_cluster_obj

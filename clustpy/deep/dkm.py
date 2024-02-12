"""
@authors:
Collin Leiber
"""

from clustpy.deep._utils import encode_batchwise, squared_euclidean_distance, predict_batchwise, \
    set_torch_seed, embedded_kmeans_prediction
from clustpy.deep._data_utils import get_dataloader, augmentation_invariance_check
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _dkm(X: np.ndarray, n_clusters: int, alphas: list, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss, autoencoder: torch.nn.Module,
         embedding_size: int, cluster_loss_weight: float, custom_dataloaders: tuple, augmentation_invariance: bool,
         initial_clustering_class: ClusterMixin, initial_clustering_params: dict,
         random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DKM clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    alphas : list
        Small values close to 0 are equivalent to homogeneous assignments to all clusters. Large values simulate a clear assignment as with kMeans.
        list of different alpha values used for the prediction
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    clustering_epochs : int
        number of epochs for each alpha value for the actual clustering procedure.
        The total number of clustering epochs therefore corresponds to: len(alphas)*clustering_epochs
    optimizer_class : torch.optim.Optimizer
        the optimizer
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DKM after the training terminated,
        The cluster centers as identified by DKM after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, autoencoder, _, n_clusters, _, init_centers, _ = get_standard_initial_deep_clustering_setting(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, loss_fn, autoencoder,
        embedding_size, custom_dataloaders, initial_clustering_class, initial_clustering_params, random_state)
    # Setup DKM Module
    dkm_module = _DKM_Module(init_centers, alphas, augmentation_invariance).to(device)
    # Use DKM optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(dkm_module.parameters()),
                                **clustering_optimizer_params)
    # DKM Training loop
    dkm_module.fit(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn, cluster_loss_weight)
    # Get labels
    dkm_labels = predict_batchwise(testloader, autoencoder, dkm_module, device)
    dkm_centers = dkm_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dkm_labels, dkm_centers, autoencoder


def _get_default_alphas(init_alpha: float = 0.1, n_alphas: int = 40) -> list:
    """
    Return a list with alphas equal to \alpha_{i+1}=2^{1/log(i)^2}*\alpha_i, where \alpha_1=0.1 and maximum i=40

    Parameters
    ----------
    init_alpha : float
        the initial alpha value (default: 0.1)
    n_alphas : int
        number of alphas to obtain (default: 40)

    Returns
    -------
    all_alphas : list
        List containing the different alpha values
    """
    # Values taken from the publication
    last_alpha = init_alpha
    all_alphas = [last_alpha]
    # Start loop to get all alpha values
    for i in range(2, n_alphas + 1):
        new_alpha = 2 ** (1 / np.log(i) ** 2) * last_alpha
        all_alphas.append(new_alpha)
        last_alpha = new_alpha
    return all_alphas


def _dkm_get_probs(squared_diffs: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Predict soft cluster labels given embedded samples.

    Parameters
    ----------
    squared_diffs : torch.Tensor
        the squared distances between points and centers
    alpha : float
        the alpha value

    Returns
    -------
    prob : torch.Tensor
        The predicted soft labels
    """
    # Shift distances for exponent with min value to avoid underflow (see original implementaion: https://github.com/MaziarMF/deep-k-means/blob/master/compgraph.py)
    shifted_squared_diffs = squared_diffs - squared_diffs.min(1)[0].reshape((-1, 1))
    exponent = torch.exp(-alpha * (shifted_squared_diffs))
    param_softmax = exponent / exponent.sum(1).reshape((-1, 1))
    return param_softmax


class _DKM_Module(torch.nn.Module):
    """
    The _DKM_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers
    alphas : list
        list of different alpha values used for the prediction
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    alphas : list
        list of different alpha values used for the prediction
    centers : torch.Tensor:
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(self, init_centers: np.ndarray, alphas: list, augmentation_invariance: bool = False):
        super().__init__()
        self.alphas = alphas
        self.augmentation_invariance = augmentation_invariance
        # Centers are learnable parameters
        self.centers = torch.nn.Parameter(torch.tensor(init_centers), requires_grad=True)

    def predict(self, embedded: torch.Tensor, alpha: float = 1000) -> torch.Tensor:
        """
        Soft prediction of given embedded samples. Returns the corresponding soft labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        alpha : float
            the alpha value (default: 1000)

        Returns
        -------
        pred : torch.Tensor
            The predicted soft labels
        """
        squared_diffs = squared_euclidean_distance(embedded, self.centers)
        pred = _dkm_get_probs(squared_diffs, alpha)
        return pred

    def predict_hard(self, embedded: torch.Tensor, alpha: float = 1000) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the soft prediction method and then applies argmax.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        alpha : float
            the alpha value (default: 1000)

        Returns
        -------
        pred_hard : torch.Tensor
            The predicted hard labels
        """
        pred_hard = self.predict(embedded, alpha).argmax(1)
        return pred_hard

    def dkm_loss(self, embedded: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Calculate the DKM loss of given embedded samples and given alpha.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        alpha : float
            the alpha value

        Returns
        -------
        loss : torch.Tensor
            the final DKM loss
        """
        squared_diffs = squared_euclidean_distance(embedded, self.centers)
        probs = _dkm_get_probs(squared_diffs, alpha)
        loss = (squared_diffs.sqrt() * probs).sum(1).mean()
        return loss

    def dkm_augmentation_invariance_loss(self, embedded: torch.Tensor, embedded_aug: torch.Tensor,
                                         alpha: float) -> torch.Tensor:
        """
        Calculate the DKM loss of given embedded samples with augmentation invariance and given alpha.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor
            the embedded augmented samples
        alpha : float
            the alpha value

        Returns
        -------
        loss : torch.Tensor
            the final DKM loss
        """
        # Get loss of non-augmented data
        squared_diffs = squared_euclidean_distance(embedded, self.centers)
        probs = _dkm_get_probs(squared_diffs, alpha)
        clean_loss = (squared_diffs.sqrt() * probs).sum(1).mean()
        # Get loss of augmented data
        squared_diffs_augmented = squared_euclidean_distance(embedded_aug, self.centers)
        aug_loss = (squared_diffs_augmented.sqrt() * probs).sum(1).mean()
        # average losses
        loss = (clean_loss + aug_loss) / 2
        return loss

    def _loss(self, batch: list, alpha: float, autoencoder: torch.nn.Module, cluster_loss_weight: float,
              loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete DKM + Autoencoder loss.

        Parameters
        ----------
        batch : list
            the minibatch
        alpha : float
            the alpha value
        autoencoder : torch.nn.Module
            the autoencoder
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DKM loss
        """
        # Calculate combined total loss
        if self.augmentation_invariance:
            # Calculate reconstruction loss
            ae_loss, embedded, _ = autoencoder.loss([batch[0], batch[2]], loss_fn, device)
            ae_loss_aug, embedded_aug, _ = autoencoder.loss([batch[0], batch[1]], loss_fn, device)
            ae_loss = (ae_loss + ae_loss_aug) / 2
            # Calculate clustering loss
            cluster_loss = self.dkm_augmentation_invariance_loss(embedded, embedded_aug, alpha)
        else:
            # Calculate reconstruction loss
            ae_loss, embedded, _ = autoencoder.loss(batch, loss_fn, device)
            # Calculate clustering loss
            cluster_loss = self.dkm_loss(embedded, alpha)
        loss = ae_loss + cluster_loss * cluster_loss_weight
        return loss

    def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
            cluster_loss_weight: float) -> '_DKM_Module':
        """
        Trains the _DKM_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            the autoencoder
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        n_epochs : int
            number of epochs for the clustering procedure.
            The total number of epochs therefore corresponds to: len(alphas)*n_epochs
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss

        Returns
        -------
        self : _DKM_Module
            this instance of the _DKM_Module
        """
        for alpha in self.alphas:
            for e in range(n_epochs):
                for batch in trainloader:
                    loss = self._loss(batch, alpha, autoencoder, cluster_loss_weight, loss_fn, device)
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        return self


class DKM(BaseEstimator, ClusterMixin):
    """
    The Deep k-Means (DKM) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DKM loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    alphas : tuple
        tuple of different alpha values used for the prediction.
        Small values close to 0 are equivalent to homogeneous assignments to all clusters. Large values simulate a clear assignment as with kMeans.
        If None, the default calculation of the paper will be used.
        This is equal to \alpha_{i+1}=2^{1/log(i)^2}*\alpha_i with \alpha_1=0.1 and maximum i=40.
        Alpha can also be a tuple with (None, \alpha_1, maximum i) (default: (1000))
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 50)
    clustering_epochs : int
        number of epochs for each alpha value for the actual clustering procedure.
        The total number of clustering epochs therefore corresponds to: len(alphas)*clustering_epochs (default: 100)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 1)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dkm_labels_ : np.ndarray
        The final DKM labels
    dkm_cluster_centers_ : np.ndarray
        The final DKM cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DKM
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> dkm = DKM(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> dkm.fit(data)

    References
    ----------
    Fard, Maziar Moradi, Thibaut Thonet, and Eric Gaussier. "Deep k-means: Jointly clustering with k-means and learning representations."
    Pattern Recognition Letters 138 (2020): 185-192.
    """

    def __init__(self, n_clusters: int, alphas: tuple = (1000), batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 50, clustering_epochs: int = 100,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, cluster_loss_weight: float = 1, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        if alphas is None:
            alphas = _get_default_alphas()
        elif (type(alphas) is tuple or type(alphas) is list) and len(alphas) == 3 and alphas[0] is None:
            alphas = _get_default_alphas(init_alpha=alphas[1], n_alphas=alphas[2])
        elif type(alphas) is int or type(alphas) is float:
            alphas = [alphas]
        assert type(alphas) is tuple or type(alphas) is list, "alphas must be a list, int or tuple"
        self.alphas = alphas
        self.batch_size = batch_size
        self.pretrain_optimizer_params = {"lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {"lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.cluster_loss_weight = cluster_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = {} if initial_clustering_params is None else initial_clustering_params
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DKM':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DKM
            this instance of the DKM algorithm
        """
        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        kmeans_labels, kmeans_centers, dkm_labels, dkm_centers, autoencoder = _dkm(X, self.n_clusters, self.alphas,
                                                                                   self.batch_size,
                                                                                   self.pretrain_optimizer_params,
                                                                                   self.clustering_optimizer_params,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.cluster_loss_weight,
                                                                                   self.custom_dataloaders,
                                                                                   self.augmentation_invariance,
                                                                                   self.initial_clustering_class,
                                                                                   self.initial_clustering_params,
                                                                                   self.random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dkm_labels_ = dkm_labels
        self.dkm_cluster_centers_ = dkm_centers
        self.autoencoder = autoencoder
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        dataloader = get_dataloader(X, self.batch_size, False, False)
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, self.autoencoder)
        return predicted_labels

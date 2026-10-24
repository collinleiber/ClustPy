"""SpectralNet: minibatch spectral clustering with an out-of-sample neural map."""

from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._encoding_utils import encode_batchwise
from clustpy.deep._train_utils import get_trained_network
from clustpy.deep._utils import detect_device
from clustpy.deep.neural_networks._abstract_neural_network import _AbstractNeuralNetwork
from clustpy.utils.checks import check_parameters


def _spectralnet_affinity(features: torch.Tensor, n_neighbors: int, scale: float | None = None) -> torch.Tensor:
    """Build the symmetric, Gaussian nearest-neighbor affinity of one minibatch."""
    squared_distances = torch.cdist(features, features).square()
    k = min(n_neighbors, features.shape[0] - 1)
    without_self = squared_distances.masked_fill(
        torch.eye(features.shape[0], dtype=torch.bool, device=features.device), float("inf"))
    neighbor_distances, neighbor_indices = without_self.topk(k, largest=False)
    squared_scale = neighbor_distances[:, -1].median() if scale is None else features.new_tensor(scale ** 2)
    if not torch.isfinite(squared_scale) or squared_scale <= 0:
        raise ValueError("SpectralNet needs a positive, finite nearest-neighbor distance.")
    mask = torch.zeros_like(squared_distances, dtype=torch.bool)
    mask.scatter_(1, neighbor_indices, True)
    weights = torch.exp(-squared_distances / (2 * squared_scale)) * mask
    weights.fill_diagonal_(0)
    return (weights + weights.T) / 2


def _spectralnet_orthogonalizer(raw_outputs: torch.Tensor) -> torch.Tensor:
    """Return the frozen right multiplier enforcing Y.T @ Y / batch_size = I."""
    if raw_outputs.shape[0] < raw_outputs.shape[1]:
        raise ValueError("The orthogonalization batch must contain at least n_clusters samples.")
    _, triangular = torch.linalg.qr(raw_outputs, mode="reduced")
    if not torch.isfinite(triangular).all():
        raise ValueError("SpectralNet outputs contain non-finite values.")
    singular_values = torch.linalg.svdvals(triangular)
    threshold = torch.finfo(raw_outputs.dtype).eps * max(raw_outputs.shape) * singular_values[0]
    if singular_values[-1] <= threshold:
        raise ValueError("SpectralNet outputs lost column rank; reduce the learning rate or change the network.")
    identity = torch.eye(triangular.shape[0], dtype=raw_outputs.dtype, device=raw_outputs.device)
    return raw_outputs.shape[0] ** 0.5 * torch.linalg.solve_triangular(triangular, identity, upper=True)


def _spectralnet_loss(weights: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Unnormalized graph-Laplacian energy, scaled by minibatch size (paper Appendix D)."""
    return (weights * torch.cdist(outputs, outputs).square()).sum() / outputs.shape[0]


class _SpectralNetEncoder(_AbstractNeuralNetwork):
    """A dense spectral map with a registered, frozen orthogonalization layer."""

    orthogonal_weights: torch.Tensor

    def __init__(self, input_dim: int, n_clusters: int, hidden_layers: tuple[int, ...],
                 feature_network: _AbstractNeuralNetwork | None = None):
        super().__init__()
        self.feature_network = feature_network
        layers: list[torch.nn.Module] = []
        for hidden_dim in hidden_layers:
            layers.extend([torch.nn.Linear(input_dim, hidden_dim), torch.nn.LeakyReLU()])
            input_dim = hidden_dim
        layers.extend([torch.nn.Linear(input_dim, n_clusters), torch.nn.Tanh()])
        self.spectral_encoder = torch.nn.Sequential(*layers)
        self.register_buffer("orthogonal_weights", torch.eye(n_clusters))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_network is not None:
            x = self.feature_network.encode(x)
            if isinstance(x, tuple):
                x = x[0]
        return self.spectral_encoder(x) @ self.orthogonal_weights


def _spectralnet_siamese(features: np.ndarray, hidden_layers: tuple[int, ...], n_neighbors: int,
                         n_epochs: int, batch_size: int, optimizer_params: dict,
                         device: torch.device, random_state: np.random.RandomState) -> tuple[torch.nn.Module, list[float]]:
    """Learn affinities from neighbor/non-neighbor pairs, without cluster labels."""
    n_samples, input_dim = features.shape
    if n_samples < 3:
        raise ValueError("Siamese training needs at least three samples.")
    k = min(n_neighbors, n_samples - 2)
    candidates = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(features, return_distance=False)
    neighbors = np.array([row[row != i][:k] for i, row in enumerate(candidates)])
    layers: list[torch.nn.Module] = []
    for hidden_dim in hidden_layers[:-1]:
        layers.extend([torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()])
        input_dim = hidden_dim
    layers.append(torch.nn.Linear(input_dim, hidden_layers[-1]))
    network = torch.nn.Sequential(*layers).to(device)
    optimizer = torch.optim.RMSprop(network.parameters(), **optimizer_params)
    data = torch.as_tensor(features, dtype=torch.float32)
    losses = []
    steps_per_epoch = max(1, int(np.ceil(n_samples / batch_size)))
    for _ in range(n_epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            anchors = random_state.randint(n_samples, size=batch_size)
            positives = neighbors[anchors, random_state.randint(k, size=batch_size)]
            negatives = random_state.randint(n_samples, size=batch_size)
            invalid = (negatives == anchors) | (negatives[:, None] == neighbors[anchors]).any(1)
            while invalid.any():
                negatives[invalid] = random_state.randint(n_samples, size=int(invalid.sum()))
                invalid = (negatives == anchors) | (negatives[:, None] == neighbors[anchors]).any(1)
            encoded = network(data[anchors].to(device))
            positive_loss = (encoded - network(data[positives].to(device))).square().sum(1)
            negative_distance = torch.linalg.vector_norm(encoded - network(data[negatives].to(device)), dim=1)
            loss = (positive_loss + (1 - negative_distance).clamp_min(0).square()).mean()
            if not torch.isfinite(loss):
                raise ValueError("Siamese training produced a non-finite loss.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        losses.append(total_loss / steps_per_epoch)
    return network.eval(), losses


class SpectralNet(_AbstractDeepClusteringAlgo):
    """Spectral clustering with a learned out-of-sample embedding.

    A Siamese network optionally learns a distance from nearest-neighbor pairs.
    Spectral training alternates independent orthogonalization and gradient
    minibatches. The last linear layer is then frozen and KMeans clusters the
    learned embedding. No training-point affinities are needed at prediction.

    Parameters
    ----------
    n_clusters : int
        Number of clusters and dimensions of the spectral embedding.
    batch_size : int
        Minibatch size, at least n_clusters. Larger batches retain more graph
        structure; affinity storage is quadratic in this value (default: 256).
    n_neighbors : int
        Number of affinity neighbors within each spectral minibatch. Choose this
        for the data density: too few neighbors can fragment the graph (default: 10).
    affinity_scale : float | None
        Gaussian standard deviation in the metric space. If None, use the median
        training-set distance to the n_neighbors-th neighbor. This scale is fixed
        throughout spectral training (default: None).
    clustering_epochs : int
        Number of spectral training epochs (default: 100).
    spectral_hidden_layers : tuple[int, ...]
        Hidden widths of the spectral map (default: (128, 128)).
    clustering_optimizer_params : dict | None
        RMSprop parameters. Defaults to lr=1e-5, alpha=0.9.
    use_siamese : bool
        Learn a distance with a Siamese network (default: True).
    siamese_epochs : int
        Number of Siamese training epochs (default: 100).
    siamese_hidden_layers : tuple[int, ...]
        Siamese hidden widths followed by its embedding width (default: (128, 128, 10)).
    siamese_n_neighbors : int
        Neighbors used as positive Siamese pairs (default: 2).
    siamese_optimizer_params : dict | None
        RMSprop parameters. Defaults to lr=1e-3, alpha=0.9.
    use_autoencoder : bool
        Create a default autoencoder before Siamese/spectral training. An explicitly
        supplied feature network or weights is used regardless of this flag (default: False).
    pretrain_epochs : int
        Feature-network pretraining epochs (default: 100).
    pretrain_optimizer_params : dict | None
        Adam parameters for optional feature pretraining (default: None).
    neural_network : _AbstractNeuralNetwork | tuple | None
        Optional feature network, following the other deep clustering algorithms.
        If supplied, it is used even when use_autoencoder=False (default: None).
    neural_network_weights : str | Path | None
        Optional feature-network state dictionary (default: None).
    embedding_size : int
        Autoencoder code dimension, not the spectral output dimension (default: 10).
    device : torch.device | int | str | None
        Training device; inferred when None (default: None).
    random_state : np.random.RandomState | int | None
        Seed for initialization, independent minibatch sampling and KMeans (default: None).

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster assignments, dtype int32.
    cluster_centers_ : np.ndarray
        KMeans centers in the n_clusters-dimensional spectral embedding.
    neural_network_trained_ : _AbstractNeuralNetwork
        Spectral map, including optional feature preprocessing and frozen output layer.
    spectral_loss_history_ : list[float]
        Mean spectral training loss for each epoch.
    siamese_loss_history_ : list[float]
        Mean contrastive losses, empty when Siamese training is disabled.
    affinity_scale_ : float
        Fitted Gaussian standard deviation in the affinity space.

    Notes
    -----
    Inputs are two-dimensional feature arrays. Image arrays must first be flattened
    or encoded. This implementation uses the unnormalized Laplacian. Training
    quality depends on affinity scale, connectivity and optimization; a finite or
    small loss alone does not establish correct clustering. Rank-deficient output
    layers raise instead of silently returning a collapsed embedding.

    References
    ----------
    Shaham, Uri, et al. "SpectralNet: Spectral Clustering using Deep Neural Networks."
    ICLR 2018. https://arxiv.org/abs/1801.01587
    """

    def __init__(self, n_clusters: int, batch_size: int = 256, n_neighbors: int = 10, affinity_scale: float | None = None,
                 clustering_epochs: int = 100, spectral_hidden_layers: tuple[int, ...] = (128, 128),
                 clustering_optimizer_params: dict | None = None, use_siamese: bool = True,
                 siamese_epochs: int = 100, siamese_hidden_layers: tuple[int, ...] = (128, 128, 10),
                 siamese_n_neighbors: int = 2, siamese_optimizer_params: dict | None = None,
                 use_autoencoder: bool = False, pretrain_epochs: int = 100,
                 pretrain_optimizer_params: dict | None = None,
                 neural_network: _AbstractNeuralNetwork | tuple | None = None,
                 neural_network_weights: str | Path | None = None, embedding_size: int = 10,
                 device: torch.device | int | str | None = None,
                 random_state: np.random.RandomState | int | None = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.affinity_scale = affinity_scale
        self.clustering_epochs = clustering_epochs
        self.spectral_hidden_layers = spectral_hidden_layers
        self.clustering_optimizer_params = clustering_optimizer_params
        self.use_siamese = use_siamese
        self.siamese_epochs = siamese_epochs
        self.siamese_hidden_layers = siamese_hidden_layers
        self.siamese_n_neighbors = siamese_n_neighbors
        self.siamese_optimizer_params = siamese_optimizer_params
        self.use_autoencoder = use_autoencoder
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_optimizer_params = pretrain_optimizer_params

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'SpectralNet':
        """Learn the metric, spectral map and cluster centers without using y."""
        X, _, random_state, pretrain_params, _, _ = self._check_parameters(X)
        if X.ndim != 2:
            raise ValueError("SpectralNet expects a two-dimensional feature array.")
        if not isinstance(self.n_clusters, (int, np.integer)) or not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters must be an integer between 1 and the number of samples.")
        for name in ("batch_size", "n_neighbors", "siamese_n_neighbors", "clustering_epochs", "siamese_epochs", "pretrain_epochs"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError("{0} must be an integer.".format(name))
        if self.batch_size < max(2, self.n_clusters):
            raise ValueError("batch_size must be at least max(2, n_clusters).")
        if self.n_neighbors < 1 or self.siamese_n_neighbors < 1:
            raise ValueError("Neighbor counts must be positive.")
        if self.affinity_scale is not None and (not np.isfinite(self.affinity_scale) or self.affinity_scale <= 0):
            raise ValueError("affinity_scale must be positive and finite.")
        if self.clustering_epochs < 0 or self.siamese_epochs < 0 or self.pretrain_epochs < 0:
            raise ValueError("Epoch counts must be nonnegative.")
        if not self.siamese_hidden_layers or any(width < 1 for width in (*self.spectral_hidden_layers, *self.siamese_hidden_layers)):
            raise ValueError("Network widths must be positive and the Siamese architecture must not be empty.")
        device = detect_device(self.device)
        feature_network = None
        features = X.astype(np.float32)
        if self.use_autoencoder or self.neural_network is not None or self.neural_network_weights is not None:
            feature_network = get_trained_network(data=X, n_epochs=self.pretrain_epochs, batch_size=self.batch_size,
                                                  optimizer_params=pretrain_params, device=device,
                                                  embedding_size=self.embedding_size, neural_network=self.neural_network,
                                                  neural_network_weights=self.neural_network_weights,
                                                  random_state=random_state)
            feature_network.eval()
            with torch.no_grad():
                features = encode_batchwise(get_dataloader(X, self.batch_size, False), feature_network).astype(np.float32)
        model = _SpectralNetEncoder(features.shape[1], self.n_clusters, self.spectral_hidden_layers, feature_network).to(device)
        metric_features = features
        self.siamese_loss_history_: list[float] = []
        if self.use_siamese:
            siamese_params = {"lr": 1e-3, "alpha": 0.9} if self.siamese_optimizer_params is None else self.siamese_optimizer_params
            self.siamese_network_, self.siamese_loss_history_ = _spectralnet_siamese(
                features, self.siamese_hidden_layers, self.siamese_n_neighbors, self.siamese_epochs,
                self.batch_size, siamese_params, device, random_state)
            with torch.no_grad():
                metric_features = np.concatenate([
                    self.siamese_network_(torch.as_tensor(features[i:i + self.batch_size], device=device)).cpu().numpy()
                    for i in range(0, len(features), self.batch_size)])
        elif hasattr(self, "siamese_network_"):
            del self.siamese_network_
        if self.affinity_scale is None:
            neighbor_count = min(self.n_neighbors + 1, len(X))
            neighbor_distances = NearestNeighbors(n_neighbors=neighbor_count).fit(metric_features).kneighbors(metric_features)[0]
            self.affinity_scale_ = float(np.median(neighbor_distances[:, -1]))
        else:
            self.affinity_scale_ = float(self.affinity_scale)
        if not np.isfinite(self.affinity_scale_) or self.affinity_scale_ <= 0:
            raise ValueError("SpectralNet needs positive metric distances or an explicit positive affinity_scale.")
        data = torch.as_tensor(features)
        metric_data = torch.as_tensor(metric_features)
        size = min(self.batch_size, len(X))
        optimizer_params = {"lr": 1e-5, "alpha": 0.9} if self.clustering_optimizer_params is None else self.clustering_optimizer_params
        optimizer = torch.optim.RMSprop(model.spectral_encoder.parameters(), **optimizer_params)
        self.spectral_loss_history_: list[float] = []
        for _ in range(self.clustering_epochs):
            losses = []
            for _ in range(max(1, int(np.ceil(len(X) / size)))):
                orth_indices = random_state.choice(len(X), size, replace=False)
                gradient_indices = random_state.choice(len(X), size, replace=False)
                with torch.no_grad():
                    weights = _spectralnet_affinity(metric_data[gradient_indices].to(device), self.n_neighbors, self.affinity_scale_)
                    raw = model.spectral_encoder(data[orth_indices].to(device))
                    model.orthogonal_weights.copy_(_spectralnet_orthogonalizer(raw))
                outputs = model.spectral_encoder(data[gradient_indices].to(device)) @ model.orthogonal_weights
                loss = _spectralnet_loss(weights, outputs)
                if not torch.isfinite(loss):
                    raise ValueError("Spectral training produced a non-finite loss.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            self.spectral_loss_history_.append(float(np.mean(losses)))
        with torch.no_grad():
            final_indices = random_state.choice(len(X), size, replace=False)
            model.orthogonal_weights.copy_(_spectralnet_orthogonalizer(model.spectral_encoder(data[final_indices].to(device))))
        self.neural_network_trained_ = model.eval()
        self.set_n_features_in(X)
        embedding = self.transform(X)
        kmeans = KMeans(self.n_clusters, n_init=10, random_state=random_state).fit(embedding)
        self.labels_ = kmeans.labels_.astype(np.int32)
        self.cluster_centers_ = kmeans.cluster_centers_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed unseen points in bounded batches, using the frozen learned map."""
        check_is_fitted(self, ["neural_network_trained_", "n_features_in_"])
        X, _, _ = check_parameters(X, allow_size_1=True, estimator_obj=self)
        model = self.neural_network_trained_
        assert isinstance(model, _AbstractNeuralNetwork)
        with torch.no_grad():
            return encode_batchwise(get_dataloader(X, self.batch_size, False), model)

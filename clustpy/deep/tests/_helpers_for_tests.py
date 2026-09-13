import torch
import numpy as np
from clustpy.data import create_subspace_data, create_nr_data
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.deep.neural_networks._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep import get_default_augmented_dataloaders
from clustpy.data import load_optdigits
import copy


def _get_dc_test_data():
    X, labels = create_subspace_data(100, subspace_features=(3, 30), random_state=42)
    return X, labels


def _get_dc_test_nr_data():
    X, labels = create_nr_data(100, subspace_features=(3, 3, 30), random_state=42)
    labels = labels[:, :-1]  # ignore noise space
    return X, labels


def _get_dc_test_neuralnetwork(input_dim: int):
    neural_network = (FeedforwardAutoencoder, {"layers": [input_dim, 10, 4], "random_state": 42})
    return neural_network


def _set_params_for_dc_algorithm(algo: _AbstractDeepClusteringAlgo, n_dims: int):
    nn = _get_dc_test_neuralnetwork(n_dims)
    algo.neural_network = nn
    algo.embedding_size = 4
    algo.random_state = 42
    algo.batch_size = 30
    if hasattr(algo, "pretrain_epochs"):
        algo.pretrain_epochs = 3
    if hasattr(algo, "clustering_epochs"):
        algo.clustering_epochs = 3


def _test_dc_algorithm_simple(algo: _AbstractDeepClusteringAlgo, check_random_state: bool = True, check_predict: bool = True, use_nr_data: bool = False):
    torch.use_deterministic_algorithms(True)
    if not use_nr_data:
        X, labels = _get_dc_test_data()
    else:
        X, labels = _get_dc_test_nr_data()
    _set_params_for_dc_algorithm(algo, X.shape[1])
    assert not hasattr(algo, "labels_")
    algo.fit(X)
    assert algo.labels_.dtype == np.int32
    assert algo.labels_.shape == labels.shape
    if hasattr(algo, "n_clusters"):
        if use_nr_data:
            assert np.array_equal([len(np.unique(algo.labels_[:, i])) for i in range(labels.shape[1])], algo.n_clusters)
        else:
            assert len(np.unique(algo.labels_)) == algo.n_clusters
    X_embed = algo.transform(X)
    assert X_embed.shape == (X.shape[0], algo.embedding_size)
    if check_random_state:
        # Test if random state is working
        algo2 = copy.deepcopy(algo)
        algo2.fit(X)
        assert np.array_equal(algo.labels_, algo2.labels_)
    else:
        algo2 = None
    if check_predict:
        # Test predict
        labels_predict = algo.predict(X)
        assert np.array_equal(algo.labels_, labels_predict)
    return algo, algo2


def _test_dc_algorithm_with_augmentation(algo: _AbstractDeepClusteringAlgo, custom_dataloaders: tuple = None, use_nr_data: bool = False):
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:100]
    labels = dataset.target[:100]
    if use_nr_data:
        labels = np.c_[labels // 2, labels % 2]
    _set_params_for_dc_algorithm(algo, data.shape[1] ** 2)
    if custom_dataloaders is None:
        aug_dl, orig_dl = get_default_augmented_dataloaders(data, batch_size=30)
        custom_dataloaders = (aug_dl, orig_dl)
    algo.custom_dataloaders = custom_dataloaders
    algo.augmentation_invariance = True
    assert not hasattr(algo, "labels_")
    algo.fit(data)
    assert algo.labels_.dtype == np.int32
    assert algo.labels_.shape == labels.shape
    if hasattr(algo, "n_clusters"):
        if use_nr_data:
            assert np.array_equal([len(np.unique(algo.labels_[:, i])) for i in range(algo.labels_.shape[1])], algo.n_clusters)
        else:
            assert len(np.unique(algo.labels_)) == algo.n_clusters


def _get_test_dataloader(data, batch_size, shuffle, drop_last):
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.arange(0, data.shape[0]), torch.from_numpy(data).float())),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last)
    return dataloader


class _TestAutoencoder(_AbstractAutoencoder):
    """
    A simple autoencoder only for test purposes.
    Encoder layers: [input_dim, embedding]
    Decoder layers: [embedding, input_dim]
    All features of the embedding will be equal to the sum of the input attributes.
    All weights are initialized as 1. Fitting function only sets fitting=True (no updates of the weights).
    """

    def __init__(self, input_dim, embedding_dim):
        super().__init__(True, 0)
        self.encoder = torch.nn.Linear(input_dim, embedding_dim, bias=False)
        self.encoder.weight.data.fill_(1)
        self.decoder = torch.nn.Linear(embedding_dim, input_dim, bias=False)
        self.decoder.weight.data.fill_(1)
        self.fitted = False
        self.allow_nd_input = False

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def loss(self, batch_data, loss_fn):
        reconstruction = self.forward(batch_data)
        loss = loss_fn(reconstruction, batch_data)
        return loss

    def fit(self):
        self.fitted = True
        return self
    
    def transform(self, X):
        torch_data = torch.from_numpy(X).float()
        embedded_data = self.encode(torch_data)
        X_embed = embedded_data.detach().numpy()
        return X_embed
    

class _TestClusterModule(torch.nn.Module):
    """
    A simple cluster module to test predict-related methods.
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def predict_hard(self, embedded, weights=None) -> torch.Tensor:
        """
        Hard prediction of given embedded samples. Returns the corresponding hard labels.
        Predicts 1 for all samples with mean(features) >= threshold and 0 for mean(features) < threshold.
        """
        predictions = (torch.mean(embedded, 1) >= self.threshold) * 1
        return predictions

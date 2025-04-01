from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.data import create_subspace_data
import torch


def test_abstract_deep_clustering_algo():
    fw_ae = FeedforwardAutoencoder([53, 4])
    adca = _AbstractDeepClusteringAlgo(256, fw_ae, None, 10, torch.device("cpu"), 1)
    adca.neural_network_trained_ = adca.neural_network
    X, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    X_embed = adca.transform(X)
    assert X_embed.shape == (1500, 4)

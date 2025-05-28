from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.data import create_subspace_data
import torch
import numpy as np
from clustpy.deep.tests._helpers_for_tests import _TestAutoencoder


def test_abstract_deep_clustering_algo():
    dummy_ae = _TestAutoencoder(5, 2)
    adca = _AbstractDeepClusteringAlgo(256, dummy_ae, None, 2, torch.device("cpu"), 1)
    adca.neural_network_trained_ = adca.neural_network
    adca.labels_ = None
    adca.n_features_in_ = 5
    X = np.array([[0, 1, 1, 1, 1], [5, 0, 0, 0, 0], [2, 3, 1, 0, 0], [10, 0, 0, 0, 0], [2, 3, 4, 5, 0], [5, 5, 5, 5, 1], [4, 4, 4, 4, 4]])
    X_embed = adca.transform(X)
    assert X_embed.shape == (7, 2)
    # Test predict - Create dataset and centers
    cluster_centers = np.array([[10, 10], [20, 20], [5, 5]])
    expected = np.array([2, 2, 2, 0, 0, 1, 1])
    # Predict embedded labels
    predicted_labels = adca.predict(X, cluster_centers)
    assert np.array_equal(expected, predicted_labels)
    adca.cluster_centers_ = cluster_centers
    predicted_labels = adca.predict(X)
    assert np.array_equal(expected, predicted_labels)

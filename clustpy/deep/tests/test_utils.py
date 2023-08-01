from clustpy.deep._utils import squared_euclidean_distance, detect_device, encode_batchwise, predict_batchwise, window, \
    int_to_one_hot, decode_batchwise, encode_decode_batchwise, run_initial_clustering, embedded_kmeans_prediction
from clustpy.deep.tests._helpers_for_tests import _get_test_dataloader, _TestAutoencoder, _TestClusterModule
from clustpy.data import create_subspace_data
import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from clustpy.partition import XMeans


def test_squared_euclidean_distance():
    tensor1 = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    tensor2 = torch.tensor([[1, 1, 1], [3, 4, 5], [5, 5, 5]])
    dist_tensor = squared_euclidean_distance(tensor1, tensor2)
    desired = torch.tensor([[0, 4 + 9 + 16, 16 * 3],
                            [1 * 3, 1 + 4 + 9, 9 * 3],
                            [4 * 3, 0 + 1 + 4, 4 * 3],
                            [9 * 3, 1 + 0 + 1, 1 * 3]])
    assert torch.equal(dist_tensor, desired)
    weights = torch.tensor([0.1, 0.2, 0.3])
    dist_tensor = squared_euclidean_distance(tensor1, tensor2, weights)
    desired = torch.tensor([[0, 0.01 * 4 + 0.04 * 9 + 0.09 * 16, 0.01 * 16 + 0.04 * 16 + 0.09 * 16],
                            [0.01 + 0.04 + 0.09, 0.01 * 1 + 0.04 * 4 + 0.09 * 9, 0.01 * 9 + 0.04 * 9 + 0.09 * 9],
                            [0.01 * 4 + 0.04 * 4 + 0.09 * 4, 0 + 0.04 * 1 + 0.09 * 4, 0.01 * 4 + 0.04 * 4 + 0.09 * 4],
                            [0.01 * 9 + 0.04 * 9 + 0.09 * 9, 0.01 * 1 + 0 + 0.09 * 1, 0.01 + 0.04 + 0.09]])
    assert torch.all(torch.isclose(dist_tensor, desired))  # torch.equal is not working due to numerical issues


def test_detect_device():
    # TODO idea for better test
    device = detect_device()
    assert type(device) is torch.device
    assert device.type == "cpu" or device.type == "cuda"


def test_encode_batchwise():
    # Load dataset
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    embedding_size = 5
    device = torch.device('cpu')
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    encoded = encode_batchwise(dataloader, autoencoder, device)
    # Each embedded feature should match the sum of the original features
    desired = np.sum(data, axis=1).reshape((-1, 1))
    desired = np.tile(desired, embedding_size)
    assert np.allclose(encoded, desired, atol=1e-5)


def test_predict_batchwise():
    # Load dataset
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    threshold = np.mean(np.sum(data, axis=1))
    embedding_size = 5
    device = torch.device('cpu')
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    cluster_module = _TestClusterModule(threshold)
    predictions = predict_batchwise(dataloader, autoencoder, cluster_module, device)
    # Check whether sum of the features (= embedded samples) is larger than the threshold
    desired = (np.sum(data, axis=1) >= threshold) * 1
    assert np.array_equal(predictions, desired)


def test_decode_batchwise():
    # Load dataset
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    embedding_size = 5
    device = torch.device('cpu')
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    decoded = decode_batchwise(dataloader, autoencoder, device)
    assert data.shape == decoded.shape


def test_encode_decode_batchwise():
    # Load dataset
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    embedding_size = 5
    device = torch.device('cpu')
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    encoded, decoded = encode_decode_batchwise(dataloader, autoencoder, device)
    # Each embedded feature should match the sum of the original features
    desired = np.sum(data, axis=1).reshape((-1, 1))
    desired = np.tile(desired, embedding_size)
    assert np.allclose(encoded, desired, atol=1e-5)
    assert data.shape == decoded.shape


def test_window():
    pass  # TODO


def test_int_to_one_hot():
    labels = torch.tensor([0, 0, 1, 2, 1])
    desired = torch.tensor([[1., 0., 0.],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
    onehot = int_to_one_hot(labels, 3)
    assert torch.equal(onehot, desired)


def test_run_initial_clustering():
    random_state = np.random.RandomState(1)
    X, L = make_blobs(1000, 3, centers=5, center_box=(-50, 50), random_state=random_state)
    # Test with KMeans
    n_clusters, labels, centers, clustering_algo = run_initial_clustering(X, 3, KMeans, {}, random_state)
    assert n_clusters == 3
    assert labels.shape == L.shape
    assert centers.shape == (n_clusters, X.shape[1])
    assert type(clustering_algo) is KMeans
    # Test with GMM
    n_clusters, labels, centers, clustering_algo = run_initial_clustering(X, 3, GaussianMixture, {"n_init": 3},
                                                                          random_state)
    assert n_clusters == 3
    assert labels.shape == L.shape
    assert centers.shape == (n_clusters, X.shape[1])
    assert type(clustering_algo) is GaussianMixture
    # Test with DBSCAN
    n_clusters, labels, centers, clustering_algo = run_initial_clustering(X, None, DBSCAN,
                                                                          {"eps": 2, "min_samples": 3}, random_state)
    assert n_clusters == 5
    assert labels.shape == L.shape
    assert centers.shape == (n_clusters, X.shape[1])
    assert type(clustering_algo) is DBSCAN
    # Test with XMeans
    n_clusters, labels, centers, clustering_algo = run_initial_clustering(X, None, XMeans, {}, random_state)
    assert n_clusters == 5
    assert labels.shape == L.shape
    assert centers.shape == (n_clusters, X.shape[1])
    assert type(clustering_algo) is XMeans


def test_embedded_kmeans_prediction():
    # Create dataset and centers
    data = np.array([[2, 1, 2], [1, 2, 2], [2, 2, 1], [10, 0, 1], [11, 0, 0], [19, 1, 0], [18, 2, 0]])
    cluster_centers = np.array([[10, 10], [20, 20], [5, 5]])
    # Create AE-related objects
    embedding_size = 2
    dataloader = _get_test_dataloader(data, 256, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    # Predict embedded labels
    predicted_labels = embedded_kmeans_prediction(dataloader, cluster_centers, autoencoder)
    expected = np.array([2, 2, 2, 0, 0, 1, 1])
    assert np.array_equal(expected, predicted_labels)

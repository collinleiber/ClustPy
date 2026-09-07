import numpy as np
from clustpy.deep._encoding_utils import encode_batchwise, predict_batchwise, decode_batchwise, encode_decode_batchwise
from clustpy.deep.tests._helpers_for_tests import _get_dc_test_data
from clustpy.deep.neural_networks import ConvolutionalAutoencoder
from clustpy.deep.tests._helpers_for_tests import _get_test_dataloader, _TestAutoencoder, _TestClusterModule


def test_encode_batchwise():
    # Load dataset
    data, _ = _get_dc_test_data()
    embedding_size = 4
    dataloader = _get_test_dataloader(data, 30, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    encoded = encode_batchwise(dataloader, autoencoder)
    # Each embedded feature should match the sum of the original features
    desired = np.sum(data, axis=1).reshape((-1, 1))
    desired = np.tile(desired, embedding_size)
    assert np.allclose(encoded, desired, atol=1e-5)
    # Test for Conv
    X_images = np.array([[[[11] * 32] * 32, [[12] * 32] * 32, [[13] * 32] * 32],
                         [[[10] * 32] * 32, [[20] * 32] * 32, [[30] * 32] * 32],
                         [[[10] * 32] * 32, [[40] * 32] * 32, [[70] * 32] * 32],
                         [[[1] * 32] * 32, [[1] * 32] * 32, [[1] * 32] * 32]])
    dataloader_images = _get_test_dataloader(X_images, 2, False, False)
    autoencoder_images = ConvolutionalAutoencoder(32, [512, 10])
    encoded_images = encode_batchwise(dataloader_images, autoencoder_images)
    assert encoded_images.shape == (4, 10)


def test_predict_batchwise():
    # Load dataset
    data, _ = _get_dc_test_data()
    threshold = np.mean(np.sum(data, axis=1))
    embedding_size = 4
    dataloader = _get_test_dataloader(data, 30, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    cluster_module = _TestClusterModule(threshold)
    predictions = predict_batchwise(dataloader, autoencoder, cluster_module)
    # Check whether sum of the features (= embedded samples) is larger than the threshold
    desired = (np.sum(data, axis=1) >= threshold) * 1
    assert np.array_equal(predictions, desired)


def test_decode_batchwise():
    # Load dataset
    data, _ = _get_dc_test_data()
    embedding_size = 4
    dataloader = _get_test_dataloader(data, 30, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    decoded = decode_batchwise(dataloader, autoencoder)
    assert data.shape == decoded.shape
    # Test for Conv
    X_images = np.array([[[[11] * 32] * 32, [[12] * 32] * 32, [[13] * 32] * 32],
                         [[[10] * 32] * 32, [[20] * 32] * 32, [[30] * 32] * 32],
                         [[[10] * 32] * 32, [[40] * 32] * 32, [[70] * 32] * 32],
                         [[[1] * 32] * 32, [[1] * 32] * 32, [[1] * 32] * 32]])
    dataloader_images = _get_test_dataloader(X_images, 2, False, False)
    autoencoder_images = ConvolutionalAutoencoder(32, [512, 10])
    decoded_images = decode_batchwise(dataloader_images, autoencoder_images)
    assert X_images.shape == decoded_images.shape


def test_encode_decode_batchwise():
    # Load dataset
    data, _ = _get_dc_test_data()
    embedding_size = 4
    dataloader = _get_test_dataloader(data, 30, False, False)
    autoencoder = _TestAutoencoder(data.shape[1], embedding_size)
    encoded, decoded = encode_decode_batchwise(dataloader, autoencoder)
    # Each embedded feature should match the sum of the original features
    desired = np.sum(data, axis=1).reshape((-1, 1))
    desired = np.tile(desired, embedding_size)
    assert np.allclose(encoded, desired, atol=1e-5)
    assert data.shape == decoded.shape
    # Test for Conv
    X_images = np.array([[[[11] * 32] * 32, [[12] * 32] * 32, [[13] * 32] * 32],
                         [[[10] * 32] * 32, [[20] * 32] * 32, [[30] * 32] * 32],
                         [[[10] * 32] * 32, [[40] * 32] * 32, [[70] * 32] * 32],
                         [[[1] * 32] * 32, [[1] * 32] * 32, [[1] * 32] * 32]])
    dataloader_images = _get_test_dataloader(X_images, 2, False, False)
    autoencoder_images = ConvolutionalAutoencoder(32, [512, 10])
    encoded_images, decoded_images = encode_decode_batchwise(dataloader_images, autoencoder_images)
    assert encoded_images.shape == (4, 10)
    assert X_images.shape == decoded_images.shape

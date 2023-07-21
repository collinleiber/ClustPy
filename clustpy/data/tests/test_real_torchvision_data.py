from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader, _check_normalized_channels
from clustpy.data import load_usps, load_mnist, load_fmnist, load_kmnist, load_cifar10, load_svhn, load_stl10
import torchvision.datasets
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_torchvision")


@pytest.fixture(autouse=True, scope='function')
def run_around_tests():
    # Code that will run before the tests
    if not os.path.isdir(TEST_DOWNLOAD_PATH):
        os.makedirs(TEST_DOWNLOAD_PATH)
    # Test functions will be run at this point
    yield
    # Code that will run after the tests
    shutil.rmtree(TEST_DOWNLOAD_PATH)


# Check if loading methods still exist (could be renamed/moved)
@pytest.mark.data
def test_torchvision_data_methods():
    assert "USPS" in dir(torchvision.datasets)
    assert "MNIST" in dir(torchvision.datasets)
    assert "KMNIST" in dir(torchvision.datasets)
    assert "FashionMNIST" in dir(torchvision.datasets)
    assert "CIFAR10" in dir(torchvision.datasets)
    assert "SVHN" in dir(torchvision.datasets)
    assert "STL10" in dir(torchvision.datasets)


# Do not skip USPS as it is the smallest dataset and can check the torchvision data loading mechanism
@pytest.mark.data
def test_load_usps():
    # Full data set
    data, labels = load_usps("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 9298, 256, 10)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_usps("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 7291, 256, 10)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_usps("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2007, 256, 10)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_usps("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (9298, 16, 16)


@pytest.mark.largedata
@pytest.mark.data
def test_load_mnist():
    # Full data set
    data, labels = load_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 70000, 784, 10)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 60000, 784, 10)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10000, 784, 10)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (70000, 28, 28)


@pytest.mark.largedata
@pytest.mark.data
def test_load_kmnist():
    # Full data set
    data, labels = load_kmnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 70000, 784, 10)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_kmnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 60000, 784, 10)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_kmnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10000, 784, 10)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_kmnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (70000, 28, 28)


@pytest.mark.largedata
@pytest.mark.data
def test_load_fmnist():
    # Full data set
    data, labels = load_fmnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 70000, 784, 10)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_fmnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 60000, 784, 10)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_fmnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10000, 784, 10)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_fmnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (70000, 28, 28)


# Do not skip cifar10 as it is the smallest 3-channel dataset and can check channel normalization
@pytest.mark.data
def test_load_cifar10():
    # Full data set
    data, labels = load_cifar10("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 60000, 3072, 10)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_cifar10("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 50000, 3072, 10)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_cifar10("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10000, 3072, 10)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_cifar10("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (60000, 3, 32, 32)


@pytest.mark.largedata
@pytest.mark.data
def test_load_svhn():
    # Full data set
    data, labels = load_svhn("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 99289, 3072, 10)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_svhn("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 73257, 3072, 10)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_svhn("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 26032, 3072, 10)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_svhn("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (99289, 3, 32, 32)


@pytest.mark.largedata
@pytest.mark.data
def test_load_stl10():
    # Full data set
    data, labels = load_stl10("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 13000, 27648, 10)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_stl10("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 5000, 27648, 10)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_stl10("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 8000, 27648, 10)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_stl10("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (13000, 3, 96, 96)

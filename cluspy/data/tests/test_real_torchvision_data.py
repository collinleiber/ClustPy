from cluspy.data.tests.test_real_world_data import _helper_test_data_loader
from cluspy.data.real_torchvision_data import *
from pathlib import Path
import os
import shutil
import pytest
import torchvision.datasets

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/cluspy_testfiles_torchvision")


@pytest.fixture(autouse=True, scope='session')
def run_around_tests():
    # Code that will run before the tests
    os.makedirs(TEST_DOWNLOAD_PATH)
    # Test functions will be run at this point
    yield
    # Code that will run after the tests
    shutil.rmtree(TEST_DOWNLOAD_PATH)


def _check_normalized_channels(data, channels, should_be_normalized=True):
    imprecision = 1e-5
    # Check is simple if we only have a single channel, i.e. a grayscale image
    if channels == 1:
        if should_be_normalized:
            assert np.mean(data) < imprecision
            assert abs(np.std(data) - 1) < imprecision
        else:
            assert np.mean(data) > imprecision
            assert abs(np.std(data) - 1) > imprecision
    else:
        # Else we have to check each channel separately
        for i in range(channels):
            if should_be_normalized:
                assert np.mean(data[:, np.arange(data.shape[1]) % channels == i]) < imprecision
                assert abs(np.std(data[:, np.arange(data.shape[1]) % channels == i]) - 1) < imprecision
            else:
                assert np.mean(data[:, np.arange(data.shape[1]) % channels == i]) > imprecision
                assert abs(np.std(data[:, np.arange(data.shape[1]) % channels == i]) - 1) > imprecision


# Check if loading methods still exist (could be renamed/moved)
def test_torchvision_data_methods():
    assert "USPS" in dir(torchvision.datasets)
    assert "MNIST" in dir(torchvision.datasets)
    assert "KMNIST" in dir(torchvision.datasets)
    assert "FashionMNIST" in dir(torchvision.datasets)
    assert "CIFAR10" in dir(torchvision.datasets)
    assert "SVHN" in dir(torchvision.datasets)
    assert "STL10" in dir(torchvision.datasets)


# Do not skip USPS as it is the smallest dataset and can check the torchvision data loading mechanism
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


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS and cifar10 test)")
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


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS and cifar10 test)")
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


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS and cifar10 test)")
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


# Do not skip cifar10 as it is the smallest 3-channel dataset and can check channel normalization
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


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS and cifar10 test)")
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


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS and cifar10 test)")
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

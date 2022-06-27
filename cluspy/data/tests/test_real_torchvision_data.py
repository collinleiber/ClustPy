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


# Check if loading methods still exist (could be renamed/moved)
def test_torchvision_data_methods():
    assert "MNIST" in dir(torchvision.datasets)
    assert "KMNIST" in dir(torchvision.datasets)
    assert "FashionMNIST" in dir(torchvision.datasets)
    assert "CIFAR10" in dir(torchvision.datasets)
    assert "SVHN" in dir(torchvision.datasets)
    assert "STL10" in dir(torchvision.datasets)


# Do not skip USPS as it is the smallest dataset and can check the torchvision data loading mechanism
def test_load_usps():
    # Full data set
    data, labels = load_usps("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (9298, 256)
    assert labels.shape == (9298,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_usps("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (7291, 256)
    assert labels.shape == (7291,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_usps("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (2007, 256)
    assert labels.shape == (2007,)
    assert np.array_equal(np.unique(labels), range(10))


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS test)")
def test_load_mnist():
    # Full data set
    data, labels = load_mnist("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (70000, 784)
    assert labels.shape == (70000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_mnist("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (60000, 784)
    assert labels.shape == (60000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (10000, 784)
    assert labels.shape == (10000,)
    assert np.array_equal(np.unique(labels), range(10))


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS test)")
def test_load_kmnist():
    # Full data set
    data, labels = load_kmnist("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (70000, 784)
    assert labels.shape == (70000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_kmnist("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (60000, 784)
    assert labels.shape == (60000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_kmnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (10000, 784)
    assert labels.shape == (10000,)
    assert np.array_equal(np.unique(labels), range(10))


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS test)")
def test_load_fmnist():
    # Full data set
    data, labels = load_fmnist("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (70000, 784)
    assert labels.shape == (70000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_fmnist("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (60000, 784)
    assert labels.shape == (60000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_fmnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (10000, 784)
    assert labels.shape == (10000,)
    assert np.array_equal(np.unique(labels), range(10))


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS test)")
def test_load_cifar10():
    # Full data set
    data, labels = load_cifar10("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (60000, 3072)
    assert labels.shape == (60000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_cifar10("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (50000, 3072)
    assert labels.shape == (50000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_cifar10("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (10000, 3072)
    assert labels.shape == (10000,)
    assert np.array_equal(np.unique(labels), range(10))


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS test)")
def test_load_svhn():
    # Full data set
    data, labels = load_svhn("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (99289, 3072)
    assert labels.shape == (99289,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_svhn("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (73257, 3072)
    assert labels.shape == (73257,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_svhn("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (26032, 3072)
    assert labels.shape == (26032,)
    assert np.array_equal(np.unique(labels), range(10))


@pytest.mark.skip(reason="torchvision should already test this (keep only USPS test)")
def test_load_stl10():
    # Full data set
    data, labels = load_stl10("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (13000, 27648)
    assert labels.shape == (13000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_stl10("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (5000, 27648)
    assert labels.shape == (5000,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_stl10("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (8000, 27648)
    assert labels.shape == (8000,)
    assert np.array_equal(np.unique(labels), range(10))

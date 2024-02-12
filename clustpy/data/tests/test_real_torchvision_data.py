from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_usps, load_mnist, load_fmnist, load_kmnist, load_cifar10, load_svhn, load_stl10, \
    load_gtsrb, load_cifar100
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
    assert "GTSRB" in dir(torchvision.datasets)
    assert "CIFAR100" in dir(torchvision.datasets)


# Do not skip USPS as it is the smallest dataset and can check the torchvision data loading mechanism
@pytest.mark.data
def test_load_usps():
    # Full data set
    dataset = _helper_test_data_loader(load_usps, 9298, 256, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (9298, 16, 16)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_usps, 7291, 256, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (7291, 16, 16)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_usps, 2007, 256, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (2007, 16, 16)
    assert dataset.image_format == "HW"


@pytest.mark.largedata
@pytest.mark.data
def test_load_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_mnist, 70000, 784, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (70000, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_mnist, 60000, 784, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (60000, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_mnist, 10000, 784, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10000, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.largedata
@pytest.mark.data
def test_load_kmnist():
    # Full data set
    dataset = _helper_test_data_loader(load_kmnist, 70000, 784, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (70000, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_kmnist, 60000, 784, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (60000, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_kmnist, 10000, 784, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10000, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.largedata
@pytest.mark.data
def test_load_fmnist():
    # Full data set
    dataset = _helper_test_data_loader(load_fmnist, 70000, 784, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (70000, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_fmnist, 60000, 784, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (60000, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_fmnist, 10000, 784, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10000, 28, 28)
    assert dataset.image_format == "HW"


# Do not skip cifar10 as it is the smallest 3-channel dataset and can check channel normalization
@pytest.mark.data
def test_load_cifar10():
    # Full data set
    dataset = _helper_test_data_loader(load_cifar10, 60000, 3072, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (60000, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_cifar10, 50000, 3072, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (50000, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_cifar10, 10000, 3072, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10000, 3, 32, 32)
    assert dataset.image_format == "CHW"


@pytest.mark.largedata
@pytest.mark.data
def test_load_cifar100():
    # Full data set
    dataset = _helper_test_data_loader(load_cifar100, 60000, 3072, 100,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (60000, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_cifar100, 50000, 3072, 100,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (50000, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_cifar100, 10000, 3072, 20,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH,
                                                          "use_superclasses": True})
    # Non-flatten
    assert dataset.images.shape == (10000, 3, 32, 32)
    assert dataset.image_format == "CHW"


@pytest.mark.largedata
@pytest.mark.data
def test_load_svhn():
    # Full data set
    dataset = _helper_test_data_loader(load_svhn, 99289, 3072, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (99289, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_svhn, 73257, 3072, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (73257, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_svhn, 26032, 3072, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (26032, 3, 32, 32)
    assert dataset.image_format == "CHW"


@pytest.mark.largedata
@pytest.mark.data
def test_load_stl10():
    # Full data set
    dataset = _helper_test_data_loader(load_stl10, 13000, 27648, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (13000, 3, 96, 96)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_stl10, 5000, 27648, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (5000, 3, 96, 96)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_stl10, 8000, 27648, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (8000, 3, 96, 96)
    assert dataset.image_format == "CHW"


@pytest.mark.data
# Do not skip GTSRB as the loading mechanism is different to the other torchvision dataloaders
def test_load_gtsrb():
    # Full data set
    dataset = _helper_test_data_loader(load_gtsrb, 39270, 3072, 43,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (39270, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_gtsrb, 26640, 3072, 43,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (26640, 3, 32, 32)
    assert dataset.image_format == "CHW"
    # Test data set (with image size 30x30)
    dataset = _helper_test_data_loader(load_gtsrb, 12630, 2700, 43,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH,
                                                          "image_size": (30, 30)})
    # Non-flatten
    assert dataset.images.shape == (12630, 3, 30, 30)
    assert dataset.image_format == "CHW"

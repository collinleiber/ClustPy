from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_iris, load_wine, load_breast_cancer, load_olivetti_faces, load_newsgroups, load_reuters, \
    load_imagenet_dog, load_imagenet10, load_coil20, load_coil100, load_webkb
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_realworld")


@pytest.fixture(autouse=True, scope='function')
def run_around_tests():
    # Code that will run before the tests
    if not os.path.isdir(TEST_DOWNLOAD_PATH):
        os.makedirs(TEST_DOWNLOAD_PATH)
    # Test functions will be run at this point
    yield
    # Code that will run after the tests
    shutil.rmtree(TEST_DOWNLOAD_PATH)


@pytest.mark.data
def test_load_iris():
    _helper_test_data_loader(load_iris, 150, 4, 3)


@pytest.mark.data
def test_load_wine():
    _helper_test_data_loader(load_wine, 178, 13, 3)


@pytest.mark.data
def test_load_breast_cancer():
    _helper_test_data_loader(load_breast_cancer, 569, 30, 2)


@pytest.mark.data
def test_load_olivetti_faces():
    dataset = _helper_test_data_loader(load_olivetti_faces, 400, 4096, 40)
    # Non-flatten
    assert dataset.images.shape == (400, 64, 64)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_newsgroups():
    # Full data set
    _helper_test_data_loader(load_newsgroups, 18846, 2000, 20, dataloader_params={"subset": "all"})
    # Train data set
    _helper_test_data_loader(load_newsgroups, 11314, 2000, 20, dataloader_params={"subset": "train"})
    # Test data set and different number of features
    _helper_test_data_loader(load_newsgroups, 7532, 500, 20, dataloader_params={"subset": "test", "n_features": 500})


@pytest.mark.data
@pytest.mark.largedata
def test_load_reuters():
    # Full data set
    _helper_test_data_loader(load_reuters, 685071, 2000, 4, dataloader_params={"subset": "all"})
    # Train data set
    _helper_test_data_loader(load_reuters, 19806, 2000, 4, dataloader_params={"subset": "train"})
    # Test data set and different number of features
    _helper_test_data_loader(load_reuters, 665265, 500, 4, dataloader_params={"subset": "test", "n_features": 500})


@pytest.mark.data
@pytest.mark.largedata
def test_load_imagenet_dog():
    # Full data set
    dataset = _helper_test_data_loader(load_imagenet_dog, 20580, 150528, 120,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH,
                                                          "breeds": None})
    # Non-flatten
    assert dataset.images.shape == (20580, 3, 224, 224)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_imagenet_dog, 12000, 150528, 120,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH,
                                                          "breeds": None})
    # Non-flatten
    assert dataset.images.shape == (12000, 3, 224, 224)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_imagenet_dog, 8580, 150528, 120,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH,
                                                          "breeds": None})
    # Non-flatten
    assert dataset.images.shape == (8580, 3, 224, 224)
    assert dataset.image_format == "CHW"
    # Test default breeds and different image size
    dataset = _helper_test_data_loader(load_imagenet_dog, 2574, 3072, 15,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH,
                                                          "image_size": (32, 32)})
    # Non-flatten
    assert dataset.images.shape == (2574, 3, 32, 32)
    assert dataset.image_format == "CHW"


@pytest.mark.data
@pytest.mark.largedata
def test_load_imagenet10():
    # Full data set
    dataset = _helper_test_data_loader(load_imagenet10, 13000, 150528, 10,
                                       dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (13000, 3, 224, 224)
    assert dataset.image_format == "CHW"
    # Test different image size
    dataset = _helper_test_data_loader(load_imagenet10, 13000, 27648, 10,
                                       dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH,
                                                          "use_224_size": False})
    # Non-flatten
    assert dataset.images.shape == (13000, 3, 96, 96)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_coil20():
    dataset = _helper_test_data_loader(load_coil20, 1440, 16384, 20,
                                       dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1440, 128, 128)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_coil100():
    dataset = _helper_test_data_loader(load_coil100, 7200, 49152, 100, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (7200, 3, 128, 128)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_webkb():
    _helper_test_data_loader(load_webkb, 1041, 323, [4, 4], dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})

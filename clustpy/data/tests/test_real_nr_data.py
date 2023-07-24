from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_aloi_small, load_fruit, load_nrletters, load_stickfigures, load_cmu_faces, load_webkb
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_nr")


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
def test_load_aloi_small():
    data, labels = load_aloi_small()
    _helper_test_data_loader(data, labels, 288, 611, [2, 2])


@pytest.mark.data
def test_load_fruit():
    data, labels = load_fruit()
    _helper_test_data_loader(data, labels, 105, 6, [3, 3])


@pytest.mark.data
def test_load_nrletters():
    data, labels = load_nrletters()
    _helper_test_data_loader(data, labels, 10000, 189, [6, 3, 4])
    # Test non-flatten
    data, _ = load_nrletters(flatten=False)
    assert data.shape == (10000, 3, 9, 7)


@pytest.mark.data
def test_load_stickfigures():
    data, labels = load_stickfigures()
    _helper_test_data_loader(data, labels, 900, 400, [3, 3])
    # Test non-flatten
    data, _ = load_stickfigures(flatten=False)
    assert data.shape == (900, 20, 20)


@pytest.mark.data
def test_load_cmu_faces():
    data, labels = load_cmu_faces(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 624, 960, [20, 4, 4, 2])
    # Test non-flatten
    data, _ = load_cmu_faces(downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (624, 30, 32)


@pytest.mark.data
def test_load_webkb():
    data, labels = load_webkb(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1041, 323, [4, 4])

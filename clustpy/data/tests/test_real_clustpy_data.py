from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_aloi_small, load_fruit, load_nrletters, load_stickfigures
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_clustpy")


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
    _helper_test_data_loader(load_aloi_small, 288, 611, [2, 2])


@pytest.mark.data
def test_load_fruit():
    _helper_test_data_loader(load_fruit, 105, 6, [3, 3])


@pytest.mark.data
def test_load_nrletters():
    dataset = _helper_test_data_loader(load_nrletters, 10000, 189, [6, 3, 4])
    # Test non-flatten
    assert dataset.images.shape == (10000, 3, 9, 7)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_stickfigures():
    dataset = _helper_test_data_loader(load_stickfigures, 900, 400, [3, 3])
    # Test non-flatten
    assert dataset.images.shape == (900, 20, 20)
    assert dataset.image_format == "HW"

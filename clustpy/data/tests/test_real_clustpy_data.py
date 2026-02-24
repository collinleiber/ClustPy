from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_aloi_small, load_fruit, load_nrletters, load_stickfigures
import pytest


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

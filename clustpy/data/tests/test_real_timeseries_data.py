from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_motestrain, load_proximal_phalanx_outline, load_diatom_size_reduction, load_symbols, \
    load_olive_oil, load_plane, load_sony_aibo_robot_surface, load_two_patterns, load_lsst
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_timeseries")


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
@pytest.mark.timeseriesdata
def test_load_motestrain():
    # Full data set
    data, labels = load_motestrain("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1272, 84, 2)
    # Train data set
    data, labels = load_motestrain("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 20, 84, 2)
    # Test data set
    data, labels = load_motestrain("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1252, 84, 2)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_proximal_phalanx_outline():
    # Full data set
    data, labels = load_proximal_phalanx_outline("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 876, 80, 2)
    # Train data set
    data, labels = load_proximal_phalanx_outline("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 600, 80, 2)
    # Test data set
    data, labels = load_proximal_phalanx_outline("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 276, 80, 2)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_diatom_size_reduction():
    # Full data set
    data, labels = load_diatom_size_reduction("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 322, 345, 4)
    # Train data set
    data, labels = load_diatom_size_reduction("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 16, 345, 4)
    # Test data set
    data, labels = load_diatom_size_reduction("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 306, 345, 4)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_symbols():
    # Full data set
    data, labels = load_symbols("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1020, 398, 6)
    # Train data set
    data, labels = load_symbols("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 25, 398, 6)
    # Test data set
    data, labels = load_symbols("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 995, 398, 6)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_olive_oil():
    # Full data set
    data, labels = load_olive_oil("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 60, 570, 4)
    # Train data set
    data, labels = load_olive_oil("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 30, 570, 4)
    # Test data set
    data, labels = load_olive_oil("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 30, 570, 4)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_plane():
    # Full data set
    data, labels = load_plane("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 210, 144, 7)
    # Train data set
    data, labels = load_plane("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 105, 144, 7)
    # Test data set
    data, labels = load_plane("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 105, 144, 7)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_sony_aibo_robot_surface():
    # Full data set
    data, labels = load_sony_aibo_robot_surface("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 621, 70, 2)
    # Train data set
    data, labels = load_sony_aibo_robot_surface("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 20, 70, 2)
    # Test data set
    data, labels = load_sony_aibo_robot_surface("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 601, 70, 2)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_two_patterns():
    # Full data set
    data, labels = load_two_patterns("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 5000, 128, 4)
    # Train data set
    data, labels = load_two_patterns("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1000, 128, 4)
    # Test data set
    data, labels = load_two_patterns("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 4000, 128, 4)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_lsst():
    # Full data set
    data, labels = load_lsst("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 4925, 216, 14)
    # Train data set
    data, labels = load_lsst("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2459, 216, 14)
    # Test data set
    data, labels = load_lsst("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2466, 216, 14)

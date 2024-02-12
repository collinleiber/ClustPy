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
    _helper_test_data_loader(load_motestrain, 1272, 84, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_motestrain, 20, 84, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_motestrain, 1252, 84, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_proximal_phalanx_outline():
    # Full data set
    _helper_test_data_loader(load_proximal_phalanx_outline, 876, 80, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_proximal_phalanx_outline, 600, 80, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_proximal_phalanx_outline, 276, 80, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_diatom_size_reduction():
    # Full data set
    _helper_test_data_loader(load_diatom_size_reduction, 322, 345, 4, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_diatom_size_reduction, 16, 345, 4, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_diatom_size_reduction, 306, 345, 4, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_symbols():
    # Full data set
    _helper_test_data_loader(load_symbols, 1020, 398, 6, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_symbols, 25, 398, 6, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_symbols, 995, 398, 6, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_olive_oil():
    # Full data set
    _helper_test_data_loader(load_olive_oil, 60, 570, 4, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_olive_oil, 30, 570, 4, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_olive_oil, 30, 570, 4, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_plane():
    # Full data set
    _helper_test_data_loader(load_plane, 210, 144, 7, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_plane, 105, 144, 7, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_plane, 105, 144, 7, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_sony_aibo_robot_surface():
    # Full data set
    _helper_test_data_loader(load_sony_aibo_robot_surface, 621, 70, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_sony_aibo_robot_surface, 20, 70, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_sony_aibo_robot_surface, 601, 70, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_two_patterns():
    # Full data set
    _helper_test_data_loader(load_two_patterns, 5000, 128, 4, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_two_patterns, 1000, 128, 4, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_two_patterns, 4000, 128, 4, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_lsst():
    # Full data set
    _helper_test_data_loader(load_lsst, 4925, 216, 14, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_lsst, 2459, 216, 14, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_lsst, 2466, 216, 14, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})

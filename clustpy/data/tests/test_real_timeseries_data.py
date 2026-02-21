from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_motestrain, load_proximal_phalanx_outline, load_diatom_size_reduction, load_symbols, \
    load_olive_oil, load_plane, load_sony_aibo_robot_surface, load_two_patterns, load_lsst
import pytest
import shutil


@pytest.fixture(autouse=True, scope='function')
def my_tmp_dir(tmp_path):
    # Code that will run before the tests
    tmp_dir = str(tmp_path)
    # Test functions will be run at this point
    yield tmp_dir
    # Code that will run after the tests
    shutil.rmtree(tmp_dir)


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_motestrain(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_motestrain, 1272, 84, 2, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_motestrain, 20, 84, 2, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_motestrain, 1252, 84, 2, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_proximal_phalanx_outline(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_proximal_phalanx_outline, 876, 80, 2, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_proximal_phalanx_outline, 600, 80, 2, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_proximal_phalanx_outline, 276, 80, 2, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_diatom_size_reduction(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_diatom_size_reduction, 322, 345, 4, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_diatom_size_reduction, 16, 345, 4, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_diatom_size_reduction, 306, 345, 4, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_symbols(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_symbols, 1020, 398, 6, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_symbols, 25, 398, 6, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_symbols, 995, 398, 6, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_olive_oil(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_olive_oil, 60, 570, 4, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_olive_oil, 30, 570, 4, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_olive_oil, 30, 570, 4, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_plane(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_plane, 210, 144, 7, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_plane, 105, 144, 7, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_plane, 105, 144, 7, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_sony_aibo_robot_surface(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_sony_aibo_robot_surface, 621, 70, 2, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_sony_aibo_robot_surface, 20, 70, 2, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_sony_aibo_robot_surface, 601, 70, 2, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_two_patterns(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_two_patterns, 5000, 128, 4, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_two_patterns, 1000, 128, 4, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_two_patterns, 4000, 128, 4, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})


@pytest.mark.data
@pytest.mark.timeseriesdata
def test_load_lsst(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_lsst, 4925, 216, 14, dataloader_params={"subset": "all", "downloads_path":my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_lsst, 2459, 216, 14, dataloader_params={"subset": "train", "downloads_path":my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_lsst, 2466, 216, 14, dataloader_params={"subset": "test", "downloads_path":my_tmp_dir})

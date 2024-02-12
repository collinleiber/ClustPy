from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_banknotes, load_spambase, load_seeds, load_skin, load_soybean_small, load_soybean_large, \
    load_pendigits, load_ecoli, load_htru2, load_letterrecognition, load_har, load_statlog_shuttle, load_mice_protein, \
    load_user_knowledge, load_breast_tissue, load_forest_types, load_dermatology, load_multiple_features, \
    load_statlog_australian_credit_approval, load_breast_cancer_wisconsin_original, load_optdigits, load_semeion, \
    load_cmu_faces
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_uci")


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
def test_load_banknotes():
    _helper_test_data_loader(load_banknotes, 1372, 4, 2, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_spambase():
    _helper_test_data_loader(load_spambase, 4601, 57, 2, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_seeds():
    _helper_test_data_loader(load_seeds, 210, 7, 3, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_skin():
    _helper_test_data_loader(load_skin, 245057, 3, 2, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_soybean_small():
    _helper_test_data_loader(load_soybean_small, 47, 35, 4, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_soybean_large():
    # Full data set
    _helper_test_data_loader(load_soybean_large, 562, 35, 15,
                             dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_soybean_large, 266, 35, 15,
                             dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_soybean_large, 296, 35, 15,
                             dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_pendigits():
    # Full data set
    _helper_test_data_loader(load_pendigits, 10992, 16, 10,
                             dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_pendigits, 7494, 16, 10,
                             dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_pendigits, 3498, 16, 10,
                             dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_ecoli():
    _helper_test_data_loader(load_ecoli, 336, 7, 8, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Check if ignoring small clusters works
    _helper_test_data_loader(load_ecoli, 327, 7, 5,
                             dataloader_params={"ignore_small_clusters": True, "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_hrtu2():
    _helper_test_data_loader(load_htru2, 17898, 8, 2, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_letterrecognition():
    _helper_test_data_loader(load_letterrecognition, 20000, 16, 26,
                             dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_har():
    # Full data set
    _helper_test_data_loader(load_har, 10299, 561, 6,
                             dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_har, 7352, 561, 6,
                             dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_har, 2947, 561, 6,
                             dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_statlog_shuttle():
    # 7z probably not installed! -> data and labels can be None
    dataset = load_statlog_shuttle(downloads_path=TEST_DOWNLOAD_PATH)
    if dataset is not None:
        # Full data set
        _helper_test_data_loader(load_statlog_shuttle, 58000, 9, 7,
                                 dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
        # Train data set
        _helper_test_data_loader(load_statlog_shuttle, 43500, 9, 7,
                                 dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
        # Test data set
        _helper_test_data_loader(load_statlog_shuttle, 14500, 9, 7,
                                 dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_mice_protein():
    _helper_test_data_loader(load_mice_protein, 1077, 68, 8, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Check if additional labels work
    _helper_test_data_loader(load_mice_protein, 1077, 68, [8, 72, 2, 2, 2],
                             dataloader_params={"return_additional_labels": True, "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_user_knowledge():
    # Full data set
    _helper_test_data_loader(load_user_knowledge, 403, 5, 4,
                             dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_user_knowledge, 258, 5, 4,
                             dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_user_knowledge, 145, 5, 4,
                             dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_breast_tissue():
    _helper_test_data_loader(load_breast_tissue, 106, 9, 6, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_forest_types():
    # Full data set
    _helper_test_data_loader(load_forest_types, 523, 27, 4,
                             dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Train data set
    _helper_test_data_loader(load_forest_types, 198, 27, 4,
                             dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Test data set
    _helper_test_data_loader(load_forest_types, 325, 27, 4,
                             dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_dermatology():
    _helper_test_data_loader(load_dermatology, 358, 34, 6, dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_multiple_features():
    _helper_test_data_loader(load_multiple_features, 2000, 649, 10,
                             dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_statlog_australian_credit_approval():
    _helper_test_data_loader(load_statlog_australian_credit_approval, 690, 14, 2,
                             dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_breast_cancer_wisconsin_original():
    _helper_test_data_loader(load_breast_cancer_wisconsin_original, 683, 9, 2,
                             dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})


@pytest.mark.data
def test_load_optdigits():
    # Full data set
    dataset = _helper_test_data_loader(load_optdigits, 5620, 64, 10,
                                       dataloader_params={"subset": "all", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (5620, 8, 8)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_optdigits, 3823, 64, 10,
                                       dataloader_params={"subset": "train", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (3823, 8, 8)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_optdigits, 1797, 64, 10,
                                       dataloader_params={"subset": "test", "downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1797, 8, 8)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_semeion():
    dataset = _helper_test_data_loader(load_semeion, 1593, 256, 10,
                                       dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1593, 16, 16)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_cmu_faces():
    dataset = _helper_test_data_loader(load_cmu_faces, 624, 960, [20, 4, 4, 2],
                                       dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (624, 30, 32)
    assert dataset.image_format == "HW"

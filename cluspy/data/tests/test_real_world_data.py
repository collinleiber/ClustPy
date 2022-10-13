from cluspy.data.real_world_data import *
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/cluspy_testfiles_realworld")


@pytest.fixture(autouse=True, scope='session')
def run_around_tests():
    # Code that will run before the tests
    os.makedirs(TEST_DOWNLOAD_PATH)
    # Test functions will be run at this point
    yield
    # Code that will run after the tests
    shutil.rmtree(TEST_DOWNLOAD_PATH)


def _helper_test_data_loader(data, labels, N, d, k):
    assert data.shape == (N, d)
    if type(k) is int:
        assert labels.shape == (N,)
        assert np.array_equal(np.unique(labels), range(k))
    else:
        # In case of datasets for alternative clusterings
        assert labels.shape == (N, len(k))
        unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
        assert [len(l) for l in unique_labels] == k  # Checks that number of labels is correct
        for ul in unique_labels:
            assert np.array_equal(ul, range(len(ul)))  # Checks that labels go from 0 to k


def test_load_iris():
    data, labels = load_iris()
    _helper_test_data_loader(data, labels, 150, 4, 3)


def test_load_wine():
    data, labels = load_wine()
    _helper_test_data_loader(data, labels, 178, 13, 3)


def test_load_breast_cancer():
    data, labels = load_breast_cancer()
    _helper_test_data_loader(data, labels, 569, 30, 2)


def test_load_newsgroups():
    # Full data set
    data, labels = load_newsgroups("all")
    _helper_test_data_loader(data, labels, 18846, 2000, 20)
    # Train data set
    data, labels = load_newsgroups("train")
    _helper_test_data_loader(data, labels, 11314, 2000, 20)
    # Test data set and different number of features
    data, labels = load_newsgroups("test", 500)
    _helper_test_data_loader(data, labels, 7532, 500, 20)


def test_load_reuters():
    # Full data set
    data, labels = load_reuters("all")
    _helper_test_data_loader(data, labels, 685071, 2000, 4)
    # Train data set
    data, labels = load_reuters("train")
    _helper_test_data_loader(data, labels, 19806, 2000, 4)
    # Test data set and different number of features
    data, labels = load_reuters("test", 500)
    _helper_test_data_loader(data, labels, 665265, 500, 4)


def test_load_banknotes():
    data, labels = load_banknotes(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1372, 4, 2)


def test_load_spambase():
    data, labels = load_spambase(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 4601, 57, 2)


def test_load_seeds():
    data, labels = load_seeds(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 210, 7, 3)


def test_load_skin():
    data, labels = load_skin(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 245057, 3, 2)


def test_load_soybean_small():
    data, labels = load_soybean_small(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 47, 35, 4)


def test_load_soybean_large():
    # Full data set
    data, labels = load_soybean_large("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 562, 35, 15)
    # Train data set
    data, labels = load_soybean_large("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 266, 35, 15)
    # Test data set
    data, labels = load_soybean_large("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 296, 35, 15)


def test_load_optdigits():
    # Full data set
    data, labels = load_optdigits("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 5620, 64, 10)
    # Train data set
    data, labels = load_optdigits("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 3823, 64, 10)
    # Test data set
    data, labels = load_optdigits("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1797, 64, 10)


def test_load_pendigits():
    # Full data set
    data, labels = load_pendigits("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10992, 16, 10)
    # Train data set
    data, labels = load_pendigits("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 7494, 16, 10)
    # Test data set
    data, labels = load_pendigits("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 3498, 16, 10)


def test_load_ecoli():
    data, labels = load_ecoli(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 336, 7, 8)
    # Check if ignoring small clusters works
    data, labels = load_ecoli(ignore_small_clusters=True, downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 327, 7, 5)


def test_load_hrtu2():
    data, labels = load_htru2(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 17898, 8, 2)


def test_load_letterrecognition():
    data, labels = load_letterrecognition(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 20000, 16, 26)


def test_load_har():
    # Full data set
    data, labels = load_har("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10299, 561, 6)
    # Train data set
    data, labels = load_har("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 7352, 561, 6)
    # Test data set
    data, labels = load_har("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2947, 561, 6)


def test_load_statlog_shuttle():
    # 7z probably not installed! -> data and labels can be None
    # Full data set
    data, labels = load_statlog_shuttle("all", downloads_path=TEST_DOWNLOAD_PATH)
    if data is None:
        assert labels is None
    else:
        _helper_test_data_loader(data, labels, 58000, 9, 7)
    # Train data set
    data, labels = load_statlog_shuttle("train", downloads_path=TEST_DOWNLOAD_PATH)
    if data is None:
        assert labels is None
    else:
        _helper_test_data_loader(data, labels, 43500, 9, 7)
    # Test data set
    data, labels = load_statlog_shuttle("test", downloads_path=TEST_DOWNLOAD_PATH)
    if data is None:
        assert labels is None
    else:
        _helper_test_data_loader(data, labels, 14500, 9, 7)


def test_load_mice_protein():
    data, labels = load_mice_protein(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1077, 68, 8)
    # Check if additional labels work
    data, labels = load_mice_protein(return_additional_labels=True, downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1077, 68, [8, 72, 2, 2, 2])


def test_load_user_knowledge():
    # Full data set
    data, labels = load_user_knowledge("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 403, 5, 4)
    # Train data set
    data, labels = load_user_knowledge("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 258, 5, 4)
    # Test data set
    data, labels = load_user_knowledge("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 145, 5, 4)


def test_load_breast_tissue():
    data, labels = load_breast_tissue(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 106, 9, 6)


def test_load_forest_types():
    # Full data set
    data, labels = load_forest_types("all", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 523, 27, 4)
    # Train data set
    data, labels = load_forest_types("train", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 198, 27, 4)
    # Test data set
    data, labels = load_forest_types("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 325, 27, 4)


def test_load_dermatology():
    data, labels = load_dermatology(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 358, 34, 6)


def test_load_multiple_features():
    data, labels = load_multiple_features(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2000, 649, 10)


def test_load_statlog_australian_credit_approval():
    data, labels = load_statlog_australian_credit_approval(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 690, 14, 2)


def test_load_breast_cancer_wisconsin_original():
    data, labels = load_breast_cancer_wisconsin_original(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 683, 9, 2)


def test_load_semeion():
    data, labels = load_semeion(downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1593, 256, 10)


"""
Test timeseries classification data
"""


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

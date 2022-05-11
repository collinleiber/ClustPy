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


def test_load_iris():
    data, labels = load_iris()
    assert data.shape == (150, 4)
    assert labels.shape == (150,)
    assert np.array_equal(np.unique(labels), range(3))


def test_load_wine():
    data, labels = load_wine()
    assert data.shape == (178, 13)
    assert labels.shape == (178,)
    assert np.array_equal(np.unique(labels), range(3))


def test_load_breast_cancer():
    data, labels = load_breast_cancer()
    assert data.shape == (569, 30)
    assert labels.shape == (569,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_newsgroups():
    # Full data set
    data, labels = load_newsgroups("all")
    assert data.shape == (18846, 2000)
    assert labels.shape == (18846,)
    assert np.array_equal(np.unique(labels), range(20))
    # Train data set
    data, labels = load_newsgroups("train")
    assert data.shape == (11314, 2000)
    assert labels.shape == (11314,)
    assert np.array_equal(np.unique(labels), range(20))
    # Test data set and different number of features
    data, labels = load_newsgroups("test", 500)
    assert data.shape == (7532, 500)
    assert labels.shape == (7532,)
    assert np.array_equal(np.unique(labels), range(20))


def test_load_reuters():
    # Full data set
    data, labels = load_reuters("all")
    assert data.shape == (685071, 2000)
    assert labels.shape == (685071,)
    assert np.array_equal(np.unique(labels), range(4))
    # Train data set
    data, labels = load_reuters("train")
    assert data.shape == (19806, 2000)
    assert labels.shape == (19806,)
    assert np.array_equal(np.unique(labels), range(4))
    # Test data set and different number of features
    data, labels = load_reuters("test", 500)
    assert data.shape == (665265, 500)
    assert labels.shape == (665265,)
    assert np.array_equal(np.unique(labels), range(4))


def test_load_banknotes():
    data, labels = load_banknotes(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1372, 4)
    assert labels.shape == (1372,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_spambase():
    data, labels = load_spambase(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (4601, 57)
    assert labels.shape == (4601,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_seeds():
    data, labels = load_seeds(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (210, 7)
    assert labels.shape == (210,)
    assert np.array_equal(np.unique(labels), range(3))


def test_load_skin():
    data, labels = load_skin(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (245057, 3)
    assert labels.shape == (245057,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_soybean_small():
    data, labels = load_soybean_small(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (47, 35)
    assert labels.shape == (47,)
    assert np.array_equal(np.unique(labels), range(4))


def test_load_soybean_large():
    # Full data set
    data, labels = load_soybean_large("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (562, 35)
    assert labels.shape == (562,)
    assert np.array_equal(np.unique(labels), range(15))
    # Train data set
    data, labels = load_soybean_large("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (266, 35)
    assert labels.shape == (266,)
    assert np.array_equal(np.unique(labels), range(15))
    # Test data set
    data, labels = load_soybean_large("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (296, 35)
    assert labels.shape == (296,)
    assert np.array_equal(np.unique(labels), range(15))


def test_load_optdigits():
    # Full data set
    data, labels = load_optdigits("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (5620, 64)
    assert labels.shape == (5620,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_optdigits("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (3823, 64)
    assert labels.shape == (3823,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_optdigits("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1797, 64)
    assert labels.shape == (1797,)
    assert np.array_equal(np.unique(labels), range(10))


def test_load_pendigits():
    # Full data set
    data, labels = load_pendigits("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (10992, 16)
    assert labels.shape == (10992,)
    assert np.array_equal(np.unique(labels), range(10))
    # Train data set
    data, labels = load_pendigits("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (7494, 16)
    assert labels.shape == (7494,)
    assert np.array_equal(np.unique(labels), range(10))
    # Test data set
    data, labels = load_pendigits("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (3498, 16)
    assert labels.shape == (3498,)
    assert np.array_equal(np.unique(labels), range(10))


def test_load_ecoli():
    data, labels = load_ecoli(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (336, 7)
    assert labels.shape == (336,)
    assert np.array_equal(np.unique(labels), range(8))
    # Check if ignoring small clusters works
    data, labels = load_ecoli(ignore_small_clusters=True, downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (327, 7)
    assert labels.shape == (327,)
    assert np.array_equal(np.unique(labels), range(5))


def test_load_hrtu2():
    data, labels = load_htru2(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (17898, 8)
    assert labels.shape == (17898,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_letterrecognition():
    data, labels = load_letterrecognition(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (20000, 16)
    assert labels.shape == (20000,)
    assert np.array_equal(np.unique(labels), range(26))


def test_load_har():
    # Full data set
    data, labels = load_har("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (10299, 561)
    assert labels.shape == (10299,)
    assert np.array_equal(np.unique(labels), range(6))
    # Train data set
    data, labels = load_har("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (7352, 561)
    assert labels.shape == (7352,)
    assert np.array_equal(np.unique(labels), range(6))
    # Test data set
    data, labels = load_har("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (2947, 561)
    assert labels.shape == (2947,)
    assert np.array_equal(np.unique(labels), range(6))


def test_load_statlog_shuttle():
    # 7z probably not installed!
    # Full data set
    data, labels = load_statlog_shuttle("all", downloads_path=TEST_DOWNLOAD_PATH)
    if data is None:
        assert labels is None
    else:
        assert data.shape == (58000, 9)
        assert labels.shape == (58000,)
        assert np.array_equal(np.unique(labels), range(7))
    # Train data set
    data, labels = load_statlog_shuttle("train", downloads_path=TEST_DOWNLOAD_PATH)
    if data is None:
        assert labels is None
    else:
        assert data.shape == (43500, 9)
        assert labels.shape == (43500,)
        assert np.array_equal(np.unique(labels), range(7))
    # Test data set
    data, labels = load_statlog_shuttle("test", downloads_path=TEST_DOWNLOAD_PATH)
    if data is None:
        assert labels is None
    else:
        assert data.shape == (14500, 9)
        assert labels.shape == (14500,)
        assert np.array_equal(np.unique(labels), range(7))


def test_load_mice_protein():
    data, labels = load_mice_protein(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1077, 68)
    assert labels.shape == (1077,)
    assert np.array_equal(np.unique(labels), range(8))
    # Check if additional labels work
    data, labels = load_mice_protein(return_additional_labels=True, downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1077, 68)
    assert labels.shape == (1077, 5)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [8, 72, 2, 2, 2]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))


def test_load_user_knowledge():
    # Full data set
    data, labels = load_user_knowledge("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (403, 5)
    assert labels.shape == (403,)
    assert np.array_equal(np.unique(labels), range(4))
    # Train data set
    data, labels = load_user_knowledge("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (258, 5)
    assert labels.shape == (258,)
    assert np.array_equal(np.unique(labels), range(4))
    # Test data set
    data, labels = load_user_knowledge("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (145, 5)
    assert labels.shape == (145,)
    assert np.array_equal(np.unique(labels), range(4))


def test_load_breast_tissue():
    data, labels = load_breast_tissue(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (106, 9)
    assert labels.shape == (106,)
    assert np.array_equal(np.unique(labels), range(6))


def test_load_forest_types():
    # Full data set
    data, labels = load_forest_types("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (523, 27)
    assert labels.shape == (523,)
    assert np.array_equal(np.unique(labels), range(4))
    # Train data set
    data, labels = load_forest_types("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (198, 27)
    assert labels.shape == (198,)
    assert np.array_equal(np.unique(labels), range(4))
    # Test data set
    data, labels = load_forest_types("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (325, 27)
    assert labels.shape == (325,)
    assert np.array_equal(np.unique(labels), range(4))


def test_load_motestrain():
    # Full data set
    data, labels = load_motestrain("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1272, 84)
    assert labels.shape == (1272,)
    assert np.array_equal(np.unique(labels), range(2))
    # Train data set
    data, labels = load_motestrain("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (20, 84)
    assert labels.shape == (20,)
    assert np.array_equal(np.unique(labels), range(2))
    # Test data set
    data, labels = load_motestrain("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1252, 84)
    assert labels.shape == (1252,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_proximal_phalanx_outline():
    # Full data set
    data, labels = load_proximal_phalanx_outline("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (876, 80)
    assert labels.shape == (876,)
    assert np.array_equal(np.unique(labels), range(2))
    # Train data set
    data, labels = load_proximal_phalanx_outline("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (600, 80)
    assert labels.shape == (600,)
    assert np.array_equal(np.unique(labels), range(2))
    # Test data set
    data, labels = load_proximal_phalanx_outline("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (276, 80)
    assert labels.shape == (276,)
    assert np.array_equal(np.unique(labels), range(2))


def test_load_diatom_size_reduction():
    # Full data set
    data, labels = load_diatom_size_reduction("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (322, 345)
    assert labels.shape == (322,)
    assert np.array_equal(np.unique(labels), range(4))
    # Train data set
    data, labels = load_diatom_size_reduction("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (16, 345)
    assert labels.shape == (16,)
    assert np.array_equal(np.unique(labels), range(4))
    # Test data set
    data, labels = load_diatom_size_reduction("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (306, 345)
    assert labels.shape == (306,)
    assert np.array_equal(np.unique(labels), range(4))


def test_load_symbols():
    # Full data set
    data, labels = load_symbols("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1020, 398)
    assert labels.shape == (1020,)
    assert np.array_equal(np.unique(labels), range(6))
    # Train data set
    data, labels = load_symbols("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (25, 398)
    assert labels.shape == (25,)
    assert np.array_equal(np.unique(labels), range(6))
    # Test data set
    data, labels = load_symbols("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (995, 398)
    assert labels.shape == (995,)
    assert np.array_equal(np.unique(labels), range(6))


def test_load_olive_oil():
    # Full data set
    data, labels = load_olive_oil("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (60, 570)
    assert labels.shape == (60,)
    assert np.array_equal(np.unique(labels), range(4))
    # Train data set
    data, labels = load_olive_oil("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (30, 570)
    assert labels.shape == (30,)
    assert np.array_equal(np.unique(labels), range(4))
    # Test data set
    data, labels = load_olive_oil("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (30, 570)
    assert labels.shape == (30,)
    assert np.array_equal(np.unique(labels), range(4))


def test_load_plane():
    # Full data set
    data, labels = load_plane("all", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (210, 144)
    assert labels.shape == (210,)
    assert np.array_equal(np.unique(labels), range(7))
    # Train data set
    data, labels = load_plane("train", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (105, 144)
    assert labels.shape == (105,)
    assert np.array_equal(np.unique(labels), range(7))
    # Test data set
    data, labels = load_plane("test", downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (105, 144)
    assert labels.shape == (105,)
    assert np.array_equal(np.unique(labels), range(7))

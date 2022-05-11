from cluspy.data.real_nr_data import *
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/cluspy_testfiles_nr")


@pytest.fixture(autouse=True, scope='session')
def run_around_tests():
    # Code that will run before the tests
    os.makedirs(TEST_DOWNLOAD_PATH)
    # Test functions will be run at this point
    yield
    # Code that will run after the tests
    shutil.rmtree(TEST_DOWNLOAD_PATH)


def test_load_aloi_small():
    data, labels = load_aloi_small()
    assert data.shape == (288, 611)
    assert labels.shape == (288, 2)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [2, 2]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))


def test_load_fruit():
    data, labels = load_fruit()
    assert data.shape == (105, 6)
    assert labels.shape == (105, 2)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [3, 3]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))


def test_load_nrletters():
    data, labels = load_nrletters()
    assert data.shape == (10000, 189)
    assert labels.shape == (10000, 3)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [6, 3, 4]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))


def test_load_stickfigures():
    data, labels = load_stickfigures()
    assert data.shape == (900, 400)
    assert labels.shape == (900, 2)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [3, 3]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))


def test_load_cmu_faces():
    data, labels = load_cmu_faces(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (624, 960)
    assert labels.shape == (624, 4)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [20, 4, 4, 2]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))


def test_load_webkb():
    data, labels = load_webkb(downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (1041, 323)
    assert labels.shape == (1041, 2)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [4, 4]
    for ul in unique_labels:
        assert np.array_equal(ul, range(len(ul)))

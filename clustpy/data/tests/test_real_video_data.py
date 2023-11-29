from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader, _check_normalized_channels
from clustpy.data import load_video_weizmann, load_video_keck_gesture
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_video")


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
def test_load_video_weizmann():
    data, labels = load_video_weizmann(downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 5687, 77760, [10, 9])
    _check_normalized_channels(data, 3, True)
    # Without normalize
    data, labels = load_video_weizmann(downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False,
                                       image_size=(100, 100))
    _helper_test_data_loader(data, labels, 5687, 30000, [10, 9])
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_video_weizmann(flatten=False, downloads_path=TEST_DOWNLOAD_PATH,
                                  frame_sampling_ratio=0.5)
    assert data.shape[0] / 5687 < 0.55 and data.shape[0] / 5687 > 0.5
    assert data.shape == (data.shape[0], 3, 144, 180)


@pytest.mark.largedata
@pytest.mark.data
def test_load_video_keck_gesture():
    data, labels = load_video_keck_gesture(subset="all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 25457, 120000, [15, 4])
    _check_normalized_channels(data, 3, True)
    # Test data
    data, labels = load_video_keck_gesture(subset="test", image_size=(150, 150),
                                           downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 11911, 67500, [15, 4])
    _check_normalized_channels(data, 3, False)
    # Train data
    data, labels = load_video_keck_gesture(subset="train", image_size=(150, 150),
                                           downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 13546, 67500, [15, 3])
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_video_keck_gesture(subset="test", flatten=False, downloads_path=TEST_DOWNLOAD_PATH,
                                      frame_sampling_ratio=0.5)
    assert data.shape[0] / 11911 < 0.55 and data.shape[0] / 11911 > 0.5
    assert data.shape == (data.shape[0], 3, 200, 200)

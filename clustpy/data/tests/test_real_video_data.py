import numpy as np
from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader, _check_normalized_channels
from clustpy.data import load_video_weizmann, load_video_keck_gesture
from clustpy.data.real_video_data import _downsample_frames
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


def test_downsample_frames():
    data_in = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    labels_in = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    data_out, labels_out = _downsample_frames(data_in, labels_in, 1)
    assert np.array_equal(data_out, np.array(data_in)) and np.array_equal(labels_out, np.array(labels_in))
    data_out, labels_out = _downsample_frames(data_in, labels_in, 0.75)
    assert data_out.shape[0] == 9
    assert np.array_equal(data_out, np.array([1, 2, 3, 4, 5, 7, 8, 9, 10])) and np.array_equal(labels_out, data_out)
    data_out, labels_out = _downsample_frames(data_in, labels_in, 0.5)
    assert data_out.shape[0] == 6
    assert np.array_equal(data_out, np.array([1, 3, 5, 6, 8, 10])) and np.array_equal(labels_out, data_out)
    data_out, labels_out = _downsample_frames(data_in, labels_in, 0.25)
    assert data_out.shape[0] == 3
    assert np.array_equal(data_out, np.array([2, 5, 9])) and np.array_equal(labels_out, data_out)
    data_out, labels_out = _downsample_frames(data_in, labels_in, 0.0001)
    assert data_out.shape[0] == 1
    assert np.array_equal(data_out, np.array([5])) and np.array_equal(labels_out, data_out)


@pytest.mark.data
def test_load_video_weizmann():
    data, labels = load_video_weizmann(downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, None, 77760, [10, 9]) # N not always 5687
    _check_normalized_channels(data, 3, True)
    # Without normalize
    data, labels = load_video_weizmann(downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False,
                                       image_size=(100, 100))
    _helper_test_data_loader(data, labels, None, 30000, [10, 9]) # N not always 5687
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_video_weizmann(flatten=False, downloads_path=TEST_DOWNLOAD_PATH,
                                  frame_sampling_ratio=0.5)
    assert data.shape[0] / 5687 < 0.55 and data.shape[0] / 5687 > 0.49
    assert data.shape == (data.shape[0], 3, 144, 180)


@pytest.mark.largedata
@pytest.mark.data
def test_load_video_keck_gesture():
    data, labels = load_video_keck_gesture(subset="all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, None, 120000, [15, 4]) # N not always 25457
    _check_normalized_channels(data, 3, True)
    # Test data
    data, labels = load_video_keck_gesture(subset="test", image_size=(150, 150),
                                           downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, None, 67500, [15, 4]) # N not always 11911
    _check_normalized_channels(data, 3, False)
    # Train data
    data, labels = load_video_keck_gesture(subset="train", image_size=(150, 150),
                                           downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, None, 67500, [15, 3]) # N not always 13546
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_video_keck_gesture(subset="test", flatten=False, downloads_path=TEST_DOWNLOAD_PATH,
                                      frame_sampling_ratio=0.5)
    assert data.shape[0] / 11911 < 0.55 and data.shape[0] / 11911 > 0.49
    assert data.shape == (data.shape[0], 3, 200, 200)

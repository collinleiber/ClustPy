import numpy as np
from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
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
    dataset = _helper_test_data_loader(load_video_weizmann, None, 77760, [10, 9],
                                       dataloader_params={"downloads_path": TEST_DOWNLOAD_PATH})  # N not always 5687
    # Non-flatten
    assert dataset.images.shape[1:] == (3, 144, 180)
    assert dataset.image_format == "CHW"
    # Change image size and downsample
    dataset = _helper_test_data_loader(load_video_weizmann, None, 30000, [10, 9],
                                       dataloader_params={"image_size": (100, 100), "frame_sampling_ratio": 0.5,
                                                          "downloads_path": TEST_DOWNLOAD_PATH})  # N not always 5687
    # Non-flatten
    assert dataset.images.shape[1:] == (3, 100, 100)
    assert dataset.image_format == "CHW"
    # Check downsampling
    data = dataset.data
    assert data.shape[0] / 5687 < 0.55 and data.shape[0] / 5687 > 0.49


@pytest.mark.largedata
@pytest.mark.data
def test_load_video_keck_gesture():
    dataset = _helper_test_data_loader(load_video_keck_gesture, None, 120000, [15, 4],
                                       dataloader_params={"subset": "all",
                                                          "downloads_path": TEST_DOWNLOAD_PATH})  # N not always 25457
    # Non-flatten
    assert dataset.images.shape[1:] == (3, 200, 200)
    assert dataset.image_format == "CHW"
    # Test data
    dataset = _helper_test_data_loader(load_video_keck_gesture, None, 120000, [15, 3],
                                       dataloader_params={"subset": "train",
                                                          "downloads_path": TEST_DOWNLOAD_PATH})  # N not always 11911
    # Non-flatten
    assert dataset.images.shape[1:] == (3, 200, 200)
    assert dataset.image_format == "CHW"
    # Train data and Change image size and downsample
    dataset = _helper_test_data_loader(load_video_keck_gesture, None, 30000, [15, 4],
                                       dataloader_params={"image_size": (100, 100), "frame_sampling_ratio": 0.5,
                                                          "subset": "test",
                                                          "downloads_path": TEST_DOWNLOAD_PATH})  # N not always 13546
    # Non-flatten
    assert dataset.images.shape[1:] == (3, 100, 100)
    assert dataset.image_format == "CHW"
    # Check downsampling
    data = dataset.data
    assert data.shape[0] / 11911 < 0.55 and data.shape[0] / 11911 > 0.49

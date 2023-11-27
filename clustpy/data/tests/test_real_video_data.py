from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader, _check_normalized_channels
from clustpy.data import load_video_weizmann
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
    data, labels = load_video_weizmann(downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 5687, 77760, [10, 9])
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_video_weizmann(flatten=False, downloads_path=TEST_DOWNLOAD_PATH)
    assert data.shape == (5687, 3, 144, 180)

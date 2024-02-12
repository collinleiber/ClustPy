from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_path_mnist, load_chest_mnist, load_derma_mnist, load_oct_mnist, load_pneumonia_mnist, \
    load_retina_mnist, load_breast_mnist, load_blood_mnist, load_tissue_mnist, load_organ_a_mnist, load_organ_c_mnist, \
    load_organ_s_mnist, load_organ_mnist_3d, load_nodule_mnist_3d, load_adrenal_mnist_3d, load_fracture_mnist_3d, \
    load_vessel_mnist_3d, load_synapse_mnist_3d
from pathlib import Path
import os
import shutil
import pytest

TEST_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_testfiles_medical_mnist")


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
def test_load_path_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_path_mnist, 107180, 2352, 9, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (107180, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_path_mnist, 89996, 2352, 9, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (89996, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Validation data set
    dataset = _helper_test_data_loader(load_path_mnist, 10004, 2352, 9, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10004, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_path_mnist, 7180, 2352, 9, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (7180, 3, 28, 28)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_chest_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_chest_mnist, 112120, 784, [2] * 14, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (112120, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_chest_mnist, 78468, 784, [2] * 14, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (78468, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_chest_mnist, 11219, 784, [2] * 14, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (11219, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_chest_mnist, 22433, 784, [2] * 14, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (22433, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_derma_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_derma_mnist, 10015, 2352, 7, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10015, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_derma_mnist, 7007, 2352, 7, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (7007, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Validation data set
    dataset = _helper_test_data_loader(load_derma_mnist, 1003, 2352, 7, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1003, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_derma_mnist, 2005, 2352, 7, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (2005, 3, 28, 28)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_oct_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_oct_mnist, 109309, 784, 4, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (109309, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_oct_mnist, 97477, 784, 4, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (97477, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_oct_mnist, 10832, 784, 4, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (10832, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_oct_mnist, 1000, 784, 4, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1000, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_pneumonia_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_pneumonia_mnist, 5856, 784, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (5856, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_pneumonia_mnist, 4708, 784, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (4708, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_pneumonia_mnist, 524, 784, 2, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (524, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_pneumonia_mnist, 624, 784, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (624, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_retina_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_retina_mnist, 1600, 2352, 5, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1600, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_retina_mnist, 1080, 2352, 5, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1080, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Validation data set
    dataset = _helper_test_data_loader(load_retina_mnist, 120, 2352, 5, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (120, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_retina_mnist, 400, 2352, 5, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (400, 3, 28, 28)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_breast_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_breast_mnist, 780, 784, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (780, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_breast_mnist, 546, 784, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (546, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_breast_mnist, 78, 784, 2, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (78, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_breast_mnist, 156, 784, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (156, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_blood_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_blood_mnist, 17092, 2352, 8, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (17092, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Train data set
    dataset = _helper_test_data_loader(load_blood_mnist, 11959, 2352, 8, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (11959, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Validation data set
    dataset = _helper_test_data_loader(load_blood_mnist, 1712, 2352, 8, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1712, 3, 28, 28)
    assert dataset.image_format == "CHW"
    # Test data set
    dataset = _helper_test_data_loader(load_blood_mnist, 3421, 2352, 8, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (3421, 3, 28, 28)
    assert dataset.image_format == "CHW"


@pytest.mark.data
def test_load_tissue_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_tissue_mnist, 236386, 784, 8, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (236386, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_tissue_mnist, 165466, 784, 8, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (165466, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_tissue_mnist, 23640, 784, 8, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (23640, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_tissue_mnist, 47280, 784, 8, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (47280, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_organ_a_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_organ_a_mnist, 58850, 784, 11, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (58850, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_organ_a_mnist, 34581, 784, 11, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (34581, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_organ_a_mnist, 6491, 784, 11, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (6491, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_organ_a_mnist, 17778, 784, 11, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (17778, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_organ_c_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_organ_c_mnist, 23660, 784, 11, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (23660, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_organ_c_mnist, 13000, 784, 11, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (13000, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_organ_c_mnist, 2392, 784, 11, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (2392, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_organ_c_mnist, 8268, 784, 11, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (8268, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_organ_s_mnist():
    # Full data set
    dataset = _helper_test_data_loader(load_organ_s_mnist, 25221, 784, 11, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (25221, 28, 28)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_organ_s_mnist, 13940, 784, 11, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (13940, 28, 28)
    assert dataset.image_format == "HW"
    # Validation data set
    dataset = _helper_test_data_loader(load_organ_s_mnist, 2452, 784, 11, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (2452, 28, 28)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_organ_s_mnist, 8829, 784, 11, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (8829, 28, 28)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_organ_mnist_3d():
    # Full data set
    dataset = _helper_test_data_loader(load_organ_mnist_3d, 1743, 21952, 11, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (1743, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Train data set
    dataset = _helper_test_data_loader(load_organ_mnist_3d, 972, 21952, 11, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (972, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Validation data set
    dataset = _helper_test_data_loader(load_organ_mnist_3d, 161, 21952, 11, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (161, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Test data set
    dataset = _helper_test_data_loader(load_organ_mnist_3d, 610, 21952, 11, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape == (610, 28, 28, 28)
    assert dataset.image_format == "HWD"


@pytest.mark.data
def test_load_nodule_mnist_3d():
    # Full data set
    dataset = _helper_test_data_loader(load_nodule_mnist_3d, 1633, 21952, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1633, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Train data set
    dataset = _helper_test_data_loader(load_nodule_mnist_3d, 1158, 21952, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1158, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Validation data set
    dataset = _helper_test_data_loader(load_nodule_mnist_3d, 165, 21952, 2, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (165, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Test data set
    dataset = _helper_test_data_loader(load_nodule_mnist_3d, 310, 21952, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (310, 28, 28, 28)
    assert dataset.image_format == "HWD"


@pytest.mark.data
def test_load_adrenal_mnist_3d():
    # Full data set
    dataset = _helper_test_data_loader(load_adrenal_mnist_3d, 1584, 21952, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1584, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Train data set
    dataset = _helper_test_data_loader(load_adrenal_mnist_3d, 1188, 21952, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1188, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Validation data set
    dataset = _helper_test_data_loader(load_adrenal_mnist_3d, 98, 21952, 2, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (98, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Test data set
    dataset = _helper_test_data_loader(load_adrenal_mnist_3d, 298, 21952, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (298, 28, 28, 28)
    assert dataset.image_format == "HWD"


@pytest.mark.data
def test_load_fracture_mnist_3d():
    # Full data set
    dataset = _helper_test_data_loader(load_fracture_mnist_3d, 1370, 21952, 3, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1370, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Train data set
    dataset = _helper_test_data_loader(load_fracture_mnist_3d, 1027, 21952, 3, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1027, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Validation data set
    dataset = _helper_test_data_loader(load_fracture_mnist_3d, 103, 21952, 3, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (103, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Test data set
    dataset = _helper_test_data_loader(load_fracture_mnist_3d, 240, 21952, 3, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (240, 28, 28, 28)
    assert dataset.image_format == "HWD"


@pytest.mark.data
def test_load_vessel_mnist_3d():
    # Full data set
    dataset = _helper_test_data_loader(load_vessel_mnist_3d, 1909, 21952, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1909, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Train data set
    dataset = _helper_test_data_loader(load_vessel_mnist_3d, 1335, 21952, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1335, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Validation data set
    dataset = _helper_test_data_loader(load_vessel_mnist_3d, 192, 21952, 2, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (192, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Test data set
    dataset = _helper_test_data_loader(load_vessel_mnist_3d, 382, 21952, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (382, 28, 28, 28)
    assert dataset.image_format == "HWD"


@pytest.mark.data
def test_load_synapse_mnist_3d():
    # Full data set
    dataset = _helper_test_data_loader(load_synapse_mnist_3d, 1759, 21952, 2, dataloader_params={"subset": "all", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1759, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Train data set
    dataset = _helper_test_data_loader(load_synapse_mnist_3d, 1230, 21952, 2, dataloader_params={"subset": "train", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (1230, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Validation data set
    dataset = _helper_test_data_loader(load_synapse_mnist_3d, 177, 21952, 2, dataloader_params={"subset": "val", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (177, 28, 28, 28)
    assert dataset.image_format == "HWD"
    # Test data set
    dataset = _helper_test_data_loader(load_synapse_mnist_3d, 352, 21952, 2, dataloader_params={"subset": "test", "downloads_path":TEST_DOWNLOAD_PATH})
    # Non-flatten
    assert dataset.images.shape ==  (352, 28, 28, 28)
    assert dataset.image_format == "HWD"

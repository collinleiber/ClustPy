from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader, _check_normalized_channels
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
    data, labels = load_path_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 107180, 2352, 9)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_path_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 89996, 2352, 9)
    _check_normalized_channels(data, 3, False)
    # Validation data set
    data, labels = load_path_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10004, 2352, 9)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_path_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 7180, 2352, 9)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_path_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (107180, 3, 28, 28)


@pytest.mark.data
def test_load_chest_mnist():
    # Full data set
    data, labels = load_chest_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 112120, 784, [2] * 14)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_chest_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 78468, 784, [2] * 14)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_chest_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 11219, 784, [2] * 14)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_chest_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 22433, 784, [2] * 14)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_chest_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (112120, 28, 28)


@pytest.mark.data
def test_load_derma_mnist():
    # Full data set
    data, labels = load_derma_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 10015, 2352, 7)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_derma_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 7007, 2352, 7)
    _check_normalized_channels(data, 3, False)
    # Validation data set
    data, labels = load_derma_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1003, 2352, 7)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_derma_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2005, 2352, 7)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_derma_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (10015, 3, 28, 28)


@pytest.mark.data
def test_load_oct_mnist():
    # Full data set
    data, labels = load_oct_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 109309, 784, 4)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_oct_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 97477, 784, 4)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_oct_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 10832, 784, 4)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_oct_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1000, 784, 4)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_oct_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (109309, 28, 28)


@pytest.mark.data
def test_load_pneumonia_mnist():
    # Full data set
    data, labels = load_pneumonia_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 5856, 784, 2)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_pneumonia_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 4708, 784, 2)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_pneumonia_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 524, 784, 2)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_pneumonia_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 624, 784, 2)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_pneumonia_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (5856, 28, 28)


@pytest.mark.data
def test_load_retina_mnist():
    # Full data set
    data, labels = load_retina_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1600, 2352, 5)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_retina_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 1080, 2352, 5)
    _check_normalized_channels(data, 3, False)
    # Validation data set
    data, labels = load_retina_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 120, 2352, 5)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_retina_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 400, 2352, 5)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_retina_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1600, 3, 28, 28)


@pytest.mark.data
def test_load_breast_mnist():
    # Full data set
    data, labels = load_breast_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 780, 784, 2)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_breast_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 546, 784, 2)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_breast_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 78, 784, 2)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_breast_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 156, 784, 2)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_breast_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (780, 28, 28)


@pytest.mark.data
def test_load_blood_mnist():
    # Full data set
    data, labels = load_blood_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 17092, 2352, 8)
    _check_normalized_channels(data, 3, True)
    # Train data set
    data, labels = load_blood_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 11959, 2352, 8)
    _check_normalized_channels(data, 3, False)
    # Validation data set
    data, labels = load_blood_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 1712, 2352, 8)
    _check_normalized_channels(data, 3, False)
    # Test data set
    data, labels = load_blood_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 3421, 2352, 8)
    _check_normalized_channels(data, 3, False)
    # Test non-flatten
    data, _ = load_blood_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (17092, 3, 28, 28)


@pytest.mark.data
def test_load_tissue_mnist():
    # Full data set
    data, labels = load_tissue_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 236386, 784, 8)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_tissue_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 165466, 784, 8)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_tissue_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 23640, 784, 8)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_tissue_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 47280, 784, 8)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_tissue_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (236386, 28, 28)


@pytest.mark.data
def test_load_organ_a_mnist():
    # Full data set
    data, labels = load_organ_a_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 58850, 784, 11)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_organ_a_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 34581, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_organ_a_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 6491, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_organ_a_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 17778, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_organ_a_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (58850, 28, 28)


@pytest.mark.data
def test_load_organ_c_mnist():
    # Full data set
    data, labels = load_organ_c_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 23660, 784, 11)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_organ_c_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 13000, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_organ_c_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2392, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_organ_c_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 8268, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_organ_c_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (23660, 28, 28)


@pytest.mark.data
def test_load_organ_s_mnist():
    # Full data set
    data, labels = load_organ_s_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 25221, 784, 11)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_organ_s_mnist("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 13940, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_organ_s_mnist("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 2452, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_organ_s_mnist("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 8829, 784, 11)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_organ_s_mnist("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (25221, 28, 28)


@pytest.mark.data
def test_load_organ_mnist_3d():
    # Full data set
    data, labels = load_organ_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1743, 21952, 11)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_organ_mnist_3d("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 972, 21952, 11)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_organ_mnist_3d("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 161, 21952, 11)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_organ_mnist_3d("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 610, 21952, 11)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_organ_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1743, 28, 28, 28)


@pytest.mark.data
def test_load_nodule_mnist_3d():
    # Full data set
    data, labels = load_nodule_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1633, 21952, 2)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_nodule_mnist_3d("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 1158, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_nodule_mnist_3d("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 165, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_nodule_mnist_3d("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 310, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_nodule_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1633, 28, 28, 28)


@pytest.mark.data
def test_load_adrenal_mnist_3d():
    # Full data set
    data, labels = load_adrenal_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1584, 21952, 2)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_adrenal_mnist_3d("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 1188, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_adrenal_mnist_3d("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 98, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_adrenal_mnist_3d("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 298, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_adrenal_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1584, 28, 28, 28)


@pytest.mark.data
def test_load_fracture_mnist_3d():
    # Full data set
    data, labels = load_fracture_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1370, 21952, 3)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_fracture_mnist_3d("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 1027, 21952, 3)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_fracture_mnist_3d("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 103, 21952, 3)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_fracture_mnist_3d("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 240, 21952, 3)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_fracture_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1370, 28, 28, 28)


@pytest.mark.data
def test_load_vessel_mnist_3d():
    # Full data set
    data, labels = load_vessel_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1909, 21952, 2)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_vessel_mnist_3d("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 1335, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_vessel_mnist_3d("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 192, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_vessel_mnist_3d("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 382, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_vessel_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1909, 28, 28, 28)


@pytest.mark.data
def test_load_synapse_mnist_3d():
    # Full data set
    data, labels = load_synapse_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=True)
    _helper_test_data_loader(data, labels, 1759, 21952, 2)
    _check_normalized_channels(data, 1, True)
    # Train data set
    data, labels = load_synapse_mnist_3d("train", downloads_path=TEST_DOWNLOAD_PATH, normalize_channels=False)
    _helper_test_data_loader(data, labels, 1230, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Validation data set
    data, labels = load_synapse_mnist_3d("val", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 177, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test data set
    data, labels = load_synapse_mnist_3d("test", downloads_path=TEST_DOWNLOAD_PATH)
    _helper_test_data_loader(data, labels, 352, 21952, 2)
    _check_normalized_channels(data, 1, False)
    # Test non-flatten
    data, _ = load_synapse_mnist_3d("all", downloads_path=TEST_DOWNLOAD_PATH, flatten=False)
    assert data.shape == (1759, 28, 28, 28)

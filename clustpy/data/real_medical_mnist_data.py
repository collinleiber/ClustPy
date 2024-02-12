import numpy as np
from clustpy.data._utils import _get_download_dir, _download_file, flatten_images
import os
from sklearn.datasets._base import Bunch


def _load_medical_mnist_data(dataset_name: str, subset: str, colored: bool, multiple_labelings: bool,
                             return_X_y: bool, downloads_path: str) -> Bunch:
    """
    Helper function to load medical MNIST data from https://medmnist.com/.

    Parameters
    ----------
    dataset_name : str
        name of the data set
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data
    colored : bool
        specifies if the images in the dataset are grayscale or colored
    multiple_labelings : bool
        specifies if the data set contains multiple labelings (for alternative clusterings)
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object
    downloads_path : str
        path to the directory where the data is stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array and the labels numpy array
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test", "val"], "subset must match 'all', 'train', 'test' or 'val'. Your input {0}".format(subset)
    # Check if data exists
    filename = _get_download_dir(downloads_path) + "/" + dataset_name + ".npz"
    if not os.path.isfile(filename):
        _download_file("https://zenodo.org/record/6496656/files/" + dataset_name + ".npz?download=1", filename)
    # Load data
    dataset = np.load(filename)
    if subset == "all" or subset == "train":
        data = dataset["train_images"]
        labels = dataset["train_labels"]
    if subset == "all" or subset == "test":
        test_data = dataset["test_images"]
        test_labels = dataset["test_labels"]
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    if subset == "all" or subset == "val":
        val_data = dataset["val_images"]
        val_labels = dataset["val_labels"]
        if subset == "all":
            data = np.r_[data, val_data]
            labels = np.r_[labels, val_labels]
        else:
            data = val_data
            labels = val_labels
    dataset = None  # is needed so that the test folder can be deleted after the unit tests have finished
    # Get format of image
    if data.ndim == 3:
        image_format = "HW"
    elif data.ndim == 4:
        image_format = "HWD" if not colored else "HWC"
    else:  # data.ndim must be 5
        image_format = "HWDC"
    # Flatten data
    data_flatten = flatten_images(data, image_format)
    # Sometimes the labels are contained in a separate dimension
    if labels.ndim != 1 and not multiple_labelings:
        assert labels.shape[1] == 1, "Data should only contain a single labeling"
        labels = labels[:, 0]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Return values
    if return_X_y:
        return data_flatten, labels
    else:
        # Get images in correct format
        if colored:
            # Change to CHW format
            if data.ndim == 4:
                data_image = np.transpose(data, [0, 3, 1, 2])
                image_format = "CHW"
            else:
                data_image = np.transpose(data, [0, 4, 1, 2, 3])
                image_format = "CHWD"
        else:
            data_image = data
        return Bunch(dataset_name=dataset_name, data=data_flatten, target=labels, images=data_image,
                     image_format=image_format)


"""
Actual datasets
"""


def load_path_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the PathMNIST data set. It consists of 107180 28x28 colored images belonging to one of 9 classes.
    The data set is composed of 89996 training, 10004 validation and 7180 test samples.
    N=107180, d=2352, k=9.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (107180 x 2352), the labels numpy array (107180)

    References
    -------
    https://medmnist.com/

    Jakob Nikolas Kather, Johannes Krisam, et al., "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study,"
    PLOS Medicine, vol. 16, no. 1, pp. 1–22, 01 2019.
    """
    return _load_medical_mnist_data("pathmnist", subset, True, False, return_X_y, downloads_path)


def load_chest_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the ChestMNIST data set. It consists of 112120 28x28 grayscale images.
    The ground truth labels consist of 14 labelings with 2 clusters each.
    The data set is composed of 78468 training, 11219 validation and 22433 test samples.
    N=112120, d=784, k=[2,2,2,2,2,2,2,2,2,2,2,2,2,2].

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (112120 x 784), the labels numpy array (112120)

    References
    -------
    https://medmnist.com/

    Xiaosong Wang, Yifan Peng, et al., "Chest x-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,"
    in CVPR, 2017, pp. 3462–3471.
    """
    return _load_medical_mnist_data("chestmnist", subset, False, True, return_X_y, downloads_path)


def load_derma_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the DermaMNIST data set. It consists of 10015 28x28 colored images belonging to one of 7 classes.
    The data set is composed of 7007 training, 1003 validation and 2005 test samples.
    N=10015, d=2352, k=7.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (10015 x 2352), the labels numpy array (10015)

    References
    -------
    https://medmnist.com/

    Philipp Tschandl, Cliff Rosendahl, et al., "The ham10000 dataset, a large collection of multisource dermatoscopic images of common pigmented skin lesions,"
    Scientific data, vol. 5, pp. 180161, 2018.

    Noel Codella, Veronica Rotemberg, et al., “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”,
    2018, arXiv:1902.03368.
    """
    return _load_medical_mnist_data("dermamnist", subset, True, False, return_X_y, downloads_path)


def load_oct_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the OCTMNIST data set. It consists of 109309 28x28 grayscale images belonging to one of 4 classes.
    The data set is composed of 97477 training, 10832 validation and 1000 test samples.
    N=109309, d=784, k=4.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (109309 x 784), the labels numpy array (109309)

    References
    -------
    https://medmnist.com/

    Daniel S. Kermany, Michael Goldbaum, et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning,"
    Cell, vol. 172, no. 5, pp. 1122 – 1131.e9, 2018.
    """
    return _load_medical_mnist_data("octmnist", subset, False, False, return_X_y, downloads_path)


def load_pneumonia_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the PneumoniaMNIST data set. It consists of 5856 28x28 grayscale images belonging to one of 2 classes.
    The data set is composed of 4708 training, 524 validation and 624 test samples.
    N=5856, d=784, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (5856 x 784), the labels numpy array (5856)

    References
    -------
    https://medmnist.com/

    Daniel S. Kermany, Michael Goldbaum, et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning,"
    Cell, vol. 172, no. 5, pp. 1122 – 1131.e9, 2018.
    """
    return _load_medical_mnist_data("pneumoniamnist", subset, False, False, return_X_y, downloads_path)


def load_retina_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the RetinaMNIST data set. It consists of 1600 28x28 colored images belonging to one of 5 classes.
    The data set is composed of 1080 training, 120 validation and 400 test samples.
    N=1600, d=2352, k=5.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1600 x 2352), the labels numpy array (1600)

    References
    -------
    https://medmnist.com/

    DeepDR Diabetic Retinopathy Image Dataset (DeepDRiD), "The 2nd diabetic retinopathy grading and image quality estimation challenge,"
    https://isbi.deepdr.org/data.html, 2020.
    """
    return _load_medical_mnist_data("retinamnist", subset, True, False, return_X_y, downloads_path)


def load_breast_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the BreastMNIST data set. It consists of 780 28x28 grayscale images belonging to one of 2 classes.
    The data set is composed of 546 training, 78 validation and 156 test samples.
    N=780, d=784, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (780 x 784), the labels numpy array (780)

    References
    -------
    https://medmnist.com/

    Walid Al-Dhabyani, Mohammed Gomaa, et al., "Dataset of breast ultrasound images,"
    Data in Brief, vol. 28, pp. 104863, 2020.
    """
    return _load_medical_mnist_data("breastmnist", subset, False, False, return_X_y, downloads_path)


def load_blood_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the BloodMNIST data set. It consists of 17092 28x28 colored images belonging to one of 8 classes.
    The data set is composed of 11959 training, 1712 validation and 3421 test samples.
    N=17092, d=2352, k=8.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (17092 x 2352), the labels numpy array (17092)

    References
    -------
    https://medmnist.com/

    Andrea Acevedo, Anna Merino, et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems,"
    Data in Brief, vol. 30, pp. 105474, 2020.
    """
    return _load_medical_mnist_data("bloodmnist", subset, True, False, return_X_y, downloads_path)


def load_tissue_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the TissueMNIST data set. It consists of 236386 28x28 grayscale images belonging to one of 8 classes.
    The data set is composed of 165466 training, 23640 validation and 47280 test samples.
    N=236386, d=784, k=8.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (236386 x 784), the labels numpy array (236386)

    References
    -------
    https://medmnist.com/

    Vebjorn Ljosa, Katherine L Sokolnicki, et al., “Annotated high-throughput microscopy imagesets for validation.,”
    Nature methods, vol. 9, no. 7, pp.637–637, 2012.
    """
    return _load_medical_mnist_data("tissuemnist", subset, False, False, return_X_y, downloads_path)


def load_organ_a_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the OrganAMNIST data set. It consists of 58850 28x28 grayscale images belonging to one of 11 classes.
    The data set is composed of 34581 training, 6491 validation and 17778 test samples.
    N=58850, d=784, k=11.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (58850 x 784), the labels numpy array (58850)

    References
    -------
    https://medmnist.com/

    Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
    arXiv preprint arXiv:1901.04056, 2019.

    Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region proposal network,"
    IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.
    """
    return _load_medical_mnist_data("organamnist", subset, False, False, return_X_y, downloads_path)


def load_organ_c_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the OrganCMNIST data set. It consists of 23660 28x28 grayscale images belonging to one of 11 classes.
    The data set is composed of 13000 training, 2392 validation and 8268 test samples.
    N=23660, d=784, k=11.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (23660 x 784), the labels numpy array (23660)

    References
    -------
    https://medmnist.com/

    Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
    arXiv preprint arXiv:1901.04056, 2019.

    Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region proposal network,"
    IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.
    """
    return _load_medical_mnist_data("organcmnist", subset, False, False, return_X_y, downloads_path)


def load_organ_s_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the OrganSMNIST data set. It consists of 25221 28x28 grayscale images belonging to one of 11 classes.
    The data set is composed of 13940 training, 2452 validation and 8829 test samples.
    N=25221, d=784, k=11.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (25221 x 784), the labels numpy array (25221)

    References
    -------
    https://medmnist.com/

    Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
    arXiv preprint arXiv:1901.04056, 2019.

    Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region proposal network,"
    IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.
    """
    return _load_medical_mnist_data("organsmnist", subset, False, False, return_X_y, downloads_path)


def load_organ_mnist_3d(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the OrganMNIST3D data set. It consists of 1743 28x28x28 grayscale images belonging to one of 11 classes.
    The data set is composed of 972 training, 161 validation and 610 test samples.
    N=1743, d=21952, k=11.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1743 x 21952), the labels numpy array (1743)

    References
    -------
    https://medmnist.com/

    Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
    arXiv preprint arXiv:1901.04056, 2019.

    Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region proposal network,"
    IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.
    """
    return _load_medical_mnist_data("organmnist3d", subset, False, False, return_X_y, downloads_path)


def load_nodule_mnist_3d(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the NoduleMNIST3D data set. It consists of 1633 28x28x28 grayscale images belonging to one of 2 classes.
    The data set is composed of 1158 training, 165 validation and 310 test samples.
    N=1633, d=21952, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1633 x 21952), the labels numpy array (1633)

    References
    -------
    https://medmnist.com/

    Samuel G. Armato III, Geoffrey McLennan, et al., “The lung image database consortium (lidc) and image database resource initiative (idri): A completed reference databaseof lung nodules on ct scans,”
    Medical Physics, vol. 38,no. 2, pp. 915–931, 2011.
    """
    return _load_medical_mnist_data("nodulemnist3d", subset, False, False, return_X_y, downloads_path)


def load_adrenal_mnist_3d(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the AdrenalMNIST3D data set. It consists of 1584 28x28x28 grayscale images belonging to one of 2 classes.
    The data set is composed of 1188 training, 98 validation and 298 test samples.
    N=1584, d=21952, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1584 x 21952), the labels numpy array (1584)

    References
    -------
    https://medmnist.com/
    """
    return _load_medical_mnist_data("adrenalmnist3d", subset, False, False, return_X_y, downloads_path)


def load_fracture_mnist_3d(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the FractureMNIST3D data set. It consists of 1370 28x28x28 grayscale images belonging to one of 3 classes.
    The data set is composed of 1027 training, 103 validation and 240 test samples.
    N=1370, d=21952, k=3.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1370 x 21952), the labels numpy array (1370)

    References
    -------
    https://medmnist.com/

    Liang Jin, Jiancheng Yang, et al., “Deep-learning-assisted detection and segmentation of rib fractures from ct scans: Development and validation of fracnet,”
    EBioMedicine, vol. 62, pp. 103106, 2020.
    """
    return _load_medical_mnist_data("fracturemnist3d", subset, False, False, return_X_y, downloads_path)


def load_vessel_mnist_3d(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the VesselMNIST3D data set. It consists of 1909 28x28x28 grayscale images belonging to one of 2 classes.
    The data set is composed of 1335 training, 192 validation and 382 test samples.
    N=1909, d=21952, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1909 x 21952), the labels numpy array (1909)

    References
    -------
    https://medmnist.com/

    Xi Yang, Ding Xia, et al., “Intra: 3d intracranial aneurysm dataset for deep learning,”
    in Proceedings of the IEEE/CVF Conference onComputer Vision and Pattern Recognition (CVPR), June 2020.
    """
    return _load_medical_mnist_data("vesselmnist3d", subset, False, False, return_X_y, downloads_path)


def load_synapse_mnist_3d(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the SynapseMNIST3D data set. It consists of 1759 28x28x28 grayscale images belonging to one of 2 classes.
    The data set is composed of 1230 training, 177 validation and 352 test samples.
    N=1759, d=21952, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1759 x 21952), the labels numpy array (1759)

    References
    -------
    https://medmnist.com/
    """
    return _load_medical_mnist_data("synapsemnist3d", subset, False, False, return_X_y, downloads_path)

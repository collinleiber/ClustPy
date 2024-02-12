import numpy as np
from clustpy.data._utils import _get_download_dir, _download_file
from sklearn.datasets._base import Bunch
import os
import zipfile


def _load_timeseries_classification_data(dataset_name: str, subset: str, labels_minus_one: bool, file_type: str,
                                         last_column_are_labels: bool, return_X_y: bool, downloads_path: str) -> Bunch:
    """
    Helper function to load timeseries data from www.timeseriesclassification.com.

    Parameters
    ----------
    dataset_name : str
        name of the data set
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data
    labels_minus_one : bool
        Convert labels from 1,... to 0,...
    file_type : str
        file type within the zip file. Currently supported are "txt" and "ts". Is usually "txt"
    last_column_are_labels : bool
        specifies if the last column contains the labels. If false labels should be contained in the first column
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object
    downloads_path : str
        path to the directory where the data is stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array, the labels numpy array
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    directory = _get_download_dir(downloads_path) + "/" + dataset_name + "/"
    filename = directory + dataset_name + ".zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.timeseriesclassification.com/aeon-toolkit/" + dataset_name + ".zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # Load data and labels
    if subset == "all" or subset == "train":
        # Normally we have txt files
        if file_type == "txt":
            dataset = np.genfromtxt(directory + dataset_name + "_TRAIN.txt")
        elif file_type == "ts":
            # Ts files must be changed first
            with open(directory + dataset_name + "_TRAIN.ts", "rb") as f:
                clean_lines = (line.replace(b":", b",").replace(b"@", b"#") for line in f)
                dataset = np.genfromtxt(clean_lines, delimiter=",", comments="#")
        # Are labels in first or last column?
        if last_column_are_labels:
            data = dataset[:, :-1]
            labels = dataset[:, -1]
        else:
            data = dataset[:, 1:]
            labels = dataset[:, 0]
    if subset == "all" or subset == "test":
        # Normally we have txt files
        if file_type == "txt":
            test_dataset = np.genfromtxt(directory + dataset_name + "_TEST.txt")
        elif file_type == "ts":
            # Ts files must be changed first
            with open(directory + dataset_name + "_TEST.ts", "rb") as f:
                clean_lines = (line.replace(b":", b",").replace(b"@", b"#") for line in f)
                test_dataset = np.genfromtxt(clean_lines, delimiter=",", comments="#")
        # Are labels in first or last column?
        if last_column_are_labels:
            if subset == "all":
                data = np.r_[data, test_dataset[:, :-1]]
                labels = np.r_[labels, test_dataset[:, -1]]
            else:
                data = test_dataset[:, :-1]
                labels = test_dataset[:, -1]
        else:
            if subset == "all":
                data = np.r_[data, test_dataset[:, 1:]]
                labels = np.r_[labels, test_dataset[:, 0]]
            else:
                data = test_dataset[:, 1:]
                labels = test_dataset[:, 0]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    if labels_minus_one:
        # Convert labels from 1,... to 0,...
        labels -= 1
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name=dataset_name, data=data, target=labels)


def load_motestrain(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the motestrain data set. It consists of 1272 samples belonging to one of 2 classes.
    The data set is composed of 20 training and 1252 test samples.
    N=1272, d=84, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1272 x 84), the labels numpy array (1272)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=MoteStrain
    """
    return _load_timeseries_classification_data("MoteStrain", subset, True, "txt", False, return_X_y, downloads_path)


def load_proximal_phalanx_outline(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the proximal phalanx outline data set. It consists of 876 samples belonging to one of 2 classes.
    The data set is composed of 600 training and 276 test samples.
    N=876, d=80, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (876 x 80), the labels numpy array (876)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=ProximalPhalanxOutlineCorrect
    """
    return _load_timeseries_classification_data("DistalPhalanxOutlineCorrect", subset, False, "txt", False,
                                                return_X_y, downloads_path)


def load_diatom_size_reduction(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the diatom size reduction data set. It consists of 322 samples belonging to one of 4 classes.
    The data set is composed of 16 training and 306 test samples.
    N=322, d=345, k=4.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (322 x 345), the labels numpy array (322)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=DiatomSizeReduction
    """
    return _load_timeseries_classification_data("DiatomSizeReduction", subset, True, "txt", False,
                                                return_X_y, downloads_path)


def load_symbols(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the symbols data set. It consists of 1020 samples belonging to one of 6 classes.
    The data set is composed of 25 training and 995 test samples.
    N=1020, d=398, k=6.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1020 x 398), the labels numpy array (1020)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=Symbols
    """
    return _load_timeseries_classification_data("Symbols", subset, True, "txt", False, return_X_y, downloads_path)


def load_olive_oil(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the OliveOil data set. It consists of 60 samples belonging to one of 4 classes.
    The data set is composed of 30 training and 30 test samples.
    N=60, d=570, k=4.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (60 x 570), the labels numpy array (60)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=OliveOil
    """
    return _load_timeseries_classification_data("OliveOil", subset, True, "txt", False, return_X_y, downloads_path)


def load_plane(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the plane data set. It consists of 210 samples belonging to one of 7 classes.
    The data set is composed of 105 training and 105 test samples.
    N=210, d=144, k=7.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (210 x 144), the labels numpy array (210)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=Plane
    """
    return _load_timeseries_classification_data("Plane", subset, True, "txt", False, return_X_y, downloads_path)


def load_sony_aibo_robot_surface(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the Sony AIBO Robot Surface 1 data set. It consists of 621 samples belonging to one of 2 classes.
    The data set is composed of 20 training and 601 test samples.
    N=621, d=70, k=2.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (621 x 70), the labels numpy array (621)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=SonyAIBORobotSurface1
    """
    return _load_timeseries_classification_data("SonyAIBORobotSurface1", subset, True, "txt", False,
                                                return_X_y, downloads_path)


def load_two_patterns(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the two patterns data set. It consists of 5000 samples belonging to one of 4 classes.
    The data set is composed of 1000 training and 4000 test samples.
    N=5000, d=128, k=4.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (5000 x 128), the labels numpy array (5000)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=TwoPatterns
    """
    return _load_timeseries_classification_data("TwoPatterns", subset, True, "txt", False, return_X_y, downloads_path)


def load_lsst(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the LSST data set. It consists of 4925 samples belonging to one of 14 classes.
    The data set is composed of 2459 training and 2466 test samples.
    N=4925, d=216, k=14.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (4925 x 216), the labels numpy array (4925)

    References
    -------
    http://www.timeseriesclassification.com/description.php?Dataset=LSST
    """
    dataset = _load_timeseries_classification_data("LSST", subset, True, "ts", True, False, downloads_path)
    data = dataset.data
    labels = dataset.target
    # Current labels are: 5, 14, 15, 41, 51, 52, ... -> change to: 0, 1, 2, 3, 4, ...
    for i, l in enumerate(np.unique(labels)):
        labels[labels == l] = i
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="LSST", data=data, target=labels)

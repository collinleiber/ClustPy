try:
    from PIL import Image
except:
    print(
        "[WARNING] Could not import PIL in clustpy.data.real_world_data. Please install PIL by 'pip install Pillow' if necessary")
from clustpy.data._utils import _download_file, _get_download_dir, _decompress_z_file, _load_data_file, flatten_images
import os
import numpy as np
import zipfile
import tarfile
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.datasets._base import Bunch


def load_banknotes(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the banknote authentication data set. It consists of 1372 genuine and forged banknote samples.
    N=1372, d=4, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1372 x 4), the labels numpy array (1372)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    filename = _get_download_dir(downloads_path) + "/data_banknote_authentication.txt"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Banknotes", data=data, target=labels)


def load_spambase(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the spambase data set. It consists of 4601 spam and non-spam mails.
    N=4601, d=57, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (4601 x 57), the labels numpy array (4601)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/spambase
    """
    filename = _get_download_dir(downloads_path) + "/spambase.data"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data")
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Spambase", data=data, target=labels)


def load_seeds(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the seeds data set. It consists of 210 samples belonging to one of three varieties of wheat.
    N=210, d=7, k=3.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (210 x 7), the labels numpy array (210)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/seeds
    """
    filename = _get_download_dir(downloads_path) + "/seeds_dataset.txt"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
                                   delimiter=None)
    # Convert labels from 1,... to 0,...
    labels -= 1
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Seeds", data=data, target=labels)


def load_skin(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the Skin Segmentation data set. It consists of 245057 skin- and non-skin samples with their B, G, R color
    information.
    N=245057, d=3, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (245057 x 3), the labels numpy array (245057)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/skin+segmentation
    """
    filename = _get_download_dir(downloads_path) + "/Skin_NonSkin.txt"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt",
                                   delimiter=None)
    # Convert labels from 1,... to 0,...
    labels -= 1
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="SkinSegmentation", data=data, target=labels)


def load_soybean_small(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the small version of the soybean data set. It is a small subset of the original soybean data set.
    It consists of 47 samples belonging to one of 4 classes.
    N=47, d=35, k=4.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (47 x 35), the labels numpy array (47)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/soybean+(small)
    """
    filename = _get_download_dir(downloads_path) + "/soybean-small.data"
    if not os.path.isfile(filename):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
            filename)
    # Load data and labels
    df = pd.read_csv(filename, delimiter=",", header=None)
    labels_raw = df.iloc[:, -1]
    data = df.iloc[:, :-1].values
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="SoybeanSmall", data=data, target=labels)


def load_soybean_large(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the large version of the soybean data set. It consists of 562 samples belonging to one of 15 classes.
    Originally, the data set would have samples and 19 classes but some samples have attributes showing '?' values. Those
    will be ignored.
    The data set is composed of 266 training and 296 test samples.
    N=562, d=35, k=15.

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
        the data numpy array (562 x 35), the labels numpy array (562)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/soybean+(Large)
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    if subset == "all" or subset == "train":
        filename = _get_download_dir(downloads_path) + "/soybean-large.data"
        if not os.path.isfile(filename):
            _download_file(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data",
                filename)
        # Load data and labels
        df_train = pd.read_csv(filename, delimiter=",", header=None)
        df_train = df_train[(df_train != '?').all(axis=1)]
        labels_raw = df_train.pop(0)
        data = df_train.values
    if subset == "all" or subset == "test":
        filename = _get_download_dir(downloads_path) + "/soybean-large.test"
        if not os.path.isfile(filename):
            _download_file(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.test",
                filename)
        df_test = pd.read_csv(filename, delimiter=",", header=None)
        df_test = df_test[(df_test != '?').all(axis=1)]
        labels_test = df_test.pop(0)
        if subset == "all":
            data = np.r_[data, df_test.values]
            labels_raw = np.r_[labels_raw, labels_test]
        else:
            data = df_test.values
            labels_raw = labels_test
    # Transform data to numerical array
    data = np.array(data, dtype=int)
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="SoybeanLarge", data=data, target=labels)


def load_pendigits(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the pendigits data set. It consists of 10992 vectors of length 16, representing 8 coordinates. The coordinates
    were taken from the task of writing digits (0 to 9) on a tablet.
    The data set is composed of 7494 training and 3498 test samples.
    N=10992, d=16, k=10.

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
        the data numpy array (10992 x 16), the labels numpy array (10992)

    References
    -------
    http://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    if subset == "all" or subset == "train":
        filename = _get_download_dir(downloads_path) + "/pendigits.tra"
        data, labels = _load_data_file(filename,
                                       "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra")
    if subset == "all" or subset == "test":
        filename = _get_download_dir(downloads_path) + "/pendigits.tes"
        test_data, test_labels = _load_data_file(filename,
                                                 "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes")
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Pendigits", data=data, target=labels)


def load_ecoli(ignore_small_clusters: bool = False, return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the ecoli data set. It consists of 336 samples belonging to one of 8 classes.
    N=336, d=7, k=8.

    Parameters
    ----------
    ignore_small_clusters : bool
        specify if the three small clusters with size 2, 2 and 5 should be ignored (default: False)
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (336 x 7), the labels numpy array (336)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/ecoli
    """
    filename = _get_download_dir(downloads_path) + "/ecoli.data"
    if not os.path.isfile(filename):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
            filename)
    data = np.zeros((336, 7))
    labels_raw = []
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            splited = line.split()
            data[i] = splited[1:-1]
            labels_raw.append(splited[-1])
    if ignore_small_clusters:
        # Optional: Remove the three small clusters consisting of only 2, 2 and 5 samples
        keep_labels = [l not in ["imL", "imS", "omL"] for l in labels_raw]
        data = data[keep_labels]
        labels_raw = [l for i, l in enumerate(labels_raw) if keep_labels[i]]
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Ecoli", data=data, target=labels)


def load_htru2(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the HTRU2 data set. It consists of 17898 samples belonging to the pulsar or non-pulsar class.
    A special property is that more than 90% of the data belongs to class 0.
    N=17898, d=8, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (17898 x 8), the labels numpy array (17898)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/HTRU2
    """
    directory = _get_download_dir(downloads_path) + "/htru2/"
    filename = directory + "HTRU2.zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # Load data and labels
    dataset = np.genfromtxt(directory + "HTRU_2.csv", delimiter=",")
    data = dataset[:, :-1]
    labels = dataset[:, -1]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="HTRU2", data=data, target=labels)


def load_letterrecognition(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the Letter Recognition data set. It consists of 20000 samples where each sample represents one of the 26 capital
    letters in the English alphabet. All samples are composed of 16 numerical stimuli describing the respective letter.
    N=20000, d=16, k=26.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (20000 x 16), the labels numpy array (20000)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/letter+recognition
    """
    filename = _get_download_dir(downloads_path) + "/letter-recognition.data"
    if not os.path.isfile(filename):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
            filename)
    # Transform letters to integers
    letter_mappings = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5", "G": "6", "H": "7", "I": "8",
                       "J": "9", "K": "10", "L": "11", "M": "12", "N": "13", "O": "14", "P": "15", "Q": "16",
                       "R": "17", "S": "18", "T": "19", "U": "20", "V": "21", "W": "22", "X": "23", "Y": "24",
                       "Z": "25"}
    with open(filename, "r") as f:
        file_text = f.read()
    file_text = file_text.replace("\n", ",")
    for k in letter_mappings.keys():
        file_text = file_text.replace(k, letter_mappings[k])
    # Create numpy array
    datafile = np.fromstring(file_text, sep=",").reshape(-1, 17)
    data = datafile[:, 1:]
    labels = datafile[:, 0]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Letterrecognition", data=data, target=labels)


def load_har(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the Human Activity Recognition data set. It consists of 10299 samples each representing sensor data of a person
    performing an activity. The six activities are walking, walking_upstairs, walking_downstairs, sitting, standing and
    laying.
    The data set is composed of 7352 training and 2947 test samples.
    N=10992, d=561, k=6.

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
        the data numpy array (10992 x 561), the labels numpy array (10992)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    directory = _get_download_dir(downloads_path) + "/har/"
    filename = directory + "UCI HAR Dataset.zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # Load data and labels
    if subset == "all" or subset == "train":
        data = np.genfromtxt(directory + "UCI HAR Dataset/train/X_train.txt")
        labels = np.genfromtxt(directory + "UCI HAR Dataset/train/y_train.txt")
    if subset == "all" or subset == "test":
        test_data = np.genfromtxt(directory + "UCI HAR Dataset/test/X_test.txt")
        test_labels = np.genfromtxt(directory + "UCI HAR Dataset/test/y_test.txt")
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Convert labels from 1,... to 0,...
    labels = labels - 1
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="HAR", data=data, target=labels)


def load_statlog_shuttle(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the statlog shuttle data set. It consists of 58000 samples belonging to one of 7 classes. A special property is
    that about 80% of the data belongs to class 0.
    The data set is composed of 43500 training and 14500 test samples.
    N=58000, d=9, k=7.

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
        the data numpy array (58000 x 9), the labels numpy array (58000)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    directory = _get_download_dir(downloads_path) + "/shuttle/"
    if subset == "all" or subset == "train":
        filename = directory + "shuttle.trn.Z"
        if not os.path.isfile(filename):
            if not os.path.isdir(directory):
                os.mkdir(directory)
            _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z",
                           filename)
            # Unpack z-file
            success = _decompress_z_file(filename, directory)
            if not success:
                os.remove(filename)
                return (None, None) if return_X_y else None
        # Load data and labels
        dataset = np.genfromtxt(directory + "shuttle.trn")
        data = dataset[:, :-1]
        labels = dataset[:, -1]
    if subset == "all" or subset == "test":
        filename = directory + "shuttle.tst"
        if not os.path.isfile(filename):
            _download_file(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst",
                filename)
        test_dataset = np.genfromtxt(directory + "shuttle.tst")
        test_data = test_dataset[:, :-1]
        test_labels = test_dataset[:, -1]
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Convert labels from 1,... to 0,...
    labels -= 1
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="StatlogShuttle", data=data, target=labels)


def load_mice_protein(return_additional_labels: bool = False, return_X_y: bool = False,
                      downloads_path: str = None) -> Bunch:
    """
    Load the Mice Protein Expression data set. It consists of 1077 samples belonging to one of 8 classes.
    Each feature represents the expression level of one of 77 proteins.
    Samples containing more than 43 NaN values (3 cases) will be removed. Afterwards, all columns containing NaN values
    will be removed. This reduces the number of features from 77 to 68.
    The classes can be further subdivided by using the return_additional_labels parameter. This gives the additional
    information mouseID, behavior, treatment type and genotype.
    N=1077, d=68, k=8.

    Parameters
    ----------
    return_additional_labels : bool
        return additional labels (default: False)
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1077 x 68), the labels numpy array (1077)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression
    """
    filename = _get_download_dir(downloads_path) + "/Data_Cortex_Nuclear.xls"
    if not os.path.isfile(filename):
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls",
                       filename)
    xls = pd.ExcelFile(filename)
    # Load first page
    sheet = xls.parse(0)
    # Remove special columns
    classes_raw = sheet.pop("class")
    ids_raw = sheet.pop("MouseID")
    bahaviors_raw = sheet.pop("Behavior")
    treatments_raw = sheet.pop("Treatment")
    genotypes_raw = sheet.pop("Genotype")
    original_data = sheet.values
    # Remove rows containing 43 NaN values (3 cases)
    n_of_nans_per_row = np.sum(np.isnan(original_data), axis=1)
    data = original_data[n_of_nans_per_row < 43]
    # Remove columns containing NaN values (removes 9 columns)
    n_of_nans_per_columns = np.sum(np.isnan(data), axis=0)
    data = data[:, n_of_nans_per_columns == 0]
    # Get labels
    LE = LabelEncoder()
    labels = LE.fit_transform(classes_raw)
    if return_additional_labels:
        ids = [entry.split("_")[0] for entry in ids_raw]
        LE = LabelEncoder()
        id_labels = LE.fit_transform(ids)
        LE = LabelEncoder()
        bahaviors_labels = LE.fit_transform(bahaviors_raw)
        LE = LabelEncoder()
        treatment_labels = LE.fit_transform(treatments_raw)
        LE = LabelEncoder()
        genotype_labels = LE.fit_transform(genotypes_raw)
        labels = np.c_[labels, id_labels, bahaviors_labels, treatment_labels, genotype_labels]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Remove rows also from labels (3 cases)
    labels = labels[n_of_nans_per_row < 43]
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="MiceProtein", data=data, target=labels)


def load_user_knowledge(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the user knowledge data set. It consists of 403 samples belonging to one of 4 classes.
    The 4 classes are the knowledge levels 'very low', 'low', 'middle' and 'high'.
    The data set is composed of 258 training and 145 test samples.
    N=403, d=5, k=4.

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
        the data numpy array (403 x 5), the labels numpy array (403)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    filename = _get_download_dir(downloads_path) + "/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls"
    if not os.path.isfile(filename):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00257/Data_User_Modeling_Dataset_Hamdi%20Tolga%20KAHRAMAN.xls",
            filename)
    xls = pd.ExcelFile(filename)
    if subset == "all" or subset == "train":
        # Load second page
        sheet_train = xls.parse(1)
        # Get data and label columns
        labels_raw = sheet_train.pop(" UNS")
        data = sheet_train.values[:, :5]
    if subset == "all" or subset == "test":
        # Load third page
        sheet_test = xls.parse(2)
        # Get data and label columns
        test_data = sheet_test.values[:, :5]
        uns_test = sheet_test.pop(" UNS")
        # Fix label string 'Very Low' to 'very_low' (as in train file)
        uns_test = [l.replace("Very Low", "very_low") for l in uns_test]
        if subset == "all":
            data = np.r_[data, test_data]
            labels_raw = np.r_[labels_raw, uns_test]
        else:
            data = test_data
            labels_raw = uns_test
    # Transform labels
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    data = np.array(data, dtype=np.float64)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="UserKnowledge", data=data, target=labels)


def load_breast_tissue(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the breast tissue data set. It consists of 106 samples belonging to one of 6 classes.
    N=106, d=9, k=6.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (106 x 9), the labels numpy array (106)

    References
    -------
    http://archive.ics.uci.edu/ml/datasets/breast+tissue
    """
    filename = _get_download_dir(downloads_path) + "/BreastTissue.xls"
    if not os.path.isfile(filename):
        _download_file("http://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls",
                       filename)
    xls = pd.ExcelFile(filename)
    # Load second page
    sheet = xls.parse(1)
    # Get data and label columns
    class_column = sheet.pop("Class")
    data = sheet.values[:, 1:]
    # Transform labels
    LE = LabelEncoder()
    labels = LE.fit_transform(class_column)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="BreastTissue", data=data, target=labels)


def load_forest_types(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the forest type mapping data set. It consists of 523 samples belonging to one of 4 classes.
    The data set is composed of 198 training and 325 test samples.
    N=523, d=27, k=4.

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
        the data numpy array (523 x 27), the labels numpy array (523)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/Forest+type+mapping
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    directory = _get_download_dir(downloads_path) + "/ForestTypes/"
    filename = directory + "ForestTypes.zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00333/ForestTypes.zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # Load data and labels
    if subset == "all" or subset == "train":
        df_train = pd.read_csv(directory + "/training.csv", delimiter=",")
        labels_raw = df_train.pop("class")
        data = df_train.values
    if subset == "all" or subset == "test":
        df_test = pd.read_csv(directory + "/testing.csv", delimiter=",")
        labels_test = df_test.pop("class")
        if subset == "all":
            data = np.r_[data, df_test.values]
            labels_raw = np.r_[labels_raw, labels_test]
        else:
            data = df_test.values
            labels_raw = labels_test
    # Transform labels
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="ForestTypes", data=data, target=labels)


def load_dermatology(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the dermatology data set. It consists of 366 samples belonging to one of 6 classes.
    8 samples contain '?' values and are therefore removed.
    N=358, d=34, k=6.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (358 x 34), the labels numpy array (358)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/dermatology
    """
    filename = _get_download_dir(downloads_path) + "/dermatology.data"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data",
                                   delimiter=",")
    # Remove rows with nan
    rows_with_nan = ~np.isnan(data).any(axis=1)
    data = data[rows_with_nan]
    labels = labels[rows_with_nan]
    # Convert labels from 1,... to 0,...
    labels -= 1
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Dermatology", data=data, target=labels)


def load_multiple_features(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the multiple features data set. It consists of 2000 samples belonging to one of 10 classes.
    Each class corresponds to handwritten numerals (0-9) extracted from a collection of Dutch utility maps.
    N=2000, d=649, k=10.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (2000 x 649), the labels numpy array (2000)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/Multiple+Features
    """
    directory = _get_download_dir(downloads_path) + "/MultipleFeatures/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    data = np.zeros((2000, 0))
    # Dataset consists of multiple .xls files
    for file in ["mfeat-fac", "mfeat-fou", "mfeat-kar", "mfeat-mor", "mfeat-pix", "mfeat-zer"]:
        filename = directory + file + ".xls"
        if not os.path.isfile(filename):
            _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/" + file,
                           filename)
        data_tmp = np.genfromtxt(filename, delimiter=None)
        data = np.c_[data, data_tmp]
    # First 200 entries correspond to '0', next 200 to '1' and so on
    labels = np.repeat(range(10), 200)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="MultipleFeatures", data=data, target=labels)


def load_statlog_australian_credit_approval(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the statlog Australian Credit Approval data set. It consists of 690 samples belonging to one of 2 classes.
    N=690, d=14, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (690 x 14), the labels numpy array (690)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
    """
    filename = _get_download_dir(downloads_path) + "/australian.dat"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat",
                                   delimiter=None)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="StatlogAustralianCreditApproval", data=data, target=labels)


def load_breast_cancer_wisconsin_original(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the original breast cancer Wisconsin data set. It consists of 699 samples belonging to one of 2 classes.
    16 samples contain '?' values and will be removed.
    N=683, d=9, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (683 x 9), the labels numpy array (683)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29
    """
    filename = _get_download_dir(downloads_path) + "/breast-cancer-wisconsin.data"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                                   delimiter=",")
    # First column contains unique ids
    data = data[:, 1:]
    # Remove rows with nan
    rows_with_nan = ~np.isnan(data).any(axis=1)
    data = data[rows_with_nan]
    labels = labels[rows_with_nan]
    # labels are 2 or 4. Convert to 0 or 1
    labels = labels / 2 - 1
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="BreastCancerWisconsin", data=data, target=labels)


def load_optdigits(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the optdigits data set. It consists of 5620 8x8 grayscale images, each representing a digit (0 to 9).
    Each pixel depicts the number of marked pixel within a 4x4 block of the original 32x32 bitmaps.
    The data set is composed of 3823 training and 1797 test samples.
    N=5620, d=64, k=10.

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
        Furthermore, the original images are contained in the 'images' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (5620 x 64), the labels numpy array (5620)

    References
    -------
    http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    if subset == "all" or subset == "train":
        filename = _get_download_dir(downloads_path) + "/optdigits.tra"
        data, labels = _load_data_file(filename,
                                       "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra")
    if subset == "all" or subset == "test":
        filename = _get_download_dir(downloads_path) + "/optdigits.tes"
        test_data, test_labels = _load_data_file(filename,
                                                 "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    # Return values
    if return_X_y:
        return data, labels
    else:
        data_image = data.reshape((-1, 8, 8))
        return Bunch(dataset_name="Optdigits", data=data, target=labels, images=data_image, image_format="HW")


def load_semeion(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the semeion data set. It consists of 1593 samples belonging to one of 10 classes.
    Each sample corresponds to a grayscale 16x16 scan of handwritten digits originating from about 80 different persons.
    Further, each pixel was converted to a boolean value using a fixed threshold.
    N=1593, d=256, k=10.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1593 x 256), the labels numpy array (1593)

    References
    -------
    https://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit
    """
    filename = _get_download_dir(downloads_path) + "/semeion.data"
    if not os.path.isfile(filename):
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
                       filename)
    datafile = np.genfromtxt(filename)
    # Last columns each correspond to one label (one-hot encoding)
    data = datafile[:, :-10]
    labels = np.zeros(data.shape[0], dtype=np.int32)
    for i in range(1, 10):
        labels[datafile[:, -10 + i] == 1] = i
    # Return values
    if return_X_y:
        return data, labels
    else:
        data_image = data.reshape((-1, 16, 16))
        return Bunch(dataset_name="Semeion", data=data, target=labels, images=data_image, image_format="HW")


def load_cmu_faces(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the CMU Face Images data set. It consists of 640 30x32 grayscale images showing 20 persons in different poses
    (up, straight, left, right) and with different expressions (neutral, happy, sad, angry). Additionally, the persons
    can wear sunglasses or not.
    16 images show glitches which is why the final data set only contains 624 images.
    N=624, d=400, k=[20,4,4,2].

    Parameters
    -------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (624 x 400), the labels numpy array (624 x 4)

    References
    -------
    http://archive.ics.uci.edu/ml/datasets/cmu+face+images
    """
    directory = _get_download_dir(downloads_path) + "/cmufaces/"
    filename = directory + "faces_4.tar.gz"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces_4.tar.gz",
                       filename)
        # Unpack zipfile
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(directory)
    names = np.array(
        ["an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman", "karyadi", "kawamura",
         "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"])
    positions = np.array(["straight", "left", "right", "up"])
    expressions = np.array(["neutral", "happy", "sad", "angry"])
    eyes = np.array(["open", "sunglasses"])
    data_list = []
    label_list = []
    for name in names:
        path_images = directory + "/faces_4/" + name
        for image in os.listdir(path_images):
            if not image.endswith("_4.pgm"):
                continue
            # get image data
            image_data = Image.open(path_images + "/" + image)
            image_array = np.array(image_data)
            # Get labels
            name_parts = image.split("_")
            user_id = np.argwhere(names == name_parts[0])[0][0]
            position = np.argwhere(positions == name_parts[1])[0][0]
            expression = np.argwhere(expressions == name_parts[2])[0][0]
            eye = np.argwhere(eyes == name_parts[3])[0][0]
            label_data = np.array([user_id, position, expression, eye])
            # Save data and labels
            data_list.append(image_array)
            label_list.append(label_data)
    labels = np.array(label_list, dtype=np.int32)
    data_image = np.array(data_list)
    # Flatten data
    data_flatten = flatten_images(data_image, "HW")
    # Return values
    if return_X_y:
        return data_flatten, labels
    else:
        return Bunch(dataset_name="CMUFace", data=data_flatten, target=labels, images=data_image, image_format="HW",
                     classes=[names, positions, expressions, eyes])

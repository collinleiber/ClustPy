import torchvision
import torch
import urllib.request
import os.path
from pathlib import Path
import ssl
import numpy as np
import zipfile

# More datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps

def _get_download_dir():
    downloads_path = str(Path.home() / "Downloads/cluspy_datafiles")
    if not os.path.isdir(downloads_path):
        os.makedirs(downloads_path)
    return downloads_path


def _download_file(download_path, filename):
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(download_path, filename)
    ssl._create_default_https_context = ssl._create_default_https_context


def _load_uci_data(filename, download_path):
    if not os.path.isfile(filename):
        _download_file(download_path, filename)
    datafile = np.genfromtxt(filename, delimiter=",")
    data = datafile[:, :-1]
    labels = datafile[:, -1]
    return data, labels


def _load_torch_image_data(data_source, add_testdata, normalize_channels):
    # Get data from source
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = data_source(root=_get_download_dir(), train=True, download=True)
    data = dataset.data
    labels = dataset.targets
    if add_testdata:
        testset = data_source(root=_get_download_dir(), train=False, download=True)
        data = torch.cat([data, testset.data], dim=0)
        labels = torch.cat([labels, testset.targets], dim=0)
    ssl._create_default_https_context = ssl._create_default_https_context
    if normalize_channels:
        if data.dim() == 3:
            data_mean = [data.mean()]
            data_std = [data.std()]
        else:
            data_mean = []
            data_std = []
            for i in range(data.dim()):
                data_mean.append(data[:, i, :, :].mean())
                data_std.append(data[:, i, :, :].std())
        normalize = torchvision.transforms.Normalize(data_mean, data_std)
        data = normalize(data)
    # Flatten shape
    if data.dim() == 3:
        data = data.reshape(-1, data.shape[1] * data.shape[2])
    else:
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    # Move data to CPU
    data_cpu = data.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return data_cpu, labels_cpu


def load_mnist(add_testdata=True, normalize_channels=False):
    data, labels = _load_torch_image_data(torchvision.datasets.MNIST, add_testdata, normalize_channels)
    return data, labels


def load_kmnist(add_testdata=True, normalize_channels=False):
    data, labels = _load_torch_image_data(torchvision.datasets.KMNIST, add_testdata, normalize_channels)
    return data, labels


def load_fmnist(add_testdata=True, normalize_channels=False):
    data, labels = _load_torch_image_data(torchvision.datasets.FashionMNIST, add_testdata, normalize_channels)
    return data, labels


def load_usps(add_testdata=True):
    dataset = torchvision.datasets.USPS(root=_get_download_dir(), train=True, download=True)
    data = dataset.data
    labels = dataset.targets
    if add_testdata:
        test_dataset = torchvision.datasets.USPS(root=_get_download_dir(), train=False, download=True)
        data = np.r_[data, test_dataset.data]
        labels = np.r_[labels, test_dataset.targets]
    data = data.reshape(-1, 256)
    return data, labels


def load_optdigits(add_testdata=True):
    filename = _get_download_dir() + "/optdigits.tra"
    data, labels = _load_uci_data(filename,
                                  "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra")
    if add_testdata:
        filename = _get_download_dir() + "/optdigits.tes"
        test_data, test_labels = _load_uci_data(filename,
                                      "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    return data, labels


def load_pendigits(add_testdata=True):
    filename = _get_download_dir() + "/pendigits.tra"
    data, labels = _load_uci_data(filename,
                                  "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra")
    if add_testdata:
        filename = _get_download_dir() + "/pendigits.tes"
        test_data, test_labels = _load_uci_data(filename,
                                      "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    return data, labels


def load_letterrecognition():
    filename = _get_download_dir() + "/letter-recognition.data"
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
    return data, labels


def load_har(add_testdata=True):
    filename = _get_download_dir() + "/har/UCI HAR Dataset.zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(_get_download_dir() + "/har/"):
            os.mkdir(_get_download_dir() + "/har/")
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(_get_download_dir() + "/har/")
    # Load data and labels
    data = np.genfromtxt(_get_download_dir() + "/har/UCI HAR Dataset/train/X_train.txt")
    labels = np.genfromtxt(_get_download_dir() + "/har/UCI HAR Dataset/train/y_train.txt")
    if add_testdata:
        test_data = np.genfromtxt(_get_download_dir() + "/har/UCI HAR Dataset/test/X_test.txt")
        test_labels = np.genfromtxt(_get_download_dir() + "/har/UCI HAR Dataset/test/y_test.txt")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    return data, labels

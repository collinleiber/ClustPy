import torchvision
import torch
import urllib.request
import os
from pathlib import Path
import ssl
import numpy as np
import zipfile
import tarfile
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1, load_iris as sk_load_iris, load_wine as sk_load_wine, \
    load_breast_cancer as sk_load_breast_cancer
import pandas as pd
from PIL import Image
import re
from nltk.stem import SnowballStemmer

DEFAULT_DOWNLOAD_PATH = str(Path.home() / "Downloads/cluspy_datafiles")


# More datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps

def _get_download_dir(downloads_path):
    if downloads_path is None:
        downloads_path = DEFAULT_DOWNLOAD_PATH
    if not os.path.isdir(downloads_path):
        os.makedirs(downloads_path)
    return downloads_path


def _download_file(download_path, filename):
    print("Downloading data set from {0} to {1}".format(download_path, filename))
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(download_path, filename)
    ssl._create_default_https_context = ssl._create_default_https_context


def _load_data_file(filename, download_path, delimiter=",", last_column_are_labels=True):
    if not os.path.isfile(filename):
        _download_file(download_path, filename)
    datafile = np.genfromtxt(filename, delimiter=delimiter)
    if last_column_are_labels:
        data = datafile[:, :-1]
        labels = datafile[:, -1]
    else:
        data = datafile[:, 1:]
        labels = datafile[:, 0]
    return data, labels


def _load_timeseries_classification_data(name, add_testdata, downloads_path):
    directory = _get_download_dir(downloads_path) + "/" + name + "/"
    filename = directory + name + ".zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.timeseriesclassification.com/Downloads/" + name + ".zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # Load data and labels
    dataset = np.genfromtxt(directory + name + "_TRAIN.txt")
    data = dataset[:, 1:]
    labels = dataset[:, 0]
    if add_testdata:
        test_dataset = np.genfromtxt(directory + name + "_TEST.txt")
        data = np.r_[data, test_dataset[:, 1:]]
        labels = np.r_[labels, test_dataset[:, 0]]
    return data, labels


def _decompress_z_file(filename, directory):
    os.system("7z x {0} -o{1}".format(filename.replace("\\", "/"), directory.replace("\\", "/")))
    if os.path.isfile(filename[:-2]):
        return True
    else:
        print("[WARNING] 7Zip is needed to uncompress *.Z files!")
        return False


"""
Load torichvision datasets
"""


def _load_torch_image_data(data_source, add_testdata, normalize_channels, downloads_path):
    # Get data from source
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = data_source(root=_get_download_dir(downloads_path), train=True, download=True)
    data = dataset.data
    labels = dataset.targets
    if add_testdata:
        testset = data_source(root=_get_download_dir(downloads_path), train=False, download=True)
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


def load_mnist(add_testdata=True, normalize_channels=False, downloads_path=None):
    data, labels = _load_torch_image_data(torchvision.datasets.MNIST, add_testdata, normalize_channels, downloads_path)
    return data, labels


def load_kmnist(add_testdata=True, normalize_channels=False, downloads_path=None):
    data, labels = _load_torch_image_data(torchvision.datasets.KMNIST, add_testdata, normalize_channels, downloads_path)
    return data, labels


def load_fmnist(add_testdata=True, normalize_channels=False, downloads_path=None):
    data, labels = _load_torch_image_data(torchvision.datasets.FashionMNIST, add_testdata, normalize_channels,
                                          downloads_path)
    return data, labels


def load_usps(add_testdata=True, downloads_path=None):
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = torchvision.datasets.USPS(root=_get_download_dir(downloads_path), train=True, download=True)
    data = np.array(dataset.data)
    labels = np.array(dataset.targets)
    if add_testdata:
        test_dataset = torchvision.datasets.USPS(root=_get_download_dir(downloads_path), train=False, download=True)
        data = np.r_[data, test_dataset.data]
        labels = np.r_[labels, test_dataset.targets]
    ssl._create_default_https_context = ssl._create_default_https_context
    data = data.reshape(-1, 256)
    return data, labels


"""
Load Sklearn datasets
"""


def load_iris():
    return sk_load_iris(return_X_y=True)


def load_wine():
    return sk_load_wine(return_X_y=True)


def load_breast_cancer():
    return sk_load_breast_cancer(return_X_y=True)


def load_newsgroups(add_testdata=True, n_features=2000):
    newsgroups = fetch_20newsgroups(subset='all' if add_testdata else 'train', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=n_features, dtype=np.float64, sublinear_tf=True)
    data_sparse = vectorizer.fit_transform(newsgroups.data)
    data = np.asarray(data_sparse.todense())
    labels = newsgroups.target
    return data, labels


def load_reuters(add_testdata=True, n_features=2000):
    reuters = fetch_rcv1(subset='all' if add_testdata else 'train')
    # Get samples with relevant main categories
    relevant_cats = np.where(
        (reuters.target_names == 'CCAT') | (reuters.target_names == 'GCAT') | (reuters.target_names == 'MCAT')
        | (reuters.target_names == 'ECAT'))[0]
    filtered_labels = reuters.target[:, relevant_cats]
    # Only get documents with single category
    sum_of_labelings = np.sum(filtered_labels, axis=1)
    single_doc_ids = np.where(sum_of_labelings == 1)[0]
    # Get category of these documents
    labels = np.argmax(filtered_labels[single_doc_ids], axis=1)
    labels = np.asarray(labels)[:, 0]
    for i, cat in enumerate(relevant_cats):
        labels[labels == cat] = i
    # Get most frequent columns
    reuters_data = reuters.data[single_doc_ids]
    frequencies = np.asarray(np.sum(reuters_data, axis=0))[0]
    sorted_frequencies = np.argsort(frequencies)[::-1]
    selected_features = sorted_frequencies[:n_features]
    data = np.asarray(reuters_data[:, selected_features].todense())
    return data, labels


"""
Load UCI data
"""


def load_banknotes(downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/data_banknote_authentication.txt"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    return data, labels


def load_spambase(downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/spambase.data"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data")
    return data, labels


def load_seeds(downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/seeds_dataset.txt"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
                                   delimiter=None)
    return data, labels


def load_skin(downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/Skin_NonSkin.txt"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt",
                                   delimiter=None)
    labels -= 1
    return data, labels


def load_soybean_small(downloads_path=None):
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
    return data, labels


def load_soybean_large(add_testdata=True, downloads_path=None):
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
    if add_testdata:
        filename = _get_download_dir(downloads_path) + "/soybean-large.test"
        if not os.path.isfile(filename):
            _download_file(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.test",
                filename)
        df_test = pd.read_csv(filename, delimiter=",", header=None)
        df_test = df_test[(df_test != '?').all(axis=1)]
        labels_test = df_test.pop(0)
        data = np.r_[data, df_test.values]
        labels_raw = np.r_[labels_raw, labels_test]
    # Transform data to numerical array
    data = np.array(data, dtype=np.int)
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    return data, labels


def load_optdigits(add_testdata=True, downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/optdigits.tra"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra")
    if add_testdata:
        filename = _get_download_dir(downloads_path) + "/optdigits.tes"
        test_data, test_labels = _load_data_file(filename,
                                                 "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    return data, labels


def load_pendigits(add_testdata=True, downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/pendigits.tra"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra")
    if add_testdata:
        filename = _get_download_dir(downloads_path) + "/pendigits.tes"
        test_data, test_labels = _load_data_file(filename,
                                                 "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    return data, labels


def load_ecoli(downloads_path=None):
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
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    return data, labels


def load_htru2(downloads_path=None):
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
    return data, labels


def load_letterrecognition(downloads_path=None):
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
    return data, labels


def load_har(add_testdata=True, downloads_path=None):
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
    data = np.genfromtxt(directory + "UCI HAR Dataset/train/X_train.txt")
    labels = np.genfromtxt(directory + "UCI HAR Dataset/train/y_train.txt")
    if add_testdata:
        test_data = np.genfromtxt(directory + "UCI HAR Dataset/test/X_test.txt")
        test_labels = np.genfromtxt(directory + "UCI HAR Dataset/test/y_test.txt")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    labels = labels - 1
    return data, labels


def load_shuttle(add_testdata=True, downloads_path=None):
    directory = _get_download_dir(downloads_path) + "/shuttle/"
    filename = directory + "shuttle.trn.Z"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z",
                       filename)
        # Unpack z-file
        success = _decompress_z_file(filename, directory)
        if not success:
            # os.remove(filename)
            return None, None
    # Load data and labels
    dataset = np.genfromtxt(directory + "shuttle.trn")
    data = dataset[:, :-1]
    labels = dataset[:, -1]
    if add_testdata:
        filename = directory + "shuttle.tst"
        if not os.path.isfile(filename):
            _download_file(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst",
                filename)
        test_dataset = np.genfromtxt(directory + "shuttle.tst")
        test_data = test_dataset[:, :-1]
        test_labels = test_dataset[:, -1]
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    labels -= 1
    return data, labels


def load_mice_protein(return_multiple_labels=False, downloads_path=None):
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
    if return_multiple_labels:
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
    # Remove rows also from labels (3 cases)
    labels = labels[n_of_nans_per_row < 43]
    return data, labels


def load_user_knowledge(add_testdata=True, downloads_path=None):
    filename = _get_download_dir(downloads_path) + "/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls"
    if not os.path.isfile(filename):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00257/Data_User_Modeling_Dataset_Hamdi%20Tolga%20KAHRAMAN.xls",
            filename)
    xls = pd.ExcelFile(filename)
    # Load second page
    sheet_train = xls.parse(1)
    # Get data and label columns
    labels_raw = sheet_train.pop(" UNS")
    data = sheet_train.values[:, :5]
    # Transform labels
    if add_testdata:
        # Load third page
        sheet_test = xls.parse(2)
        # Get data and label columns
        test_data = sheet_test.values[:, :5]
        uns_test = sheet_test.pop(" UNS")
        data = np.r_[data, test_data]
        labels_raw = np.r_[labels_raw, uns_test]
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    return data, labels


def load_breast_tissue(downloads_path=None):
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
    return data, labels


def load_cmu_faces(downloads_path=None):
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
            image_data_vector = np.array(image_data).reshape(image_data.size[0] * image_data.size[1])
            # Get labels
            name_parts = image.split("_")
            user_id = np.argwhere(names == name_parts[0])[0][0]
            position = np.argwhere(positions == name_parts[1])[0][0]
            expression = np.argwhere(expressions == name_parts[2])[0][0]
            eye = np.argwhere(eyes == name_parts[3])[0][0]
            label_data = np.array([user_id, position, expression, eye])
            # Save data and labels
            data_list.append(image_data_vector)
            label_list.append(label_data)
    labels = np.array(label_list)
    data = np.array(data_list)
    return data, labels


def load_forest_types(add_testdata=True, downloads_path=None):
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
    df_train = pd.read_csv(directory + "/training.csv", delimiter=",")
    labels_raw = df_train.pop("class")
    data = df_train.values
    if add_testdata:
        df_test = pd.read_csv(directory + "/testing.csv", delimiter=",")
        labels_test = df_test.pop("class")
        data = np.r_[data, df_test.values]
        labels_raw = np.r_[labels_raw, labels_test]
    LE = LabelEncoder()
    labels = LE.fit_transform(labels_raw)
    return data, labels


"""
Load timeseries classification data
"""


def load_motestrain(add_testdata=True, downloads_path=None):
    data, labels = _load_timeseries_classification_data("MoteStrain", add_testdata, downloads_path)
    labels -= 1
    return data, labels


def load_proximal_phalanx_outline(add_testdata=True, downloads_path=None):
    data, labels = _load_timeseries_classification_data("DistalPhalanxOutlineCorrect", add_testdata, downloads_path)
    return data, labels


def load_diatom_size_reduction(add_testdata=True, downloads_path=None):
    data, labels = _load_timeseries_classification_data("DiatomSizeReduction", add_testdata, downloads_path)
    labels -= 1
    return data, labels


def load_symbols(add_testdata=True, downloads_path=None):
    data, labels = _load_timeseries_classification_data("Symbols", add_testdata, downloads_path)
    labels -= 1
    return data, labels


def load_olive_oil(add_testdata=True, downloads_path=None):
    data, labels = _load_timeseries_classification_data("OliveOil", add_testdata, downloads_path)
    labels -= 1
    return data, labels


def load_plane(add_testdata=True, downloads_path=None):
    data, labels = _load_timeseries_classification_data("Plane", add_testdata, downloads_path)
    labels -= 1
    return data, labels


"""
Load WebKB
"""


def load_webkb(remove_headers=True, use_categories=["course", "faculty", "project", "student"],
               use_universities=["cornell", "texas", "washington", "wisconsin"], downloads_path=None):
    directory = _get_download_dir(downloads_path) + "/WebKB/"
    filename = directory + "webkb-data.gtar.gz"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz",
                       filename)
        # Unpack zipfile
        with tarfile.open(filename, "r:gz") as tar:
            for obj in tar.getmembers():
                if obj.isdir():
                    # Create Directory
                    tar.extract(obj, directory)
                else:
                    # Can not handle filenames with special characters. Therefore, rename files
                    new_name = obj.name.replace("~", "_").replace(".", "_").replace("^", "_").replace(":", "_").replace(
                        "\r", "")
                    # Get file content
                    f = tar.extractfile(obj)
                    lines = f.readlines()
                    # Write file
                    with open(directory + new_name, "wb") as output:
                        for line in lines:
                            output.write(line)
    texts = []
    labels = np.empty((0, 2), dtype=np.int)
    hmtl_tags = re.compile(r'<[^>]+>')
    head_tags = re.compile(r'MIME-Version:[:,./\-\w\s]+<html>')
    number_tags = re.compile(r'\d*')
    # Read files
    for i, category in enumerate(use_categories):
        for j, univerity in enumerate(use_universities):
            inner_directory = "{0}webkb/{1}/{2}/".format(directory, category, univerity)
            files = os.listdir(inner_directory)
            for file in files:
                with open(inner_directory + file, "r") as f:
                    lines = f.read()
                    if remove_headers:
                        # Remove header
                        lines = head_tags.sub('', lines)
                    # Remove HTML tags
                    lines = hmtl_tags.sub('', lines)
                    lines = number_tags.sub('', lines)
                    texts.append(lines)
                    labels = np.r_[labels, [[i, j]]]
    # Execute TF-IDF and remove stop-words
    vectorizer = _StemmedCountVectorizer(dtype=np.float64, stop_words="english", min_df=0.01)
    data_sparse = vectorizer.fit_transform(texts)
    selector = VarianceThreshold(0.25)  # 0.25 ohne min_df
    data_sparse = selector.fit_transform(data_sparse)
    tfidf = TfidfTransformer(sublinear_tf=True)
    data_sparse = tfidf.fit_transform(data_sparse)
    data = np.asarray(data_sparse.todense())
    return data, labels


class _StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer('english')
        analyzer = super(_StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(word) for word in analyzer(doc))

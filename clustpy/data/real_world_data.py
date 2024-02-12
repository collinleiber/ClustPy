try:
    from nltk.stem import SnowballStemmer
except:
    print(
        "[WARNING] Could not import nltk in clustpy.data.real_world_data. Please install nltk by 'pip install nltk' if necessary")
try:
    from PIL import Image
except:
    print(
        "[WARNING] Could not import PIL in clustpy.data.real_world_data. Please install PIL by 'pip install Pillow' if necessary")
from clustpy.data._utils import _download_file, _get_download_dir, _download_file_from_google_drive, _load_image_data, \
    flatten_images
import os
import numpy as np
import zipfile
import tarfile
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1, load_iris as sk_load_iris, load_wine as sk_load_wine, \
    load_breast_cancer as sk_load_breast_cancer, fetch_olivetti_faces
from scipy.io import loadmat
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets._base import Bunch

# More datasets https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps


"""
Load Sklearn datasets
"""


def load_iris(return_X_y: bool = False) -> Bunch:
    """
    Load the iris data set. It consists of the petal and sepal width and length of three different types of irises (Setosa,
    Versicolour, Virginica).
    N=150, d=4, k=3.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (150 x 4), the labels numpy array (150)

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
    https://archive.ics.uci.edu/ml/datasets/iris
    """
    dataset = sk_load_iris(return_X_y=return_X_y)
    if not return_X_y:
        dataset.dataset_name = "Iris"
    return dataset


def load_wine(return_X_y: bool = False) -> Bunch:
    """
    Load the wine data set. It consists of 13 different properties of three different types of wine.
    N=178, d=13, k=3.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)


    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (178 x 13), the labels numpy array (178)

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
    https://archive.ics.uci.edu/ml/datasets/wine
    """
    dataset = sk_load_wine(return_X_y=return_X_y)
    if not return_X_y:
        dataset.dataset_name = "Wine"
    return dataset


def load_breast_cancer(return_X_y: bool = False) -> Bunch:
    """
    Load the breast cancer wisconsin data set. It consists of 32 features computed from digitized images of fine needle
    aspirate of breast mass. The classes are the result of a diagnosis (malignant or benign).
    N=569, d=30, k=2.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (569 x 30), the labels numpy array (569)

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
    https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
    """
    dataset = sk_load_breast_cancer(return_X_y=return_X_y)
    if not return_X_y:
        dataset.dataset_name = "BreastCancer"
    return dataset


def load_olivetti_faces(return_X_y: bool = False) -> Bunch:
    """
    Load the olivetti faces data set. It consists of 400 64x64 grayscale images showing faces of 40 different persons.
    N=400, d=4096, k=40.

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (400 x 4096), the labels numpy array (400)

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
    """
    dataset = fetch_olivetti_faces()
    if return_X_y:
        return dataset.data, dataset.target
    else:
        dataset.image_format = "HW"
        dataset.dataset_name = "OlivettiFaces"
        return dataset


def load_newsgroups(subset: str = "all", n_features: int = 2000, return_X_y: bool = False) -> Bunch:
    """
    Load the 20 newsgroups data set. It consists of a collection of 18846 newsgroup documents, partitioned
    (nearly) evenly across 20 different newsgroups. The documents are converted into feature vectors using TF-IDF.
    The data set is composed of 11314 training and 7532 test documents.
    N=18846, d=2000, k=20 using the default settings.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    n_features : int
        number of features used by TF-IDF (default: 2000)
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (18846 x 2000 - using the default settings), the labels numpy array (18846)

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups
    http://qwone.com/~jason/20Newsgroups/
    """
    newsgroups = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=n_features, dtype=np.float64, sublinear_tf=True)
    data_sparse = vectorizer.fit_transform(newsgroups.data)
    data = np.asarray(data_sparse.todense())
    if return_X_y:
        return data, newsgroups.target
    else:
        return Bunch(dataset_name="20Newsgroups", data=data, target=newsgroups.target)


def load_reuters(subset: str = "all", n_features: int = 2000, categories: tuple = ("CCAT", "GCAT", "MCAT", "ECAT"),
                 return_X_y: bool = False) -> Bunch:
    """
    Load the Reuters data set. It consists of over 800000 manually categorized newswire stories made available by Reuters,
    Ltd. Usually only a subset of the categories is used. Those categories are defined by the attribute 'categories'.
    We use only those articles that belong to a single category. Further, we only use the n_features most frequent
    features.
    The data set is composed of 19806 training and 665265 test documents using the default settings.
    N=685071, d=2000, k=4 using the default settings.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    n_features : int
        number of features used (default: 2000)
    categories : tuple
        the categories that should be contained (default: ("CCAT", "GCAT", "MCAT", "ECAT"))
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (685071 x 2000 - using the default settings), the labels numpy array (685071 - using the default settings)

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html#sklearn.datasets.fetch_rcv1

    and

    Lewis, David D., et al. "Rcv1: A new benchmark collection for text categorization research." Journal of machine
    learning research 5.Apr (2004): 361-397.
    """
    reuters = fetch_rcv1(subset=subset)
    # Get samples with relevant main categories
    relevant_cats = [i for i, tn in enumerate(reuters.target_names) if tn in categories]
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
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="Reuters", data=data, target=labels)


"""
Other datasets
"""


def load_imagenet_dog(subset: str = "all",
                      image_size: tuple = (224, 224),
                      breeds: list = ["n02085936-Maltese_dog", "n02086646-Blenheim_spaniel", "n02088238-basset",
                                      "n02091467-Norwegian_elkhound", "n02097209-standard_schnauzer",
                                      "n02099601-golden_retriever", "n02101388-Brittany_spaniel", "n02101556-clumber",
                                      "n02102177-Welsh_springer_spaniel", "n02105056-groenendael", "n02105412-kelpie",
                                      "n02105855-Shetland_sheepdog", "n02107142-Doberman", "n02110958-pug",
                                      "n02112137-chow"],
                      return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the ImageNet Dog data set. It consists of 20580 color images of different sizes showing 120 breeds of dogs.
    The data set is composed of 12000 training and 8580 test images.
    Usually, a subset of 15 dog breeds is used (Maltese_dog, Blenheim_spaniel, Basset, Norwegian_elkhound,
    Standard_schnauzer, Golden_retriever, Brittany_spaniel, Clumber, Welsh_springer_spaniel, Groenendael, Kelpie,
    Shetland_sheepdog, Doberman, Pug, Chow), resulting in 2574 images for the "all" subset.
    N=20580, d=image_size[0]*image_size[1]*3, k=120.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    image_size : tuple
        the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images (default: (224, 224))
    breeds : list
        list containing all the identifiers of the dog breeds that should be extracted. All entries must be of type str.
        If None, all breeds will be extracted.
        Usually, a subset consisting of 15 breeds is extracted (default: list with 15 dog breeds)
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : bool
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (20580 x image_size[0]*image_size[1]*3), the labels numpy array (20580)

    References
    -------
    http://vision.stanford.edu/aditya86/ImageNetDogs/main.html

    and

    Khosla, Aditya, et al. "Novel dataset for fine-grained image categorization: Stanford dogs."
    Proc. CVPR workshop on fine-grained visual categorization (FGVC). Vol. 2. No. 1. Citeseer, 2011.
    """
    assert len(image_size) == 2, "image_size format must match (width, height)"
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    directory = _get_download_dir(downloads_path) + "/ImageNetDog/"
    filename = directory + "images.tar"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
                       filename)
        # Unpack zipfile
        with tarfile.open(filename, "r") as tar:
            tar.extractall(directory)
    # Get files for test/train split
    train_test_filename = directory + "lists.tar"
    if not os.path.isfile(train_test_filename):
        _download_file("http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
                       train_test_filename)
        # Unpack zipfile
        with tarfile.open(train_test_filename, "r") as tar:
            tar.extractall(directory)
    # Check breeds list
    if breeds is None:
        breeds = os.listdir(directory + "/Images")
    # Load data lists
    data_list = []
    if subset == "train":
        object_list = loadmat(directory + "/train_list.mat")
    elif subset == "test":
        object_list = loadmat(directory + "/test_list.mat")
    else:
        object_list = loadmat(directory + "/file_list.mat")
    labels = object_list["labels"]
    file_list = object_list["file_list"]
    # get image data
    use_image = np.ones(labels.shape[0], dtype=bool)
    for i, file in enumerate(file_list):
        file = file[0][0]
        if file.split("/")[0] in breeds:
            image_data = _load_image_data(directory + "/Images/" + file, image_size, True)
            data_list.append(image_data)
        else:
            use_image[i] = False
    data_image = np.array(data_list)
    # Flatten data
    data_flatten = flatten_images(data_image, "HWC")
    # Convert labels to int32 format
    labels = labels[use_image, 0].astype(np.int32) - 1
    if breeds is not None:
        # Transform labels
        LE = LabelEncoder()
        labels = LE.fit_transform(labels)
    # Return values
    if return_X_y:
        return data_flatten, labels
    else:
        data_image = np.transpose(data_image, [0, 3, 1, 2])
        image_format = "CHW"
        return Bunch(dataset_name="ImagenetDog", data=data_flatten, target=labels,
                     images=data_image, image_format=image_format, classes=breeds)


def load_imagenet10(use_224_size: bool = True, return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the ImageNet-10 data set. This is a subset of the well-known ImageNet data set with only 10 classes.
    It consists of 13000 224x224 (or 96x96) color images showing different objects.
    N=13000, d=150528, k=10.

    Parameters
    ----------
    use_224_size : bool
        defines wheter the images should be loaded in the size (224 x 224) or (96 x 96) (default: True)
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
        the data numpy array (13000 x 150528), the labels numpy array (13000)

    References
    -------
    https://www.image-net.org/

    and

    Russakovsky, Olga, et al. "Imagenet large scale visual recognition challenge."
    International journal of computer vision 115 (2015): 211-252.
    """
    directory = _get_download_dir(downloads_path) + "/ImageNet10"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Source: https://drive.google.com/drive/folders/1XL0Nohi4vO2f1I4znf388n2pMP8PiKFd
    if use_224_size:
        filename_data = directory + "/data_224.npy"
        if not os.path.isfile(filename_data):
            _download_file_from_google_drive("1sLfA0U9s9Q5Cf8o32GxYoyiyrzZN1K_6", filename_data)
        filename_labels = directory + "/labels_224.npy"
        if not os.path.isfile(filename_labels):
            _download_file_from_google_drive("1OjAQwaGnAfJBW66HFkR7yODLFxnTZWWI", filename_labels)
    else:
        filename_data = directory + "/data_96.npy"
        if not os.path.isfile(filename_data):
            _download_file_from_google_drive("13VbP1qYz6bSeibnoR-w0J_jL9bQf6tGX", filename_data)
        filename_labels = directory + "/labels_96.npy"
        if not os.path.isfile(filename_labels):
            _download_file_from_google_drive("1uiuYUdjyCITLURc5eo8ByP9b51MK_Uk6", filename_labels)
    # Load data and labels
    data_image = np.load(filename_data)
    labels = np.load(filename_labels)
    # Flatten data
    data_flatten = flatten_images(data_image, "HWC")
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    # Return values
    if return_X_y:
        return data_flatten, labels
    else:
        data_image = np.transpose(data_image, [0, 3, 1, 2])
        image_format = "CHW"
        return Bunch(dataset_name="Imagenet10", data=data_flatten, target=labels,
                     images=data_image, image_format=image_format)


def load_coil20(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the COIL-20 data set.
    It consists of 1440 128x128 gray-scale images of 20 objects photographed from 72 different angles.
    N=1440, d=16384, k=20.

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
        the data numpy array (1440 x 16384), the labels numpy array (1440)

    References
    -------
    https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
    """
    directory = _get_download_dir(downloads_path) + "/COIL20/"
    filename = directory + "coil-20-proc.zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # get image data
    data_list = []
    labels = np.zeros(1440, dtype=np.int32)
    for i in range(20):
        for j in range(72):
            image_data = _load_image_data(directory + "coil-20-proc/obj{0}__{1}.png".format(i + 1, j), None, False)
            assert image_data.shape == (
                128, 128), "Shape of image obj{0}__{1}.png is not correct. Mest be (128, 128) but is {2}".format(i + 1,
                                                                                                                 j,
                                                                                                                 image_data.shape)
            data_list.append(image_data)
            labels[i * 72:(i + 1) * 72] = i
    # Convert data to numpy
    data_image = np.array(data_list)
    # Flatten data
    data_flatten = flatten_images(data_image, "HW")
    # Return values
    if return_X_y:
        return data_flatten, labels
    else:
        return Bunch(dataset_name="COIL20", data=data_flatten, target=labels, images=data_image, image_format="HW")


def load_coil100(return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the COIL-100 data set.
    It consists of 7200 128x128 color images of 100 objects photographed from 72 different angles.
    N=7200, d=49152, k=100.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (7200 x 49152), the labels numpy array (7200)

    References
    -------
    https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
    """
    directory = _get_download_dir(downloads_path) + "/COIL100/"
    filename = directory + "coil-100.zip"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
                       filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
    # get image data
    data_list = []
    labels = np.zeros(7200, dtype=np.int32)
    for i in range(100):
        for j in range(72):
            image_data = _load_image_data(directory + "coil-100/obj{0}__{1}.png".format(i + 1, j * 5), None, True)
            assert image_data.shape == (
                128, 128, 3), "Shape of image obj{0}__{1}.png is not correct. Mest be (128, 128, 3) but is {2}".format(
                i + 1, j, image_data.shape)
            data_list.append(image_data)
            labels[i * 72:(i + 1) * 72] = i
    # Convert data to numpy
    data_image = np.array(data_list)
    # Flatten data
    data_flatten = flatten_images(data_image, "HWC")
    # Return values
    if return_X_y:
        return data_flatten, labels
    else:
        data_image = np.transpose(data_image, [0, 3, 1, 2])
        image_format = "CHW"
        return Bunch(dataset_name="COIL100", data=data_flatten, target=labels, images=data_image,
                     image_format=image_format)


"""
Load WebKB
"""


def load_webkb(use_universities: tuple = ("cornell", "texas", "washington", "wisconsin"),
               use_categories: tuple = ("course", "faculty", "project", "student"), remove_headers: bool = True,
               min_doc_frequency: float = 0.01, min_variance: float = 0.25, return_X_y: bool = False,
               downloads_path: str = None) -> Bunch:
    """
    Load the WebKB data set. It consists of 1041 Html documents from different universities (default: "cornell", "texas",
    "washington" and "wisconsin"). These web pages have a specified category (default: "course", "faculty", "project",
    "student"). For more information see the references website.
    The data is preprocessed by using stemming and removing stop words. Furthermore, words with a document frequency
    smaller than min_doc_frequency or with a variance smaller than min_variance will be removed.
    N=1041, d=323, k=[4,4] using the default settings.

    Parameters
    ----------
    use_universities : tuple
        specify the universities (default: ("cornell", "texas", "washington", "wisconsin"))
    use_categories : tuple
        specify the categories (default: ("course", "faculty", "project", "student"))
    remove_headers : bool
        should the headers of the Html files be removed? (default: True)
    min_doc_frequency : float
        minimum document frequency of the words (default: 0.01)
    min_variance : float
        minimum variance of the words (default: 0.25)
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (1041 x 323 - using the default settings), the labels numpy array (1041 x 2 - using the default settings)

    References
    -------
    http://www.cs.cmu.edu/~webkb/
    """
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
    labels = np.empty((0, 2), dtype=np.int32)
    hmtl_tags = re.compile(r'<[^>]+>')
    head_tags = re.compile(r'MIME-Version:[:,./\-\w\s]+<html>')
    number_tags = re.compile(r'\d*')
    # Read files
    for i, category in enumerate(use_categories):
        for j, univerity in enumerate(use_universities):
            inner_directory = "{0}webkb/{1}/{2}/".format(directory, category, univerity)
            files = os.listdir(inner_directory)
            for file in files:
                with open(inner_directory + file, "r", encoding='latin-1') as f:
                    lines = f.read()
                    if remove_headers:
                        # Remove header
                        lines = head_tags.sub('', lines)
                    # Remove HTML tags
                    lines = hmtl_tags.sub('', lines)
                    lines = number_tags.sub('', lines)
                    texts.append(lines)
                    labels = np.r_[labels, [[i, j]]]
    # Execute TF-IDF, remove stop-words and use the snowball stemmer
    vectorizer = _StemmedCountVectorizer(dtype=np.float64, stop_words="english", min_df=min_doc_frequency)
    data_sparse = vectorizer.fit_transform(texts)
    selector = VarianceThreshold(min_variance)
    data_sparse = selector.fit_transform(data_sparse)
    tfidf = TfidfTransformer(sublinear_tf=True)
    data_sparse = tfidf.fit_transform(data_sparse)
    data = np.asarray(data_sparse.todense())
    # Return values
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="WebKB", data=data, target=labels, classes=[use_categories, use_universities])


class _StemmedCountVectorizer(CountVectorizer):
    """
    Helper class for load_webkb(). Combines the CountVectorizer with the SnowballStemmer.
    See: https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
    """

    def build_analyzer(self):
        """
        Custom build_analyzer method. Calls the build_analyzer of the CountVectorizer parent class and then applies
        SnowballStemmer('english')

        Returns
        -------
        stemmed_words : Generator
            the stemmed words in the document
        """
        stemmer = SnowballStemmer('english')
        analyzer = super(_StemmedCountVectorizer, self).build_analyzer()
        stemmed_words = lambda doc: (stemmer.stem(word) for word in analyzer(doc))
        return stemmed_words

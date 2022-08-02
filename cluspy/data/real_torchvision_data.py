import torchvision
import torch
import numpy as np
import ssl
from cluspy.data.real_world_data import _get_download_dir

"""
Load torchvision datasets
"""


def _load_torch_image_data(data_source: torchvision.datasets.VisionDataset, subset: str, normalize_channels: bool,
                           uses_train_param: bool, downloads_path: str) -> (
        np.ndarray, np.ndarray):
    """
    Helper function to load a data set from the torchvision package.

    Parameters
    ----------
    data_source : torchvision.datasets.VisionDataset
        the data source from torchvision.datasets
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data
    normalize_channels : bool
        normalize each color-channel of the images
    uses_train_param : bool
        is the test/train parameter called 'train' or 'split' in the data loader. uses_train_param = True corresponds to 'train'
    downloads_path : str
        path to the directory where the data is stored

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test' Your input {0}".format(subset)
    # Get data from source
    ssl._create_default_https_context = ssl._create_unverified_context
    if subset == "all" or subset == "train":
        # Load training data
        if uses_train_param:
            trainset = data_source(root=_get_download_dir(downloads_path), train=True, download=True)
        else:
            trainset = data_source(root=_get_download_dir(downloads_path), split="train", download=True)
        data = trainset.data
        if hasattr(trainset, "targets"):
            # USPS, MNIST, ... use targets
            labels = trainset.targets
        else:
            # SVHN, STL10, ... use labels
            labels = trainset.labels
        if type(data) is np.ndarray:
            # Transform numpy arrays to torch tensors. Needs to be done for eg USPS
            data = torch.from_numpy(data)
            labels = torch.from_numpy(np.array(labels))
    if subset == "all" or subset == "test":
        # Load test data
        if uses_train_param:
            testset = data_source(root=_get_download_dir(downloads_path), train=False, download=True)
        else:
            testset = data_source(root=_get_download_dir(downloads_path), split="test", download=True)
        data_test = testset.data
        if hasattr(testset, "targets"):
            # USPS, MNIST, ... use targets
            labels_test = testset.targets
        else:
            # SVHN, STL10, ... use labels
            labels_test = testset.labels
        if type(data_test) is np.ndarray:
            # Transform numpy arrays to torch tensors. Needs to be done for eg USPS
            data_test = torch.from_numpy(data_test)
            labels_test = torch.from_numpy(np.array(labels_test))
        if subset == "all":
            # Add to train data
            data = torch.cat([data, data_test], dim=0)
            labels = torch.cat([labels, labels_test], dim=0)
        else:
            data = data_test
            labels = labels_test
    # Convert data to float and labels to int
    data = data.float()
    labels = labels.int()
    ssl._create_default_https_context = ssl._create_default_https_context
    if normalize_channels:
        if data.dim() == 3:
            # grayscale images
            data_mean = [data.mean()]
            data_std = [data.std()]
        elif data.dim() == 4:
            # color images
            data_mean = []
            data_std = []
            for i in range(data.dim()):
                data_mean.append(data[:, i, :, :].mean())
                data_std.append(data[:, i, :, :].std())
        else:
            raise Exception(
                "Number of dimensions for torchvision data sets should be 3 or 4. Here dim={0}".format(data.dim()))
        normalize = torchvision.transforms.Normalize(data_mean, data_std)
        data = normalize(data)
    # Flatten shape
    if data.dim() == 3:
        data = data.reshape(-1, data.shape[1] * data.shape[2])
    elif data.dim() == 4:
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    # Move data to CPU
    data_cpu = data.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return data_cpu, labels_cpu


def load_mnist(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the MNIST data set. It consists of 70000 28x28 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : bool
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    """
    data, labels = _load_torch_image_data(torchvision.datasets.MNIST, subset, normalize_channels, True,
                                          downloads_path)
    return data, labels


def load_kmnist(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the Kuzushiji-MNIST data set. It consists of 70000 28x28 grayscale images showing Kanji characters.
    It is composed of 10 different characters, each representing one column of hiragana.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST
    """
    data, labels = _load_torch_image_data(torchvision.datasets.KMNIST, subset, normalize_channels, True,
                                          downloads_path)
    return data, labels


def load_fmnist(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the Fashion-MNIST data set. It consists of 70000 28x28 grayscale images showing articles from the Zalando online store.
    Each sample belongs to one of 10 product groups.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
    """
    data, labels = _load_torch_image_data(torchvision.datasets.FashionMNIST, subset, normalize_channels, True,
                                          downloads_path)
    return data, labels


def load_usps(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the USPS data set. It consists of 9298 16x16 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 7291 training and 2007 test images.
    N=9298, d=256, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (9298 x 256), the labels numpy array (9298)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html#torchvision.datasets.USPS
    """
    data, labels = _load_torch_image_data(torchvision.datasets.USPS, subset, normalize_channels, True,
                                          downloads_path)
    return data, labels


def load_cifar10(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the CIFAR10 data set. It consists of 60000 32x32 color images showing different objects.
    The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.
    The data set is composed of 50000 training and 10000 test images.
    N=60000, d=3072, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (60000 x 3072), the labels numpy array (60000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10
    """
    data, labels = _load_torch_image_data(torchvision.datasets.CIFAR10, subset, normalize_channels, True,
                                          downloads_path)
    return data, labels


def load_svhn(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the SVHN data set. It consists of 99289 32x32 color images showing house numbers (0 to 9).
    The data set is composed of 73257 training and 26032 test images.
    N=99289, d=3072, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (99289 x 3072), the labels numpy array (99289)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
    """
    data, labels = _load_torch_image_data(torchvision.datasets.SVHN, subset, normalize_channels, False,
                                          downloads_path)
    return data, labels


def load_stl10(subset: str = "all", normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the STL10 data set. It consists of 13000 96x96 color images showing different objects.
    The classes are airplane, bird, car, cat, deer, dog, horse, monkey, ship and truck.
    The data set is composed of 5000 training and 8000 test images.
    N=13000, d=27648, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/cluspy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (13000 x 27648), the labels numpy array (13000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html#torchvision.datasets.STL10
    """
    data, labels = _load_torch_image_data(torchvision.datasets.STL10, subset, normalize_channels, False,
                                          downloads_path)
    return data, labels

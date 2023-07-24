import torchvision
import torch
import numpy as np
import ssl
from clustpy.data._utils import _get_download_dir

"""
Load torchvision datasets
"""


def _load_torch_image_data(data_source: torchvision.datasets.VisionDataset, subset: str, flatten: bool,
                           normalize_channels: bool, uses_train_param: bool, downloads_path: str,
                           is_color_channel_last: bool) -> (np.ndarray, np.ndarray):
    """
    Helper function to load a data set from the torchvision package.
    All data sets will be returned as a two-dimensional tensor, created out of the HWC (height, width, color channels) image representation.

    Parameters
    ----------
    data_source : torchvision.datasets.VisionDataset
        the data source from torchvision.datasets
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, color images will be returned in the CHW format
    normalize_channels : bool
        normalize each color-channel of the images
    uses_train_param : bool
        is the test/train parameter called 'train' or 'split' in the data loader. uses_train_param = True corresponds to 'train'
    downloads_path : str
        path to the directory where the data is stored
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    # Get data from source
    default_ssl = ssl._create_default_https_context
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
    ssl._create_default_https_context = default_ssl
    # Check data dimensions
    if data.dim() < 3 or data.dim() > 5:
        raise Exception(
            "Number of dimensions for torchvision data sets should be 3, 4 or 5. Here dim={0}".format(data.dim()))
    # Normalize and flatten
    data = _torch_normalize_and_flatten(data, flatten, normalize_channels, is_color_channel_last)
    # Move data to CPU
    data_cpu = data.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return data_cpu, labels_cpu


def _torch_normalize_and_flatten(data: torch.Tensor, flatten: bool, normalize_channels: bool,
                                 is_color_channel_last: bool):
    """
    Helper function to load a data set from the torchvision package.
    All data sets will be returned as a two-dimensional tensor, created out of the HWC (height, width, color channels) image representation.

    Parameters
    ----------
    data : torch.Tensor
        The torch data tensor
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, color images will be returned in the CHW format
    normalize_channels : bool
        normalize each color-channel of the images
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images

    Returns
    -------
    data : torch.Tensor
        The (non-)normalized and (non-)flatten data tensor
    """
    # Channels can be normalized
    if normalize_channels:
        data = _torch_normalize_channels(data, is_color_channel_last)
    # Flatten shape
    if flatten:
        data = _torch_flatten_shape(data, is_color_channel_last, normalize_channels)
    elif not normalize_channels:
        # Change image to CHW format
        if data.dim() == 4 and is_color_channel_last:  # equals 2d color images
            # Change to CHW representation
            data = data.permute(0, 3, 1, 2)
            assert data.shape[1] == 3, "Colored image must consist of three channels not " + data.shape[1]
        elif data.dim() == 5 and is_color_channel_last:  # equals 3d color-images
            # Change to CHWD representation
            data = data.permute(0, 4, 1, 2, 3)
            assert data.shape[1] == 3, "Colored image must consist of three channels not {0}".format(data.shape[1])
    return data


def _torch_normalize_channels(data: torch.Tensor, is_color_channel_last: bool) -> torch.Tensor:
    """
    Normalize the color channels of a torch dataset

    Parameters
    ----------
    data : torch.Tensor
        The torch data tensor
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images

    Returns
    -------
    data : torch.Tensor
        The normalized data tensor in CHW format
    """
    if data.dim() == 3 or (data.dim() == 4 and is_color_channel_last is None):
        # grayscale images (2d or 3d)
        data_mean = [data.mean()]
        data_std = [data.std()]
    elif data.dim() == 4:  # equals 2d color images
        if is_color_channel_last:
            # Change to CHW representation
            data = data.permute(0, 3, 1, 2)
        assert data.shape[1] == 3, "Colored image must consist of three channels not " + data.shape[1]
        # color images
        data_mean = data.mean([0, 2, 3])
        data_std = data.std([0, 2, 3])
    elif data.dim() == 5:  # equals 3d color-images
        if is_color_channel_last:
            # Change to CHWD representation
            data = data.permute(0, 4, 1, 2, 3)
        assert data.shape[1] == 3, "Colored image must consist of three channels not {0}".format(data.shape[1])
        # color images
        data_mean = data.mean([0, 2, 3, 4])
        data_std = data.std([0, 2, 3, 4])
    normalize = torchvision.transforms.Normalize(data_mean, data_std)
    data = normalize(data)
    return data


def _torch_flatten_shape(data: torch.Tensor, is_color_channel_last: bool, normalize_channels: bool) -> torch.Tensor:
    """
    Convert torch data tensor from image to numerical vector.

    Parameters
    ----------
    data : torch.Tensor
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images
    normalize_channels : bool
        normalize each color-channel of the images

    Returns
    -------
    data : torch.Tensor
        The flatten data vector
    """
    # Flatten shape
    if data.dim() == 3:
        data = data.reshape(-1, data.shape[1] * data.shape[2])
    elif data.dim() == 4:
        # In case of 3d grayscale image is_color_channel_last is None
        if is_color_channel_last is not None and (not is_color_channel_last or normalize_channels):
            # Change representation to HWC
            data = data.permute(0, 2, 3, 1)
        assert is_color_channel_last is None or data.shape[
            3] == 3, "Colored image must consist of three channels not {0}".format(data.shape[3])
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    elif data.dim() == 5:
        if not is_color_channel_last or normalize_channels:
            # Change representation to HWDC
            data = data.permute(0, 2, 3, 4, 1)
        assert data.shape[4] == 3, "Colored image must consist of three channels not {0}".format(data.shape[4])
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4])
    return data


def load_mnist(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
               downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the MNIST data set. It consists of 70000 28x28 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : bool
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    """
    data, labels = _load_torch_image_data(torchvision.datasets.MNIST, subset, flatten, normalize_channels, True,
                                          downloads_path, None)
    return data, labels


def load_kmnist(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
                downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the Kuzushiji-MNIST data set. It consists of 70000 28x28 grayscale images showing Kanji characters.
    It is composed of 10 different characters, each representing one column of hiragana.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST
    """
    data, labels = _load_torch_image_data(torchvision.datasets.KMNIST, subset, flatten, normalize_channels, True,
                                          downloads_path, None)
    return data, labels


def load_fmnist(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
                downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the Fashion-MNIST data set. It consists of 70000 28x28 grayscale images showing articles from the Zalando online store.
    Each sample belongs to one of 10 product groups.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
    """
    data, labels = _load_torch_image_data(torchvision.datasets.FashionMNIST, subset, flatten, normalize_channels, True,
                                          downloads_path, None)
    return data, labels


def load_usps(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
              downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the USPS data set. It consists of 9298 16x16 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 7291 training and 2007 test images.
    N=9298, d=256, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (9298 x 256), the labels numpy array (9298)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html#torchvision.datasets.USPS
    """
    data, labels = _load_torch_image_data(torchvision.datasets.USPS, subset, flatten, normalize_channels, True,
                                          downloads_path, None)
    return data, labels


def load_cifar10(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
                 downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the CIFAR10 data set. It consists of 60000 32x32 color images showing different objects.
    The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.
    The data set is composed of 50000 training and 10000 test images.
    N=60000, d=3072, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (60000 x 3072), the labels numpy array (60000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10
    """
    data, labels = _load_torch_image_data(torchvision.datasets.CIFAR10, subset, flatten, normalize_channels, True,
                                          downloads_path, True)
    return data, labels


def load_svhn(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
              downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the SVHN data set. It consists of 99289 32x32 color images showing house numbers (0 to 9).
    The data set is composed of 73257 training and 26032 test images.
    N=99289, d=3072, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (99289 x 3072), the labels numpy array (99289)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
    """
    data, labels = _load_torch_image_data(torchvision.datasets.SVHN, subset, flatten, normalize_channels, False,
                                          downloads_path, False)
    return data, labels


def load_stl10(subset: str = "all", flatten: bool = True, normalize_channels: bool = False,
               downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the STL10 data set. It consists of 13000 96x96 color images showing different objects.
    The classes are airplane, bird, car, cat, deer, dog, horse, monkey, ship and truck.
    The data set is composed of 5000 training and 8000 test images.
    N=13000, d=27648, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (13000 x 27648), the labels numpy array (13000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html#torchvision.datasets.STL10
    """
    data, labels = _load_torch_image_data(torchvision.datasets.STL10, subset, flatten, normalize_channels, False,
                                          downloads_path, False)
    return data, labels

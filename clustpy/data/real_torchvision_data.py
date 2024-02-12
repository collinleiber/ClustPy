try:
    import torchvision
except:
    print("[WARNING] Could not import torchvision in clustpy.data.real_torchvision_data. Please install torchvision by 'pip install torchvision' if necessary")
import torch
import numpy as np
import ssl
from clustpy.data._utils import _get_download_dir, _load_image_data, flatten_images
from sklearn.datasets._base import Bunch

"""
Torchvision datasets helpers
"""


def _get_data_and_labels(dataset: torchvision.datasets.VisionDataset, image_size: tuple) -> (
        torch.Tensor, torch.Tensor):
    """
    Extract data and labels from a torchvision dataset object.

    Parameters
    ----------
    dataset : torchvision.datasets.VisionDataset
        The torchvision dataset object
    image_size : tuple
        for some datasets (e.g., GTSRB) the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images

    Returns
    -------
    data, labels : (torch.Tensor, torch.Tensor)
        the data torch tensor, the labels torch tensor
    """
    if hasattr(dataset, "data"):
        # USPS, MNIST, ... use data parameter
        data = dataset.data
        if hasattr(dataset, "targets"):
            # USPS, MNIST, ... use targets
            labels = dataset.targets
        else:
            # SVHN, STL10, ... use labels
            labels = dataset.labels
    else:
        # GTSRB only gives path to images
        labels = []
        data_list = []
        for path, label in dataset._samples:
            labels.append(label)
            image_data = _load_image_data(path, image_size, True)
            data_list.append(image_data)
        # Convert data form list to numpy array
        data = np.array(data_list)
        labels = np.array(labels)
    if type(data) is np.ndarray:
        # Transform numpy arrays to torch tensors. Needs to be done for eg USPS
        data = torch.from_numpy(data)
        labels = torch.from_numpy(np.array(labels))
    return data, labels


def _load_torch_image_data(data_source: torchvision.datasets.VisionDataset, subset: str, uses_train_param: bool,
                           image_format: str, return_X_y: bool, downloads_path: str, image_size: tuple = None) -> Bunch:
    """
    Helper function to load a data set from the torchvision package.
    All data sets will be returned as a two-dimensional tensor, created out of the HWC (height, width, color channels) image representation.

    Parameters
    ----------
    data_source : torchvision.datasets.VisionDataset
        the data source from torchvision.datasets
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data
    uses_train_param : bool
        is the test/train parameter called 'train' or 'split' in the data loader. uses_train_param = True corresponds to 'train'
    image_format : str
        Format of the data array. Can be: "HW", "HWD", "CHW", "CHWD", "HWC", "HWDC".
        Abbreviations stand for: H: Height, W: Width, D: Depth, C: Color-channels
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object
    downloads_path : str
        path to the directory where the data is stored
    image_size : tuple
        for some datasets (e.g., GTSRB) the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images (default: None)

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
        data, labels = _get_data_and_labels(trainset, image_size)
        dataset = trainset
    if subset == "all" or subset == "test":
        # Load test data
        if uses_train_param:
            testset = data_source(root=_get_download_dir(downloads_path), train=False, download=True)
        else:
            testset = data_source(root=_get_download_dir(downloads_path), split="test", download=True)
        data_test, labels_test = _get_data_and_labels(testset, image_size)
        dataset = testset if subset == "test" else dataset
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
    # Transform data to numpy array
    data_image = data.detach().cpu().numpy()
    labels_numpy = labels.detach().cpu().numpy()
    # Flatten data
    data_flatten = flatten_images(data_image, image_format)
    # Return values
    if return_X_y:
        return data_flatten, labels_numpy
    else:
        if image_format == "HWC":
            data_image = np.transpose(data_image, [0, 3, 1, 2])
            image_format = "CHW"
        # Some dataset (e.g., SVHN) do not have the class information included
        if hasattr(dataset, "classes"):
            return Bunch(dataset_name=dataset.__class__.__name__, data=data_flatten, target=labels_numpy,
                         images=data_image, image_format=image_format, classes=dataset.classes)
        else:
            return Bunch(dataset_name=dataset.__class__.__name__, data=data_flatten, target=labels_numpy,
                         images=data_image, image_format=image_format)


"""
Actual datasets
"""


def load_mnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the MNIST data set. It consists of 70000 28x28 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
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
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

    and

    LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE 86.11 (1998): 2278-2324.
    """
    return _load_torch_image_data(torchvision.datasets.MNIST, subset, True, "HW", return_X_y, downloads_path)


def load_kmnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the Kuzushiji-MNIST data set. It consists of 70000 28x28 grayscale images showing Kanji characters.
    It is composed of 10 different characters, each representing one column of hiragana.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html

    and

    Clanuwat, Tarin, et al. "Deep learning for classical japanese literature."
    arXiv preprint arXiv:1812.01718 (2018).
    """
    return _load_torch_image_data(torchvision.datasets.KMNIST, subset, True, "HW", return_X_y, downloads_path)


def load_fmnist(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the Fashion-MNIST data set. It consists of 70000 28x28 grayscale images showing articles from the Zalando online store.
    Each sample belongs to one of 10 product groups.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html

    and

    Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms."
    arXiv preprint arXiv:1708.07747 (2017).
    """
    return _load_torch_image_data(torchvision.datasets.FashionMNIST, subset, True, "HW", return_X_y, downloads_path)


def load_usps(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the USPS data set. It consists of 9298 16x16 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 7291 training and 2007 test images.
    N=9298, d=256, k=10.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (9298 x 256), the labels numpy array (9298)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html

    and

    Hull, Jonathan J. "A database for handwritten text recognition research."
    IEEE Transactions on pattern analysis and machine intelligence 16.5 (1994): 550-554.
    """
    return _load_torch_image_data(torchvision.datasets.USPS, subset, True, "HW", return_X_y, downloads_path)


def load_cifar10(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the CIFAR10 data set. It consists of 60000 32x32 color images showing different objects.
    The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.
    The data set is composed of 50000 training and 10000 test images.
    N=60000, d=3072, k=10.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (60000 x 3072), the labels numpy array (60000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html

    and

    Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.
    """
    return _load_torch_image_data(torchvision.datasets.CIFAR10, subset, True, "HWC", return_X_y, downloads_path)


def load_cifar100(subset: str = "all", use_superclasses: bool = False, return_X_y: bool = False,
                  downloads_path: str = None) -> Bunch:
    """
    Load the CIFAR100 data set. It consists of 60000 32x32 color images showing different objects.
    A total of 100 classes are included, each depicting a specific of objects. Each class contains 600 objects.
    If use_superclasses is True, only the 20 superclasses are used.
    The data set is composed of 50000 training and 10000 test images.
    N=60000, d=3072, k=100.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    use_superclasses : bool
        If set to True, the 20 superclasses are used instead of the 100 regular classes (default: False)
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
        the data numpy array (60000 x 3072), the labels numpy array (60000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html

    and

    Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.
    """
    dataset = _load_torch_image_data(torchvision.datasets.CIFAR100, subset, True, "HWC", False, downloads_path)
    if use_superclasses:
        new_labels = {0: ["beaver", "dolphin", "otter", "seal", "whale"],
                      1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                      2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                      3: ["bottle", "bowl", "can", "cup", "plate"],
                      4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                      5: ["clock", "keyboard", "lamp", "telephone", "television"],
                      6: ["bed", "chair", "couch", "table", "wardrobe"],
                      7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                      8: ["bear", "leopard", "lion", "tiger", "wolf"],
                      9: ["bridge", "castle", "house", "road", "skyscraper"],
                      10: ["cloud", "forest", "mountain", "plain", "sea"],
                      11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                      12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                      13: ["crab", "lobster", "snail", "spider", "worm"],
                      14: ["baby", "boy", "girl", "man", "woman"],
                      15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                      16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                      17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                      18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                      19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]}
        labels_new = np.full(dataset.target.shape, -1, dtype=np.int32)
        for nl in new_labels.keys():
            labels_new[np.isin(np.array(dataset.classes)[dataset.target], new_labels[nl])] = nl
        dataset.target = labels_new
    if return_X_y:
        return dataset.data, dataset.target
    else:
        return dataset


def load_svhn(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the SVHN data set. It consists of 99289 32x32 color images showing house numbers (0 to 9).
    The data set is composed of 73257 training and 26032 test images.
    N=99289, d=3072, k=10.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (99289 x 3072), the labels numpy array (99289)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html

    and

    Netzer, Yuval, et al. "Reading digits in natural images with unsupervised feature learning." (2011).
    """
    return _load_torch_image_data(torchvision.datasets.SVHN, subset, False, "CHW", return_X_y, downloads_path)


def load_stl10(subset: str = "all", return_X_y: bool = False, downloads_path: str = None) -> Bunch:
    """
    Load the STL10 data set. It consists of 13000 96x96 color images showing different objects.
    The classes are airplane, bird, car, cat, deer, dog, horse, monkey, ship and truck.
    The data set is composed of 5000 training and 8000 test images.
    N=13000, d=27648, k=10.

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
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (13000 x 27648), the labels numpy array (13000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html

    and

    Coates, Adam, Andrew Ng, and Honglak Lee. "An analysis of single-layer networks in unsupervised feature learning."
    Proceedings of the fourteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2011.
    """
    return _load_torch_image_data(torchvision.datasets.STL10, subset, False, "CHW", return_X_y, downloads_path)


def load_gtsrb(subset: str = "all", image_size: tuple = (32, 32), return_X_y: bool = False,
               downloads_path: str = None) -> Bunch:
    """
    Load the GTSRB (German Traffic Sign Recognition Benchmark) data set. It consists of 39270 color images showing 43 different traffic signs.
    Example classes are: stop sign, speed limit 50 sign, speed limit 70 sign, construction site sign and many others.
    The data set is composed of 26640 training and 12630 test images.
    N=39270, d=image_size[0]*image_size[1]*3, k=43.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    image_size : tuple
        the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images (default: (32, 32))
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
        the data numpy array (39270 x image_size[0]*image_size[1]*3), the labels numpy array (20580)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html

    and

    https://benchmark.ini.rub.de/

    and

    Stallkamp, Johannes, et al. "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition."
    Neural networks 32 (2012): 323-332.
    """
    return _load_torch_image_data(torchvision.datasets.GTSRB, subset, False, "HWC", return_X_y, downloads_path,
                                  image_size)

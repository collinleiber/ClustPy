import torchvision
import torch
import numpy as np
import ssl
from cluspy.data.real_world_data import _get_download_dir

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

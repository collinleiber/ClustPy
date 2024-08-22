from clustpy.deep._data_utils import _ClustpyDataset, get_dataloader, get_train_and_test_dataloader, \
    get_default_augmented_dataloaders, get_data_dim_from_dataloader
from clustpy.data import create_subspace_data, load_optdigits
import torch
import torchvision
import numpy as np
import os
import pytest


def test_ClustpyDataset():
    data, labels = create_subspace_data(250, subspace_features=(3, 50), random_state=1)
    data_torch = torch.from_numpy(data)
    labels_torch = torch.from_numpy(labels)
    dataset = _ClustpyDataset(data_torch, labels_torch)
    assert len(dataset) == data.shape[0]
    assert len(dataset[0]) == 3
    # Test first sample
    assert dataset[0][0] == 0
    assert torch.equal(dataset[0][1], data_torch[0])
    assert dataset[0][2] == labels_torch[0]
    # Test 100th sample
    assert dataset[100][0] == 100
    assert torch.equal(dataset[100][1], data_torch[100])
    assert dataset[100][2] == labels_torch[100]


def test_ClustpyDataset_with_augmentation():
    data = load_optdigits().images
    data = data.reshape(-1, 1, 8, 8)
    data_torch = torch.from_numpy(data)
    # preprocessing functions
    normalize_fn = torchvision.transforms.Normalize([0], [1])
    # augmentation transforms
    transform_list = [
        # transform input tensor to PIL image for augmentation
        torchvision.transforms.ToPILImage(),
        # apply transformations
        torchvision.transforms.RandomAffine(degrees=(-16, +16), fill=0),
        # transform back to torch.tensor
        torchvision.transforms.ToTensor(),
        # preprocess
        normalize_fn
    ]
    # augmentation transforms
    aug_transforms = [torchvision.transforms.Compose(transform_list)]
    # preprocessing transforms without augmentation
    orig_transforms = [torchvision.transforms.Compose([normalize_fn])]
    dataset = _ClustpyDataset(data_torch, aug_transforms_list=aug_transforms,
                              orig_transforms_list=orig_transforms)
    assert len(dataset) == data.shape[0]
    assert len(dataset[0]) == 3
    # Test first sample
    assert dataset[0][0] == 0
    assert dataset[0][1].shape == (1, 8, 8)
    assert dataset[0][2].shape == (1, 8, 8)
    assert not torch.equal(dataset[0][1], data_torch[0])  # augmented
    assert torch.equal(dataset[0][2], data_torch[0])
    # Test 100th sample
    assert dataset[100][0] == 100
    assert dataset[100][1].shape == (1, 8, 8)
    assert dataset[100][2].shape == (1, 8, 8)
    assert not torch.equal(dataset[100][1], data_torch[100])  # augmented
    assert torch.equal(dataset[100][2], data_torch[100])


def test_get_datalaoder():
    data, labels = create_subspace_data(250, subspace_features=(3, 50), random_state=1)
    data_torch = torch.from_numpy(data).float()
    labels_torch = torch.from_numpy(labels).float()
    # Numpy entry, shuffle=False and no additional input
    dataloader = get_dataloader(data, shuffle=False, batch_size=100)
    entry = next(iter(dataloader))
    assert len(entry) == 2
    assert torch.equal(entry[0], torch.arange(0, 100))
    assert torch.equal(entry[1], data_torch[:100])
    # Torch entry, shuffle=False and additional input
    dataloader = get_dataloader(data_torch, shuffle=False, batch_size=100, additional_inputs=[labels])
    entry = next(iter(dataloader))
    assert len(entry) == 3
    assert torch.equal(entry[0], torch.arange(0, 100))
    assert torch.equal(entry[1], data_torch[:100])
    assert torch.equal(entry[2], labels_torch[:100])
    # Numpy entry, shuffle=True and no additional input
    dataloader = get_dataloader(data, shuffle=True, batch_size=200)
    entry = next(iter(dataloader))
    assert len(entry) == 2
    assert entry[0].shape[0] == 200
    assert not torch.equal(entry[0], torch.arange(0, 200))
    assert entry[1].shape == (200, data.shape[1])
    assert torch.equal(entry[1], data_torch[entry[0]])


def test_get_data_dim_from_dataloader():
    data, labels = create_subspace_data(20, subspace_features=(3, 50), random_state=1)
    dataloader = get_dataloader(data, shuffle=False, batch_size=10)
    assert get_data_dim_from_dataloader(dataloader) == 53
    data, labels = create_subspace_data(20, subspace_features=(1, 10), random_state=1)
    dataloader = get_dataloader(data, shuffle=False, batch_size=10)
    assert get_data_dim_from_dataloader(dataloader) == 11


@pytest.fixture
def cleanup_dataloaders():
    yield
    filename1 = "trainloader.pt"
    if os.path.isfile(filename1):
        os.remove(filename1)
    filename2 = "testloader.pt"
    if os.path.isfile(filename2):
        os.remove(filename2)


@pytest.mark.usefixtures("cleanup_dataloaders")
def test_get_train_and_test_dataloader():
    filename1 = "trainloader.pt"
    filename2 = "testloader.pt"
    data, _ = create_subspace_data(250, subspace_features=(3, 50), random_state=1)
    trainloader, testloader, bs = get_train_and_test_dataloader(data, 64, None)
    assert bs == 64
    custom_dataloader = (trainloader, testloader)
    trainloader2, testloader2, bs = get_train_and_test_dataloader(data, 128, custom_dataloader)
    assert trainloader == trainloader2
    assert testloader == testloader2
    assert bs == 64
    torch.save(trainloader, filename1)
    torch.save(testloader, filename2)
    custom_dataloader = (filename1, filename2)
    trainloader2, testloader2, bs = get_train_and_test_dataloader(data, 128, custom_dataloader)
    assert trainloader != trainloader2
    assert testloader != testloader2
    assert bs == 64
    # Check values in trainloader
    iter_trainloader = iter(trainloader)
    iter_trainloader2 = iter(trainloader2)
    for i in range(4):  # 250 samples and batchsize of 64 => 4 iterations
        torch.manual_seed(123)  # Seed is required due to shuffle
        entry_train1 = next(iter_trainloader)
        torch.manual_seed(123)
        entry_train2 = next(iter_trainloader2)
        assert np.array_equal(entry_train1[0], entry_train2[0])
        assert np.array_equal(entry_train1[1], entry_train2[1])
    # Check values in testloader
    iter_testloader2 = iter(testloader2)
    for entry_test1 in testloader:
        entry_test2 = next(iter_testloader2)
        assert np.array_equal(entry_test1[0], entry_test2[0])
        assert np.array_equal(entry_test1[1], entry_test2[1])
    # Change batch_size of testloader
    testloader = get_dataloader(data, 60, False, False)
    custom_dataloader = (trainloader, testloader)
    trainloader2, testloader2, bs = get_train_and_test_dataloader(data, 64, custom_dataloader)
    assert bs == 64


def test_get_default_augmented_dataloaders():
    X = load_optdigits().images[:250]
    # conv=False and flatten=False
    trainloader, testlaoder = get_default_augmented_dataloaders(X, 63, False, False)
    trainloader_batch = next(iter(trainloader))
    testloader_batch = next(iter(testlaoder))
    assert len(trainloader_batch) == 3
    assert len(testloader_batch) == 2
    assert trainloader_batch[1].shape == (63, 1, 8, 8)
    assert trainloader_batch[2].shape == (63, 1, 8, 8)
    assert testloader_batch[1].shape == (63, 1, 8, 8)
    # conv=True and flatten=False
    trainloader, testlaoder = get_default_augmented_dataloaders(X, 62, True, False)
    trainloader_batch = next(iter(trainloader))
    testloader_batch = next(iter(testlaoder))
    assert len(trainloader_batch) == 3
    assert len(testloader_batch) == 2
    assert trainloader_batch[1].shape == (62, 3, 8, 8)
    assert trainloader_batch[2].shape == (62, 3, 8, 8)
    assert testloader_batch[1].shape == (62, 3, 8, 8)
    # conv=False and flatten=True
    trainloader, testlaoder = get_default_augmented_dataloaders(torch.from_numpy(X), 61, False, True)
    trainloader_batch = next(iter(trainloader))
    testloader_batch = next(iter(testlaoder))
    assert len(trainloader_batch) == 3
    assert len(testloader_batch) == 2
    assert trainloader_batch[1].shape == (61, 64)
    assert trainloader_batch[2].shape == (61, 64)
    assert testloader_batch[1].shape == (61, 64)
    # conv=False and flatten=True
    try:
        # Check for error
        _ = get_default_augmented_dataloaders(X, 60, True, True)
        assert False
    except:
        pass

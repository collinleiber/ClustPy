from clustpy.deep._data_utils import _ClustpyDataset, get_dataloader
from clustpy.data import create_subspace_data, load_optdigits
import torch
import torchvision
import numpy as np


def test_ClustpyDataset():
    data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
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
    data /= 255.0
    data = data.reshape(-1, 1, 8, 8)
    data = np.tile(data, (1, 3, 1, 1))
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
    assert dataset[0][1].shape == (3, 8, 8)
    assert dataset[0][2].shape == (3, 8, 8)
    assert not torch.equal(dataset[0][1], data_torch[0]) # augmented
    assert torch.equal(dataset[0][2], data_torch[0])
    # Test 100th sample
    assert dataset[100][0] == 100
    assert dataset[100][1].shape == (3, 8, 8)
    assert dataset[100][2].shape == (3, 8, 8)
    assert not torch.equal(dataset[100][1], data_torch[100]) # augmented
    assert torch.equal(dataset[100][2], data_torch[100])


def test_get_datalaoder():
    data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
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

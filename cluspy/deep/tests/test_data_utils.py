from cluspy.data import load_optdigits
from cluspy.deep._data_utils import _CluspyDataset, get_dataloader
import torch

def test_CluspyDataset_with_optdigits():
    data, labels = load_optdigits()
    data_torch = torch.from_numpy(data)
    labels_torch = torch.from_numpy(labels)
    dataset = _CluspyDataset(data_torch, labels_torch)
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

def test_get_datalaoder():
    data, labels = load_optdigits()
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

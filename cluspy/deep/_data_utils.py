import torch
import numpy as np

class _CluspyDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping tensors that has the indices always in the first entry.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Implementation is based on torch.utils.data.TensorDataset.

    Parameters
    ----------
    *tensors (torch.Tensor): tensors that have the same size of the first dimension. Usually contains the data.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple([index] + [tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.tensors[0].size(0)

def get_dataloader(X, batch_size, shuffle=True, drop_last=False, additional_inputs=[],
                   dataset_class=_CluspyDataset, **dl_kwargs):
    """
    Create a dataloader for Deep Clustering algorithms.
    First entry always contains the indices of the data samples.
    Second entry always contains the actual data samples.
    If for example labels are desired, they can be passed through the additional_inputs parameter (should be a list).
    Other customizations (e.g. augmentation) can be implemented using a custom dataset_class.
    This custom class should stick to the conventions, [index, data, ...].

    Parameters
    ----------
    X: the actual data set
    batch_size: the batch size
    shuffle: boolean that defines if the data set should be shuffled (default: True)
    drop_last: boolean that defines if the last batch should be ignored (default: False)
    additional_inputs: list containing additional inputs for the dataloader, e.g. labels (default: [])
    dataset_class: defines the class of the tensor dataset that is contained in the dataloader (default: _CluspyDataset)
    dl_kwargs: other arguments for torch.utils.data.DataLoader

    Returns
    -------
    The final dataloader
    """
    assert type(additional_inputs) is list, "additional_input should be of type list."
    if type(X) is np.ndarray:
        X = torch.from_numpy(X).float()
    dataset_input = [X]
    for input in additional_inputs:
        if type(input) is np.ndarray:
            input = torch.from_numpy(input).float()
        dataset_input.append(input)
    dataset = dataset_class(*dataset_input)
    # Create dataloader using the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dl_kwargs)
    return dataloader
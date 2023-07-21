import torch
import torchvision
import numpy as np
from typing import Callable, List

from PIL import Image

class _ClustpyDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping tensors that has the indices always in the first entry.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Optionally you can pass additional augmentation transforms and/or preprocessing transforms. The augmented tensor
    will be in the second entry and the original version in the third entry and so on for additional tensors

    Implementation is based on torch.utils.data.Dataset.

    Parameters
    ----------
    *tensors : torch.Tensor
        tensors that have the same size of the first dimension. Usually contains the data.
    aug_transforms_list : List of torchvision.transforms
        List of augmentation torchvision.transforms for each tensor in tensors. Note that multiple torchvision.transforms can be combined using
        torchvision.transforms.Compose. If a tensor in the list should not be transformed add None to the list.
        For example, [transform0, None, transform1], will apply the transform0 to the first tensor, the second tensor will not be transformed
        and the third tensor will be transformed with transform1.
    orig_transforms_list : List of torchvision.transforms
        List of torchvision.transforms for each original tensor in tensors, e.g., for preprocessing. If a tensor in the list should not be transformed add None to the list.
    
    Attributes
    ----------
    tensors : torch.Tensor
        tensors that have the same size of the first dimension. Usually contains the data.
    aug_transforms_list : List of torchvision.transforms
    orig_transforms_list : List of torchvision.transforms
    """

    def __init__(self, *tensors: torch.Tensor, aug_transforms_list: List[Callable] = None, orig_transforms_list: List[Callable] = None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        assert len(orig_transforms_list) == tensors.shape[0], "Size mismatch between tensors and orig_transforms_list"
        self.orig_transforms_list = orig_transforms_list
        assert len(aug_transforms_list) == tensors.shape[0], "Size mismatch between tensors and aug_transforms_list"
        self.aug_transforms_list = aug_transforms_list
        

    def __getitem__(self, index: int) -> tuple:
        """
        Get sample at specified index.

        Parameters
        ----------
        index : int
            index of the desired sample

        Returns
        -------
        final_tuple : tuple
            Tuple containing the sample. Consists of (index, data1, data2, ...), depending on the input tensors.
        """
        
        if self.orig_transforms_list is None and self.aug_transforms_list is None:
            final_tuple = tuple([index] + [tensor[index] for tensor in self.tensors])
        else:
            aug_list = []
            for i, tensor in enumerate(self.tensors):
                if self.aug_transforms_list is not None:
                    # apply augmentation
                    aug_transforms_i = self.aug_transforms_list[i]
                    if aug_transforms_i is not None:
                        aug_list.append(aug_transforms_i(tensor[index]))
                
                if self.orig_transforms_list is not None:
                    # apply preprocessing
                    orig_transforms_i = self.orig_transforms_list[i]
                    if orig_transforms_i is None:
                        orig_i = tensor[index]
                    else:
                        orig_i = orig_transforms_i(tensor[index])
                else:
                    orig_i = tensor[index]
                    
                aug_list.append(orig_i)
                    
            final_tuple = tuple([index] + aug_list)
        return final_tuple
    
    def __len__(self) -> int:
        """
        Get length of the dataset which equals the length of the input tensors.

        Returns
        -------
        dataset_size : int
            Length of the dataset.
        """
        dataset_size = self.tensors[0].size(0)
        return dataset_size


def get_dataloader(X: np.ndarray, batch_size: int, shuffle: bool = True, drop_last: bool = False,
                   additional_inputs: list = None, dataset_class: torch.utils.data.Dataset = _ClustpyDataset,
                   ds_kwargs: dict = {}, dl_kwargs: dict = {}) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for Deep Clustering algorithms.
    First entry always contains the indices of the data samples.
    Second entry always contains the actual data samples.
    If for example labels are desired, they can be passed through the additional_inputs parameter (should be a list).
    Other customizations (e.g. augmentation) can be implemented using a custom dataset_class.
    This custom class should stick to the conventions, [index, data, ...].

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the actual data set (can be np.ndarray or torch.Tensor)
    batch_size : int
        the batch size
    shuffle : bool
        boolean that defines if the data set should be shuffled (default: True)
    drop_last : bool
        boolean that defines if the last batch should be ignored (default: False)
    additional_inputs : list / np.ndarray / torch.Tensor
        additional inputs for the dataloader, e.g. labels. Can be None, np.ndarray, torch.Tensor or a list containing np.ndarrays/torch.Tensors (default: None)
    dataset_class : torch.utils.data.Dataset
        defines the class of the tensor dataset that is contained in the dataloader (default: _ClustpyDataset)
    ds_kwargs : any
        other arguments for dataset_class. 
        An example usage would be to include augmentation or preprocessing transforms to the _ClustpyDataset by
        passing ds_kwargs={"aug_transforms_list":[aug_transforms], "orig_transforms_list":[orig_transforms]}, where aug_transforms and orig_transforms
        are transforming the input X, e.g., using torchvision.transforms.Compose to combine multiple transformations.

        Important: If aug_transform_list is passed via ds_kwargs the returned values of the dataloader change. The first entry will still be the indices of the data sample,
                   but the second samples will be the transformed version of the actual data samples and third entry will be the original data samples. 
                   If orig_transforms_list is passed as well then the third entry will be transformed accordingly, this might be needed for preprocessing the data.
                   An example for MNIST is shown below.

    dl_kwargs : any
        other arguments for torch.utils.data.DataLoader

    Examples
    ----------
    >>> # Example for usage of data transformations with get_dataloader
    >>> from clustpy.data import load_mnist
    >>> import torch
    >>> import torchvision

    >>> # load and prepare data for torchvision.transforms   
    >>> data, labels = load_mnist()
    >>> data = data.reshape(-1, 1, 28, 28)
    >>> data /= 255.0
    >>> data = torch.from_numpy(data).float()
    >>> #
    >>> # preprocessing functions
    >>> mean = data.mean()
    >>> std = data.std()
    >>> normalize_fn = torchvision.transforms.Normalize([mean], [std])
    >>> # flatten is only needed if a FeedForward Autoencoder is used, otherwise this can be skipped.
    >>> flatten_fn = torchvision.transforms.Lambda(torch.flatten)
    >>> #
    >>> # augmentation transforms
    >>> transform_list = [
    >>>     # transform input tensor to PIL image for augmentation
    >>>     torchvision.transforms.ToPILImage(),
    >>>     # apply transformations
    >>>     torchvision.transforms.RandomAffine(degrees=(-16,+16),
    >>>                                                 translate=(0.1, 0.1),
    >>>                                                 shear=(-8, 8),
    >>>                                                 fill=0),
    >>>     # transform back to torch.tensor
    >>>     torchvision.transforms.ToTensor(),
    >>>     # preprocess and flatten
    >>>     normalize_fn,
    >>>     flatten_fn,
    >>> ]
    >>> #
    >>> # augmentation transforms
    >>> aug_transforms = torchvision.transforms.Compose(transform_list)
    >>> # preprocessing transforms without augmentation
    >>> orig_transforms = torchvision.transforms.Compose([normalize_fn, flatten_fn])
    >>> #
    >>> # pass transforms to dataloader
    >>> aug_dl = get_dataloader(data, batch_size=32, shuffle=True, 
    >>>                         ds_kwargs={"aug_transforms_list":[aug_transforms], "orig_transforms_list":[orig_transforms]},
    >>>                         )
    
    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        The final dataloader
    """
    assert type(X) in [np.ndarray, torch.Tensor], "X must be of type np.ndarray or torch.Tensor."
    assert additional_inputs is None or type(additional_inputs) in [np.ndarray, torch.Tensor,
                                                                    list], "additional_input must be None or of type np.ndarray, torch.Tensor or list."
    if type(X) is np.ndarray:
        # Convert np.ndarray to torch.Tensor
        X = torch.from_numpy(X).float()
    dataset_input = [X]
    if additional_inputs is not None:
        # Check type of additional_inputs
        if type(additional_inputs) is np.ndarray:
            dataset_input.append(torch.from_numpy(additional_inputs).float())
        elif type(additional_inputs) is torch.Tensor:
            dataset_input.append(additional_inputs)
        else:
            for input in additional_inputs:
                if type(input) is np.ndarray:
                    input = torch.from_numpy(input).float()
                elif type(input) is not torch.Tensor:
                    raise Exception(
                        "inputs of additional_inputs must be of type np.ndarray or torch.Tensor. Your input type: {0}".format(
                            type(input)))
                dataset_input.append(input)
    dataset = dataset_class(*dataset_input, **ds_kwargs)
    # Create dataloader using the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dl_kwargs)
    return dataloader


def augmentation_invariance_check(augmentation_invariance: bool, custom_dataloaders: tuple) -> None:
    """
    Check if the provided custom_dataloaders are compatible with the assumed structure for learning augmentation invariances.

    Parameters
    ----------
    augmentation_invariance : bool
        If True, custom_dataloader will be checked.
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
    """
    if augmentation_invariance and custom_dataloaders is not None:
        # Make sure that two embeddings of the same shape are returned, assuming that one is the augmented tensor and the other the original tensor
        trainloader, testloader = custom_dataloaders
        batch = next(iter(trainloader))
        if len(batch) < 3:
            raise ValueError(f"Augmentation_invariance is True, but custom_dataloaders[0] only returns a list of size {len(batch)} (index, tensor)")
        if not (all(batch[0].size(0) == tensor.size(0) for tensor in batch) and batch[1].shape == batch[2].shape):
            raise ValueError(f"Augmentation_invariance is True, but the shapes of the returned batch of custom_dataloaders[0] do not match.")
        else:
            if torch.equal(batch[1], batch[2]):
                raise ValueError(f"Augmentation_invariance is True, but custom_dataloaders[0] returns identical tensors in batch[1] and batch[2] indicating that no augmentation is applied to batch[1]")
    elif augmentation_invariance and custom_dataloaders is None:
        raise ValueError("If augmentation_invariance is True, custom_dataloaders cannot be None, but should include augmented samples, e.g., using torchvision.transforms in get_dataloader.")

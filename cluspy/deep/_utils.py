import torch
from itertools import islice
from cluspy.deep.simple_autoencoder import Simple_Autoencoder
import numpy as np


class EarlyStopping():
    """Early stopping to stop the training when the loss does not improve after
    certain epochs. Adapted from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    
    Parameters
    ----------
    patience : int, default=10, how many epochs to wait before stopping when loss is not improving
    min_delta : float, default=1e-4, minimum difference between new loss and old loss for new loss to be considered as an improvement
    verbose : bool, default=False, if True will print INFO statements
    
    Attributes
    ----------
    counter : integer counting the consecutive epochs without improvement
    best_loss : best loss achieved before stopping
    early_stop : boolean indicating whether to stop training or not
    """

    def __init__(self, patience=10, min_delta=1e-4, verbose=False):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def squared_euclidean_distance(centers, embedded, weights=None):
    ta = centers.unsqueeze(0)
    tb = embedded.unsqueeze(1)
    squared_diffs = (ta - tb)
    if weights is not None:
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).sum(2)  # .mean(2) # TODO Evaluate this change
    return squared_diffs


def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


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


def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def predict_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        prediction = cluster_module.prediction_hard(model.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()


def window(seq, n):
    """Returns a sliding window (of width n) over data from the following iterable:
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ..."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


# def add_noise(batch):
#     mask = torch.empty(
#         batch.shape, device=batch.device).bernoulli_(0.8)
#     return batch * mask


def int_to_one_hot(label_tensor, n_labels):
    onehot = torch.zeros([label_tensor.shape[0], n_labels], dtype=torch.float, device=label_tensor.device)
    onehot.scatter_(1, label_tensor.unsqueeze(1).long(), 1.0)
    return onehot


def get_trained_autoencoder(trainloader, learning_rate, n_epochs, device, optimizer_class, loss_fn,
                            input_dim, embedding_size, autoencoder_class=Simple_Autoencoder):
    if embedding_size > input_dim:
        print(
            "WARNING: embedding_size is larger than the dimensionality of the input dataset. Setting embedding_size to",
            input_dim)
        embedding_size = input_dim
    # Pretrain Autoencoder
    autoencoder = autoencoder_class(input_dim=input_dim, embedding_size=embedding_size).to(device)
    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
    autoencoder.start_training(trainloader, n_epochs, device, optimizer, loss_fn)
    return autoencoder

import torch
from itertools import islice
from .simple_autoencoder import Simple_Autoencoder


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


def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    for batch in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def predict_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch.to(device)
        prediction = cluster_module.prediction_hard(model.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()


def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
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

import torch
from clustpy.data import load_nrletters


def _load_single_label_nrletters():
    X, L = load_nrletters()
    L = L[:, 0]
    return X, L


def _get_test_dataloader(data, batch_size, shuffle, drop_last):
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.arange(0, data.shape[0]), torch.from_numpy(data).float())),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last)
    return dataloader


class _TestAutoencoder(torch.nn.Module):
    """
    A simple autoencoder only for test purposes.
    Encoder layers: [input_dim, embedding]
    Decoder layers: [embedding, input_dim]
    All weights are initialized as 1. Fitting function only sets fitting=True (no updates of the weights).
    """

    def __init__(self, input_dim, embedding_dim):
        super(_TestAutoencoder, self).__init__()
        self.encoder = torch.nn.Linear(input_dim, embedding_dim, bias=False)
        self.encoder.weight.data.fill_(1)
        self.decoder = torch.nn.Linear(embedding_dim, input_dim, bias=False)
        self.decoder.weight.data.fill_(1)
        self.fitted = False

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def loss(self, batch_data, loss_fn):
        reconstruction = self.forward(batch_data)
        loss = loss_fn(reconstruction, batch_data)
        return loss

    def fit(self):
        self.fitted = True
        return self


class _TestClusterModule(torch.nn.Module):
    """
    A simple cluster module to test predict-related methods.
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def predict_hard(self, embedded, weights=None) -> torch.Tensor:
        """
        Hard prediction of given embedded samples. Returns the corresponding hard labels.
        Predicts 1 for all samples with mean(features) >= threshold and 0 for mean(features) < threshold.
        """
        predictions = (torch.mean(embedded, 1) >= self.threshold) * 1
        return predictions

import torch


def squared_euclidean_distance(centers, embedded, weights=None):
    ta = centers.unsqueeze(0)
    tb = embedded.unsqueeze(1)
    squared_diffs = (ta - tb)
    if weights is not None:
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).mean(2)
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


def _get_trained_autoencoder(trainloader, learning_rate, pretrain_iterations, device, optimizer_class, loss_fn,
                             input_dim, embedding_size):
    # Pretrain Autoencoder
    autoencoder = Autoencoder(input_dim = input_dim, embedding_size = embedding_size)
    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
    autoencoder.pretrain(trainloader, pretrain_iterations, device, optimizer, loss_fn)
    return autoencoder


class Autoencoder(torch.nn.Module):
    """A vanilla symmetric autoencoder.

    Args:
        input_dim: size of each input sample
        embedding_size: size of the inner most layer also called embedding

    Attributes:
        encoder: encoder part of the autoencoder, responsible for embedding data points
        decoder: decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    """

    def __init__(self, input_dim: int, embedding_size: int):
        super(Autoencoder, self).__init__()

        # make a sequential list of all operations you want to apply for encoding a data point
        self.encoder = torch.nn.Sequential(
            # Linear layer (just a matrix multiplication)
            torch.nn.Linear(input_dim, 256),
            # apply an elementwise non-linear function
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(64, embedding_size))

        # make a sequential list of all operations you want to apply for decoding a data point
        # In our case this is a symmetric version of the encoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points

        Returns:
            embedded: the embedded data point with dimensionality embedding_size
        """
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedded: embedded data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Applies both encode and decode function.
        The forward function is automatically called if we call self(x).
        Args:
            x: input data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def pretrain(self, trainloader, training_iterations, device, optimizer, loss_fn):
        # load model to device
        self.to(device)
        i = 0
        while (i < training_iterations):
            for batch in trainloader:
                # load batch on device
                batch_data = batch.to(device)
                reconstruction = self.forward(batch_data)
                loss = loss_fn(reconstruction, batch_data)
                # reset gradients from last iteration
                optimizer.zero_grad()
                # calculate gradients and reset the computation graph
                loss.backward()
                # update the internal params (weights, etc.)
                optimizer.step()
                if i > training_iterations:
                    break
                i += 1

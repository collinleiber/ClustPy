import torch


class Simple_Autoencoder(torch.nn.Module):
    """A vanilla symmetric autoencoder.

    Args:
        input_dim: size of each input sample
        embedding_size: size of the inner most layer also called embedding

    Attributes:
        encoder: encoder part of the autoencoder, responsible for embedding data points
        decoder: decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    """

    def __init__(self, input_dim: int, embedding_size: int, small_network=False):
        super(Simple_Autoencoder, self).__init__()

        if small_network:
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
        else:
            # make a sequential list of all operations you want to apply for encoding a data point
            self.encoder = torch.nn.Sequential(
                # Linear layer (just a matrix multiplication)
                torch.nn.Linear(input_dim, 500),
                # apply an elementwise non-linear function
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, 500),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, 2000),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(2000, embedding_size))

            # make a sequential list of all operations you want to apply for decoding a data point
            # In our case this is a symmetric version of the encoder
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, 2000),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(2000, 500),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, 500),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, input_dim),
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

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        for _ in range(n_epochs):
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

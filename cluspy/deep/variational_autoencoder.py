import torch
from cluspy.deep._data_utils import get_dataloader
from cluspy.deep.flexible_autoencoder import FullyConnectedBlock


class VariationalAutoencoder(torch.nn.Module):
    """A variational autoencoder (VAE).

    Parameters
    ----------
    layers : list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the embedding dimension.
             If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    batch_norm : bool, default=False, set True if you want to use torch.nn.BatchNorm1d
    dropout : float, default=None, set the amount of dropout you want to use.
    activation: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers, if None then it will be linear.
    bias : bool, default=True, set False if you do not want to use a bias term in the linear layers
    decoder_layers : list, default=None, list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers.
    decoder_output_fn : activation function from torch.nn, default=None, set the activation function for the decoder output layer, if None then it will be linear.
                        e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1.
    Attributes
    ----------
    encoder : encoder part of the autoencoder, responsible for embedding data points
    decoder : decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    fitted  : boolean value indicating whether the autoencoder is already fitted.
    """

    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU, bias=True,
                 decoder_layers=None, decoder_output_fn=torch.nn.Sigmoid):
        super(VariationalAutoencoder, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bias = bias
        self.decoder_output_fn = decoder_output_fn
        self.fitted = False
        if decoder_layers is None:
            self.decoder_layers = self.layers[::-1]
        else:
            self.decoder_layers = decoder_layers
        if (self.layers[-1] != self.decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {self.layers[-1]} and {self.decoder_layers[0]} respectively.")
        if (self.layers[0] != self.decoder_layers[-1]):
            raise ValueError(
                f"Output and input dimension do not match, they are {self.layers[0]} and {self.decoder_layers[-1]} respectively.")

        # Initialize encoder
        self.encoder = FullyConnectedBlock(layers=self.layers[:-1], batch_norm=self.batch_norm, dropout=self.dropout,
                                           activation_fn=self.activation_fn, bias=self.bias,
                                           output_fn=self.activation_fn)
        # naming is used for later correspondence in VaDE
        self.mean_ = torch.nn.Linear(self.layers[-2], self.layers[-1])
        # is only initialized
        self.variance_ = torch.nn.Linear(self.layers[-2], self.layers[-1])

        self.classify = torch.nn.LogSoftmax(dim=1)
        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(layers=self.decoder_layers, batch_norm=self.batch_norm, dropout=self.dropout,
                                           activation_fn=self.activation_fn, bias=self.bias,
                                           output_fn=self.decoder_output_fn)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        embedded = self.encoder(x)
        q_mean = self.mean_(embedded)
        q_var = self.variance_(embedded)
        return q_mean, q_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q_mean, q_var = self.encode(x)
        reconstruction = self.decode(q_mean)
        return q_mean, reconstruction

    def fit(self, n_epochs, lr, batch_size=128, data=None, dataloader=None, optimizer_class=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss(), device=torch.device("cpu"), model_path=None, print_step=0):
        """Trains the autoencoder in place.

        Parameters
        ----------
        n_epochs : int, number of epochs for training
        lr : float, learning rate to be used for the optimizer_class
        batch_size : int, default=128
        data : np.ndarray, default=None, train data set. If data is passed then dataloader can remain empty
        dataloader : torch.utils.data.DataLoader, default=None, dataloader to be used for training
        optimizer_class : torch.optim, default=torch.optim.Adam, optimizer to be used
        loss_fn : torch.nn, default=torch.nn.MSELoss(), loss function to be used for reconstruction
        device : torch.device, default=torch.device('cpu'), device to be trained on
        model_path : str, default=None, if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved.
        print_step : int, default=0, specifies how often the losses are printed. If 0, no prints will occur

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        params_dict = {'params': self.parameters(), 'lr': lr}
        optimizer = optimizer_class(**params_dict)
        # training loop
        for epoch_i in range(n_epochs):
            self.train()
            for batch in dataloader:
                # load batch on device
                batch_data = batch[1].to(device)
                q_mean, reconstruction = self.forward(batch_data)
                loss = loss_fn(reconstruction, batch_data) / batch_data.size(0)
                # reset gradients from last iteration
                optimizer.zero_grad()
                # calculate gradients and reset the computation graph
                loss.backward()
                # update the internal params (weights, etc.)
                optimizer.step()
                if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                    print(f"Epoch {epoch_i}/{n_epochs - 1} - Batch Reconstruction loss: {loss.item():.6f}")
        # Save last version of model
        if model_path is not None:
            torch.save(self.state_dict(), model_path)

        # Autoencoder is now pretrained
        self.fitted = True
        return self

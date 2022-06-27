import torch
from cluspy.deep._data_utils import get_dataloader
from cluspy.deep.flexible_autoencoder import FullyConnectedBlock
from cluspy.deep._early_stopping import EarlyStopping
import numpy as np


def _compute_loss(pi, p_mean, p_var, q_mean, q_var, batch_data, p_c_z, reconstruction, loss_fn):
    q_mean = q_mean.unsqueeze(1)
    p_var = p_var.unsqueeze(0)

    p_x_z = loss_fn(reconstruction, batch_data)

    p_z_c = torch.sum(p_c_z * (0.5 * np.log(2 * np.pi) + 0.5 * (
            torch.sum(p_var, dim=2) + torch.sum(torch.exp(q_var.unsqueeze(1)) / torch.exp(p_var),
                                                dim=2) + torch.sum((q_mean - p_mean).pow(2) / torch.exp(p_var),
                                                                   dim=2))))
    p_c = torch.sum(p_c_z * torch.log(pi))
    q_z_x = 0.5 * (np.log(2 * np.pi)) + 0.5 * torch.sum(1 + q_var)
    q_c_x = torch.sum(p_c_z * torch.log(p_c_z))

    loss = p_x_z + p_z_c - p_c - q_z_x + q_c_x
    loss /= batch_data.size(0)
    return loss


def _sampling(q_mean, q_var):
    std = torch.exp(0.5 * q_var)
    eps = torch.randn_like(std)
    z = q_mean + eps * std
    return z


def _get_gamma(pi, p_mean, p_var, z):
    z = z.unsqueeze(1)
    p_var = p_var.unsqueeze(0)
    pi = pi.unsqueeze(0)

    p_z_c = -torch.sum(0.5 * (np.log(2 * np.pi)) + p_var + ((z - p_mean).pow(2) / (2. * torch.exp(p_var))), dim=2)
    p_c_z_c = torch.exp(torch.log(pi) + p_z_c) + 1e-10
    p_c_z = p_c_z_c / torch.sum(p_c_z_c, dim=1, keepdim=True)

    return p_c_z


class VariationalAutoencoder(torch.nn.Module):
    """
    A variational autoencoder (VAE).

    Parameters
    ----------
    layers : list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the dimension of the central mean and variance value.
             If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    n_distributions : number of distributions in the VAE.
    batch_norm : bool, default=False, set True if you want to use torch.nn.BatchNorm1d.
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

    def __init__(self, layers, n_distributions, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU,
                 bias=True, decoder_layers=None, decoder_output_fn=torch.nn.Sigmoid):
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
        # Get size of embedding from last dimension of layers
        embedding_size = self.layers[-1]
        # Initialize encoder
        self.encoder = FullyConnectedBlock(layers=self.layers[:-1], batch_norm=self.batch_norm, dropout=self.dropout,
                                           activation_fn=self.activation_fn, bias=self.bias,
                                           output_fn=self.activation_fn)
        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(layers=self.decoder_layers, batch_norm=self.batch_norm, dropout=self.dropout,
                                           activation_fn=self.activation_fn, bias=self.bias,
                                           output_fn=self.decoder_output_fn)

        self.q_mean_ = torch.nn.Linear(self.layers[-2], embedding_size)
        self.q_variance_ = torch.nn.Linear(self.layers[-2], embedding_size)

        self.pi = torch.nn.Parameter(torch.ones(n_distributions) / self.layers[-1], requires_grad=True)
        self.p_mean = torch.nn.Parameter(torch.randn(n_distributions, embedding_size),
                                         requires_grad=True)  # if not initialized then use torch.randn
        self.p_var = torch.nn.Parameter(torch.ones(n_distributions, embedding_size), requires_grad=True)
        self.normalize_prob = torch.nn.Softmax(dim=0)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        x : input data point, can also be a mini-batch of points

        Returns
        -------
        q_mean : mean value of the central VAE layer
        q_var : variance value of the central VAE layer
        """
        embedded = self.encoder(x)
        q_mean = self.q_mean_(embedded)
        q_var = self.q_variance_(embedded)
        return q_mean, q_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z: embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """ Applies both encode and decode function.
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : input data point, can also be a mini-batch of embedded points

        Returns
        -------
        z : sampling using q_mean and q_var
        q_mean : mean value of the central VAE layer
        q_var : variance value of the central VAE layer
        reconstruction: returns the reconstruction of a data point
        """
        q_mean, q_var = self.encode(x)
        z = _sampling(q_mean, q_var)
        reconstruction = self.decode(z)
        return z, q_mean, q_var, reconstruction

    def predict(self, q_mean, q_var) -> torch.Tensor:
        z = _sampling(q_mean, q_var)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        pred = torch.argmax(p_c_z, dim=1)
        return pred

    def vae_loss(self, batch_data, reconstruction, q_mean, q_var, loss_fn) -> torch.Tensor:
        z = _sampling(q_mean, q_var)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        loss = _compute_loss(pi_normalized, self.p_mean, self.p_var, q_mean, q_var, batch_data, p_c_z, reconstruction,
                             loss_fn)
        return loss

    def evaluate(self, dataloader, loss_fn, device=torch.device("cpu")):
        """
        Evaluates the VAE.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader, dataloader to be used for training
        loss_fn : torch.nn, loss function to be used for reconstruction
        device : torch.device, default=torch.device('cpu'), device to be trained on

        Returns
        -------
        loss: returns the reconstruction loss of all samples in dataloader
        """
        with torch.no_grad():
            self.eval()
            loss = 0
            for batch in dataloader:
                batch_data = batch[1].to(device)
                _, q_mean, q_var, reconstruction = self.forward(batch_data)
                loss += self.vae_loss(batch_data, reconstruction, q_mean, q_var, loss_fn)
            loss /= len(dataloader)
        return loss

    def fit(self, n_epochs, lr, batch_size=128, data=None, data_eval=None, dataloader=None, evalloader=None,
            optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(), patience=5, scheduler=None,
            scheduler_params=None, device=torch.device("cpu"), model_path=None, print_step=0):
        """
        Trains the VAE in place.

        Parameters
        ----------
        n_epochs : int, number of epochs for training
        lr : float, learning rate to be used for the optimizer_class
        batch_size : int, default=128
        data : np.ndarray, default=None, train data set. If data is passed then dataloader can remain empty
        data_eval : np.ndarray, default=None, evaluation data set. If data_eval is passed then evalloader can remain empty.
        dataloader : torch.utils.data.DataLoader, default=None, dataloader to be used for training
        evalloader : torch.utils.data.DataLoader, default=None, dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        optimizer_class : torch.optim, default=torch.optim.Adam, optimizer to be used
        loss_fn : torch.nn, default=torch.nn.MSELoss(), loss function to be used for reconstruction
        patience : int, default=5, patience parameter for EarlyStopping
        scheduler : torch.optim.lr_scheduler, default=None, learning rate scheduler that should be used.
                    If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader.
        scheduler_params : dict, default=None, dictionary of the parameters of the scheduler object
        device : torch.device, default=torch.device('cpu'), device to be trained on
        model_path : str, default=None, if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved.
        print_step : int, default=0, specifies how often the losses are printed. If 0, no prints will occur

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        # evalloader has priority over data_eval
        if evalloader is None:
            if data_eval is not None:
                evalloader = get_dataloader(data_eval, batch_size, False)
        params_dict = {'params': self.parameters(), 'lr': lr}
        optimizer = optimizer_class(**params_dict)

        early_stopping = EarlyStopping(patience=patience)
        if scheduler is not None:
            scheduler = scheduler(optimizer=optimizer, **scheduler_params)
            # Depending on the scheduler type we need a different step function call.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_step_scheduler = True
                if evalloader is None:
                    raise ValueError(
                        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, but evalloader is None. Specify evalloader such that validation loss can be computed.")
            else:
                eval_step_scheduler = False
        best_loss = np.inf
        # training loop
        for epoch_i in range(n_epochs):
            self.train()
            for batch in dataloader:
                batch_data = batch[1].to(device)
                _, q_mean, q_var, reconstruction = self.forward(batch_data)
                loss = self.vae_loss(batch_data, reconstruction, q_mean, q_var, loss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                print(f"Epoch {epoch_i}/{n_epochs - 1} - Batch Reconstruction loss: {loss.item():.6f}")

            if scheduler is not None and not eval_step_scheduler:
                scheduler.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, loss_fn=loss_fn, device=device)
                if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                    print(f"Epoch {epoch_i} EVAL loss total: {val_loss.item():.6f}")
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)
                if early_stopping.early_stop:
                    if print_step > 0:
                        print(f"Stop training at epoch {best_epoch}")
                        print(f"Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                    break
                if scheduler is not None and eval_step_scheduler:
                    scheduler.step(val_loss)
        # Save last version of model
        if evalloader is None and model_path is not None:
            torch.save(self.state_dict(), model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self

import torch
from torch import nn
import numpy as np
from ._utils import EarlyStopping

class FullyConnectedBlock(nn.Module):
    """Feed Forward Neural Network Block

    Parameters
    ----------
    layers : list of the different layer sizes 
    batch_norm : bool, default=False, set True if you want to use torch.nn.BatchNorm1d
    dropout : float, default=None, set the amount of dropout you want to use.
    activation : activation function from torch.nn, default=None, set the activation function for the hidden layers, if None then it will be linear. 
    bias : bool, default=True, set False if you do not want to use a bias term in the linear layers
    output_fn : activation function from torch.nn, default=None, set the activation function for the last layer, if None then it will be linear. 

    Attributes
    ----------
    block: torch.nn.Sequential, feed forward neural network
    """
    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=None, bias=True, output_fn=None):
        super(FullyConnectedBlock, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bias = bias
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        fc_block_list = []
        for i in range(len(layers)-1):
            fc_block_list.append(nn.Linear(layers[i], layers[i+1], bias=self.bias))
            if self.batch_norm:
                fc_block_list.append(nn.BatchNorm1d(layers[i+1]))
            if self.dropout is not None:
                fc_block_list.append(nn.Dropout(self.dropout))
            if self.activation_fn is not None:
                # last layer is handled differently
                if (i != len(layers)-2):
                    fc_block_list.append(activation_fn())
                else:
                    if self.output_fn is not None:
                        fc_block_list.append(self.output_fn())

        self.block =  nn.Sequential(*fc_block_list)
    
    def forward(self, x):
        return self.block(x)

class FlexibleAutoencoder(torch.nn.Module):
    """A feedforward symmetric autoencoder.
    
    Parameters
    ----------
    layers : list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the embedding dimension. 
             The decoder is symmetric and goes in the same order from embedding to input.
    batch_norm : bool, default=False, set True if you want to use torch.nn.BatchNorm1d
    dropout : float, default=None, set the amount of dropout you want to use.
    activation: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers, if None then it will be linear. 
    bias : bool, default=True, set False if you do not want to use a bias term in the linear layers
    decoder_output_fn : activation function from torch.nn, default=None, set the activation function for the decoder output layer, if None then it will be linear. 
                        e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1. 
    Attributes
    ----------
    encoder : encoder part of the autoencoder, responsible for embedding data points
    decoder : decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    """
    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU, bias=True, decoder_output_fn=None):
        super(FlexibleAutoencoder, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bias = bias
        self.decoder_output_fn = decoder_output_fn
        
        # Initialize encoder
        self.encoder = FullyConnectedBlock(layers=self.layers, batch_norm=self.batch_norm, dropout=self.dropout, activation_fn=self.activation_fn, bias=self.bias, output_fn=None)
        
        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(layers=self.layers[::-1], batch_norm=self.batch_norm, dropout=self.dropout, activation_fn=self.activation_fn, bias=self.bias, output_fn=self.decoder_output_fn)

    
    def encode(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        x : input data point, can also be a mini-batch of points
        
        Returns
        -------
        embedded : the embedded data point with dimensionality embedding_size
        """
        return self.encoder(x)
    
    def decode(self, embedded:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        embedded: embedded data point, can also be a mini-batch of embedded points
        
        Returns
        -------
        reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(embedded)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """ Applies both encode and decode function. 
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : input data point, can also be a mini-batch of embedded points
        
        Returns
        -------
        reconstruction: returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction
    
    def evaluate(self, dataloader, loss_fn, device=torch.device("cpu")):
        """Evaluates the autoencoder.
        
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
                batch = batch[0].to(device)
                reconstruction = self(batch)
                loss += loss_fn(reconstruction, batch)
            loss /= len(dataloader)
        return loss

    def fit(self, n_epochs, lr, batch_size=128, data=None, data_eval=None, dataloader=None, evalloader=None, optimizer_fn=torch.optim.Adam, loss_fn=torch.nn.MSELoss(), patience=5, lr_reduction_factor=0.5, device=torch.device("cpu"), model_path=None, print_step=10):
        """Trains the autoencoder in place.
        
        Parameters
        ----------
        n_epochs : int, number of epochs for training
        lr : float, learning rate to be used for the optimizer_fn
        batch_size : int, default=128
        data : np.ndarray, default=None, train data set. If data is passed then dataloader can remain empty
        data_eval : np.ndarray, default=None, evaluation data set. If data_eval is passed then evalloader can remain empty.
        dataloader : torch.utils.data.DataLoader, default=None, dataloader to be used for training
        evalloader : torch.utils.data.DataLoader, default=None, dataloader to be used for evaluation, early stopping and learning rate scheduling
        optimizer_fn : torch.optim, default=torch.optim.Adam, optimizer to be used
        loss_fn : torch.nn, default=torch.nn.MSELoss(), loss function to be used for reconstruction
        patience : int, default=5, patience parameter for learning rate scheduler and early stopping
        lr_reduction_factor : float, default=0.5, factor for reducing the learning rate if loss plateu is reached, if None or 0 than learning rate scheduler is not used. Based on torch.optim.lr_scheduler.ReduceLROnPlateau.
        device : torch.device, default=torch.device('cpu'), device to be trained on
        model_path : str, default=None, if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved.
        print_step : int, default=10, specifies how often the losses are printed
        
        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(data).float()),
                                                    batch_size=batch_size,
                                                    shuffle=True)
        # evalloader has priority
        if evalloader is None:
            if data_eval is not None:
                evalloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(data_eval).float()),
                                                        batch_size=batch_size,
                                                        shuffle=False)
        
        params_dict = {'params': self.parameters(), 'lr': lr}
        optimizer = optimizer_fn(**params_dict)
        
        early_stopping = EarlyStopping(patience=patience)
        if (lr_reduction_factor is not None) and (lr_reduction_factor > 0):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_reduction_factor, patience=patience, verbose=True)
        else:
            scheduler = None
        best_loss = np.inf
        i = 0
        # training loop
        for epoch_i in range(n_epochs):
            self.train()
            for batch in dataloader:
                batch = batch[0].to(device)
                reconstruction = self(batch)
                loss = loss_fn(reconstruction, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch_i-1) % print_step == 0 or epoch_i == n_epochs:
                print(f"Epoch {epoch_i-1}/{n_epochs} - Batch Reconstruction loss: {loss.item():.6f}")
            
            # Evaluate autoencoder
            if evalloader is not None:
                self.eval()
                val_loss = self.evaluate(dataloader=evalloader, loss_fn=loss_fn, device=device)
                if (epoch_i-1) % print_step == 0 or epoch_i == n_epochs:
                    print(f"Epoch {epoch_i-1} EVAL loss total: {val_loss.item():.6f}")
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)

                if early_stopping.early_stop:
                    print(f"Stop training at epoch {best_epoch-1}")
                    print(f"Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                    break
                if scheduler is not None:
                    scheduler.step(val_loss)
        # Save last version of model
        if evalloader is None:
            if model_path is not None:
                torch.save(self.state_dict(), model_path)

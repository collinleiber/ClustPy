from collections.abc import Callable
import tqdm
import torch
import numpy as np
from pathlib import Path
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._early_stopping import EarlyStopping
from clustpy.deep._utils import get_device_from_module, mean_squared_error, set_torch_seed
from clustpy.utils.checks import check_parameters, check_random_state


class _AbstractNeuralNetwork(torch.nn.Module):
    """
    An abstract neural network class that can be used by other neural network implementations.

    Parameters
    ----------
    work_on_copy : bool
        If set to true, deep clustering algorithms will optimize a copy of the neural network and not the neural network itself.
        Ensures that the same neural network can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    random_state : np.random.RandomState | int | None
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)


    Attributes
    ----------
    fitted : bool
        indicates whether the neural network is already fitted
    allow_nd_input : bool
        indicates whether the neural network can handle n-dimensional input data (default: False)
    """

    def __init__(self, work_on_copy: bool = True, random_state: np.random.RandomState | int | None = None):
        super(_AbstractNeuralNetwork, self).__init__()
        self.work_on_copy = work_on_copy
        self.random_state = random_state
        self.fitted = False
        self.allow_nd_input = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for an encode function of a neural network.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        x : torch.Tensor
            should return the embedded data point
        """
        return x

    def loss(self, batch: list, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss, device: torch.device,
             corruption_fn: Callable | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss of a single batch of data.

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, samples, ...)
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on
        corruption_fn : Callable | None
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)

        Returns
        -------
        loss : tuple[torch.Tensor, torch.Tensor]
            the reconstruction loss of the input sample,
            the embedded input sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        raise NotImplementedError("The loss function of the _AbstractNeuralNetwork is only a placeholder and has to be overwriten by child classes!")

    def loss_augmentation(self, batch: list, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss, device: torch.device,
                          corruption_fn: Callable | None = None) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss of a single batch of data and an augmented version of the data.
        Note that the augmented samples come at position batch[1] and the original samples at batch[2].

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, augmented samples, original samples, ...)
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on
        corruption_fn : Callable | None
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)

        Returns
        -------
        loss : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the combined network loss of the sample and the augmented sample,
            the embedded input sample,
            the embedded augmented sample
        """
        assert type(batch) is list, "batch must come from a dataloader and therefore be of type list"
        # First entry (batch[0]) are the indices, second entry (batch[1]) are the augmented samples, third entry (batch[2]) are the original samples
        # If additional inputs are used in the dataloader, entries at an uneven position (batch[3], batch[5], ...) are augmented and entries at even positions (batch[4], batch[6], ...) original
        # Considering also the additional inputs can be relevant, e.g., when using a NeighborEncoder
        batches_orig = [batch[i] for i in range(2, len(batch), 2)]
        batches_aug = [batch[i] for i in range(1, len(batch), 2)]
        loss_orig, embedded = self.loss([batch[0]] + batches_orig, ssl_loss_fn, device, corruption_fn)
        loss_augmented, embedded_aug = self.loss([batch[0]] + batches_aug, ssl_loss_fn, device,
                                                                    corruption_fn)
        loss_total = (loss_orig + loss_augmented) / 2
        return loss_total, embedded, embedded_aug

    def evaluate(self, dataloader: torch.utils.data.DataLoader, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
                 device: torch.device) -> torch.Tensor:
        """
        Evaluates the neural network.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss
        device : torch.device
            device to be trained on

        Returns
        -------
        loss: torch.Tensor
            returns the reconstruction loss of all samples in dataloader
        """
        with torch.no_grad():
            self.eval()
            loss = torch.tensor(0.)
            for batch in dataloader:
                new_loss, _ = self.loss(batch, ssl_loss_fn, device)
                loss += new_loss
            loss /= len(dataloader)
        return loss

    def fit(self, n_epochs: int = 100, optimizer_params: dict | None = None, batch_size: int = 128,
            data: np.ndarray | torch.Tensor | None = None, data_eval: np.ndarray | torch.Tensor | None = None,
            dataloader: torch.utils.data.DataLoader | None = None, evalloader: torch.utils.data.DataLoader | None = None,
            optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
            ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error, patience: int = 5,
            scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None, scheduler_params: dict | None = None,
            corruption_fn: Callable | None = None, model_path: str | None = None) -> '_AbstractNeuralNetwork':
        """
        Trains the neural network in place.

        Parameters
        ----------
        n_epochs : int
            number of epochs for training (default: 100)
        optimizer_params : dict | None
            parameters of the optimizer, includes the learning rate (default: {"lr": 1e-3})
        batch_size : int
            size of the data batches (default: 128)
        data : np.ndarray | torch.Tensor | None
            train data set. If data is passed then dataloader can remain empty (default: None)
        data_eval : np.ndarray | torch.Tensor | None
            evaluation data set. If data_eval is passed then evalloader can remain empty (default: None)
        dataloader : torch.utils.data.DataLoader | None
            dataloader to be used for training (default: default=None)
        evalloader : torch.utils.data.DataLoader | None
            dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau (default: None)
        optimizer_class : type[torch.optim.Optimizer]
            optimizer to be used (default: torch.optim.Adam)
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss (default: mean_squared_error)
        patience : int
            patience parameter for EarlyStopping (default: 5)
        scheduler : type[torch.optim.lr_scheduler.LRScheduler] | None
            learning rate scheduler that should be used.
            If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader (default: None)
        scheduler_params : dict | None
            dictionary of the parameters of the scheduler object. If None it will be empty (default: None)
        corruption_fn : Callable | None
            Can be used to corrupt the input data, e.g., when using a denoising autoencoder.
            Note that the function must match the data and the data loaders.
            For example, if the data is normalized, this may have to be taken into account in the corruption function - e.g. in case of salt and pepper noise (default: None)
        model_path : str | None
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)

        Returns
        -------
        self : _AbstractNeuralNetwork
            this instance of the neural network

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        random_state = check_random_state(self.random_state)
        set_torch_seed(random_state)
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        # evalloader has priority over data_eval
        if evalloader is None:
            if data_eval is not None:
                evalloader = get_dataloader(data_eval, batch_size, False)
        optimizer_params_adj = {"lr": 1e-3} if optimizer_params is None else optimizer_params
        optimizer = optimizer_class(params=self.parameters(), **optimizer_params_adj)

        scheduler_params_adj = {} if scheduler_params is None else scheduler_params

        early_stopping = EarlyStopping(patience=patience)
        if scheduler is not None:
            scheduler_obj = scheduler(optimizer=optimizer, **scheduler_params_adj)
            # Depending on the scheduler type we need a different step function call.
            if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_step_scheduler = True
                if evalloader is None:
                    raise ValueError(
                        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, but evalloader is None. Specify evalloader such that validation loss can be computed.")
            else:
                eval_step_scheduler = False
        else:
            scheduler_obj = None
        best_loss = torch.tensor(np.inf)
        # training loop
        device = get_device_from_module(self)
        tbar = tqdm.trange(n_epochs, desc="AE training")
        for epoch_i in tbar:
            self.train()
            total_loss = 0.
            for batch in dataloader:
                loss, _ = self.loss(batch, ssl_loss_fn, device, corruption_fn)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Training Loss": total_loss}

            if scheduler_obj is not None and not eval_step_scheduler:
                scheduler_obj.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, ssl_loss_fn=ssl_loss_fn, device=device)
                postfix_str["Eval Loss"] = val_loss.item()
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        self.save_parameters(model_path)
                if early_stopping.early_stop:
                    print(f"Stop training at epoch {best_epoch}. Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                if scheduler_obj is not None and isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_obj.step(val_loss)
            tbar.set_postfix(postfix_str)
        # change to eval mode after training
        self.eval()
        # Save last version of model
        if evalloader is None and model_path is not None:
            self.save_parameters(model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self

    def save_parameters(self, path: str | Path) -> None:
        """
        Save the current state_dict of the model.

        Parameters
        ----------
        path : str | Path
            Path where the state_dict should be stored
        """
        # Check if directory exists
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_parameters(self, path: str | Path) -> '_AbstractNeuralNetwork':
        """
        Load a state_dict into the current model to set its parameters.

        Parameters
        ----------
        path : str | Path
            Path from where the state_dict should be loaded

        Returns
        -------
        self : _AbstractNeuralNetwork
            this instance of the neural network
        """
        self.load_state_dict(torch.load(path, weights_only=True, map_location=get_device_from_module(self)))
        self.eval()
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Embed the given data set using the trained neural network.

        Parameters
        ----------
        X: np.ndarray
            The given data set

        Returns
        -------
        X_embed : np.ndarray
            The embedded data set
        """
        if not self.fitted:
            raise ValueError("The neural network is not fitted yet. Run fit() first.")
        X, _, _ = check_parameters(X, allow_size_1=True, allow_nd=self.allow_nd_input)
        device = get_device_from_module(self)
        torch_data = torch.from_numpy(X).float().to(device)
        embedded_data = self.encode(torch_data)
        X_embed = embedded_data.detach().cpu().numpy()
        return X_embed.astype(X.dtype)

from cluspy.deep.flexible_autoencoder import FlexibleAutoencoder

def get_trained_autoencoder(trainloader, learning_rate, n_epochs, device, optimizer_class, loss_fn,
                            input_dim, embedding_size, autoencoder=None, autoencoder_class=FlexibleAutoencoder):
    """This function returns a trained autoencoder. The following cases are considered
       - If the autoencoder is initialized and trained (autoencoder.fitted==True), then return input autoencoder without training it again.
       - If the autoencoder is initialized and not trained (autoencoder.fitted==False), it will be fitted (autoencoder.fitted will be set to True) using default parameters.
       - If the autoencoder is None, a new autoencoder is created using autoencoder_class, and it will be fitted as described above.
       Beware the input autoencoder_class or autoencoder object needs both a fit() function and the fitted attribute. See cluspy.deep.flexible_autoencoder.FlexibleAutoencoder for an example.

    Parameters
    ----------
    trainloader, torch.utils.data.DataLoader, dataloader used to train autoencoder
    learning_rate : float, learning rate for the autoencoder training
    n_epochs : int, number of training epochs
    device : torch.device, device to be trained on
    optimizer_class : torch.optim, optimizer for training.
    loss_fn : torch.nn, loss function for the reconstruction.
    input_dim : int, input dimension of the first layer of the autoencoder
    embedding_size : int, dimension of the innermost layer of the autoencoder
    autoencoder : torch.nn.Module, default=None, autoencoder object to be trained (optional)
    autoencoder_class : torch.nn.Module, default=FlexibleAutoencoder, FlexibleAutoencoder class used if autoencoder=None (optional)
    
    Returns
    -------
    fitted autoencoder
    """
    if autoencoder is None:
       # Train new autoencoder
        if embedding_size > input_dim:
            print(
                "WARNING: embedding_size is larger than the dimensionality of the input dataset. Setting embedding_size to",
                input_dim)
            embedding_size = input_dim
        # Init Autoencoder parameters
        layers = [input_dim, 500, 500, 2000, embedding_size]
        autoencoder = autoencoder_class(layers=layers).to(device) 
    else:
        autoencoder.to(device)
    assert hasattr(autoencoder, "fitted"), "Autoencoder has no attribute 'fitted' and is therefore not compatible. Check documentation of fitted cluspy.deep.flexible_autoencoder.FlexibleAutoencoder"
    if not autoencoder.fitted:
        # Pretrain Autoencoder
        autoencoder.fit(n_epochs=n_epochs, lr=learning_rate, dataloader=trainloader,
            device=device, optimizer_class=optimizer_class, loss_fn=loss_fn)
    return autoencoder




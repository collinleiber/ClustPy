from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep import get_dataloader
from clustpy.data import create_subspace_data
import torch


def test_abstract_autoencoder():
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    batch_size = 256
    data_batch = torch.Tensor(data[:batch_size])
    autoencoder = _AbstractAutoencoder()
    # Test encoding
    embedded = autoencoder.encode(data_batch)
    assert torch.equal(data_batch, embedded)
    # Test decoding
    decoded = autoencoder.decode(embedded)
    assert torch.equal(data_batch, decoded)
    # Test forwarding
    forwarded = autoencoder.forward(data_batch)
    assert torch.equal(data_batch, forwarded)


def test_abstract_autoencoder_with_dummy_torch_parameter():
    data, _ = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    batch_size = 256
    data_batch = [torch.arange(256), torch.Tensor(data[:batch_size])]
    device = torch.device('cpu')
    loss_fn = torch.nn.MSELoss()
    autoencoder = _AbstractAutoencoder()
    autoencoder.dummy_parameter = torch.nn.Parameter(torch.tensor([0.]))  # Needed for fit to work
    autoencoder.encode = lambda x: x + autoencoder.dummy_parameter
    # Test loss
    loss, embedded, decoded = autoencoder.loss(data_batch, loss_fn, device)
    assert torch.equal(data_batch[1], embedded)
    assert torch.equal(data_batch[1], decoded)
    assert torch.equal(loss, torch.tensor(0.))
    # Test evaluate
    dataloader = get_dataloader(data, batch_size)
    loss = autoencoder.evaluate(dataloader, loss_fn, device)
    assert torch.equal(loss, torch.tensor(0.))
    # Test fitting (with data)
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True
    autoencoder.fitted = False
    # Test fitting (with dataloader)
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, dataloader=dataloader)
    assert autoencoder.fitted is True
    autoencoder.fitted = False
    # Test fitting (with data and eval_data)
    eval_data, _ = create_subspace_data(500, subspace_features=(3, 50), random_state=1)
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data, data_eval=eval_data)
    assert autoencoder.fitted is True
    autoencoder.fitted = False
    # Test fitting (with dataloader and eval_dataloader)
    eval_dataloader = get_dataloader(data, batch_size)
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, dataloader=dataloader, evalloader=eval_dataloader)
    assert autoencoder.fitted is True
    autoencoder.fitted = False
    # Test fitting (with data and eval_data and scheduler)
    scheduler = torch.optim.lr_scheduler.StepLR
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data, data_eval=eval_data, scheduler=scheduler,
                    scheduler_params={"step_size": 0.1})
    assert autoencoder.fitted is True

"""Epoch scheduling of the actual DEC/IDEC clustering optimizer."""
import numpy as np
import pytest
import torch
from sklearn.base import clone
from clustpy.deep import DEC, IDEC
from clustpy.deep.neural_networks import FeedforwardAutoencoder


class RecordingSGD(torch.optim.SGD):
    """Record the learning rate used by actual minibatch updates."""
    rates = []

    def step(self, closure=None):
        self.rates.append(self.param_groups[0]["lr"])
        return super().step(closure)


def make_estimator(cls, **kwargs):
    """Use a small real autoencoder and deterministic clustering data."""
    options = dict(pretrain_epochs=0, clustering_epochs=3, batch_size=8,
               neural_network=(FeedforwardAutoencoder, {"layers": [4, 3, 2]}),
               optimizer_class=RecordingSGD, clustering_optimizer_params={"lr": 0.1},
               device="cpu", random_state=42)
    options.update(kwargs)
    return cls(2, **options)


@pytest.mark.parametrize("cls", [DEC, IDEC])
def test_clustering_scheduler_steps_after_each_epoch_and_restarts(cls):
    """A public estimator schedules minibatches by epoch and resets on refit."""
    data = np.random.RandomState(0).normal(size=(32, 4)).astype(np.float32)
    options = {"step_size": 1, "gamma": 0.5}
    model = make_estimator(cls, pretrain_epochs=2, scheduler=torch.optim.lr_scheduler.StepLR,
                           scheduler_params=options)
    copied = clone(model)
    assert copied.scheduler is torch.optim.lr_scheduler.StepLR
    assert copied.scheduler_params == options
    for _ in range(2):
        RecordingSGD.rates = []
        model.fit(data)
        np.testing.assert_allclose(RecordingSGD.rates, np.repeat([0.001, 0.001, 0.1, 0.05, 0.025], 4))
        assert np.isfinite(model.dec_cluster_centers_).all()
    assert options == {"step_size": 1, "gamma": 0.5}


@pytest.mark.parametrize("cls", [DEC, IDEC])
def test_noop_scheduler_preserves_unscheduled_training(cls):
    """An identity schedule leaves the trained model and labels unchanged."""
    data = np.random.RandomState(0).normal(size=(32, 4)).astype(np.float32)
    ordinary = make_estimator(cls).fit(data)
    scheduled = make_estimator(cls, scheduler=torch.optim.lr_scheduler.StepLR,
                              scheduler_params={"step_size": 1, "gamma": 1.0}).fit(data)
    np.testing.assert_array_equal(ordinary.dec_labels_, scheduled.dec_labels_)
    np.testing.assert_array_equal(ordinary.dec_cluster_centers_, scheduled.dec_cluster_centers_)
    for expected, actual in zip(ordinary.neural_network_trained_.parameters(),
                                scheduled.neural_network_trained_.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [DEC, IDEC])
def test_metric_scheduler_is_rejected_without_validation_loss(cls):
    """Do not silently feed an undefined metric to plateau schedulers."""
    model = make_estimator(cls, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau)
    with pytest.raises(ValueError, match="validation loss"):
        model.fit(np.zeros((32, 4), dtype=np.float32))

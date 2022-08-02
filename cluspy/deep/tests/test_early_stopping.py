from cluspy.deep._early_stopping import EarlyStopping
import torch

def test_early_stopping():
    early_stopping = EarlyStopping(patience=3, min_delta=0.2)
    early_stopping(torch.tensor(2))
    assert early_stopping.best_loss == 2
    assert early_stopping.counter == 0
    assert early_stopping.early_stop is False
    early_stopping(torch.tensor(2))
    assert early_stopping.best_loss == 2
    assert early_stopping.counter == 1
    assert early_stopping.early_stop is False
    early_stopping(torch.tensor(1.5))
    assert early_stopping.best_loss == 1.5
    assert early_stopping.counter == 0
    assert early_stopping.early_stop is False
    early_stopping(torch.tensor(1.6))
    assert early_stopping.best_loss == 1.5
    assert early_stopping.counter == 1
    assert early_stopping.early_stop is False
    early_stopping(torch.tensor(1.4))
    assert early_stopping.best_loss == 1.5
    assert early_stopping.counter == 2
    assert early_stopping.early_stop is False
    early_stopping(torch.tensor(1.5))
    assert early_stopping.best_loss == 1.5
    assert early_stopping.counter == 3
    assert early_stopping.early_stop is True
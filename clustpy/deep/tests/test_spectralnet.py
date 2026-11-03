import copy
import pickle

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.datasets import make_blobs, load_digits
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from clustpy.deep import SpectralNet
from clustpy.deep.neural_networks import FeedforwardAutoencoder, VariationalAutoencoder
from clustpy.deep.spectralnet import _spectralnet_affinity, _spectralnet_loss, _spectralnet_orthogonalizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectralnet_orthogonality(dtype):
    """The frozen output layer satisfies the empirical orthogonality constraint."""
    generator = torch.Generator().manual_seed(24)
    raw = torch.randn(40, 3, generator=generator, dtype=dtype)
    output = raw @ _spectralnet_orthogonalizer(raw)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    np.testing.assert_allclose((output.T @ output / len(output)).numpy(), np.eye(3), atol=tolerance)


def test_spectralnet_rejects_rank_loss():
    """A collapsed map must fail instead of returning a spurious zero objective."""
    with pytest.raises(ValueError, match="column rank"):
        _spectralnet_orthogonalizer(torch.ones(20, 3))


def test_spectralnet_laplacian_objective():
    """Pairwise energy equals the independent Laplacian quadratic-form reference."""
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [2., 1.]], dtype=torch.float64)
    weights = _spectralnet_affinity(x, 2)
    output = torch.tensor([[0., 1.], [2., 0.], [-1., 3.], [4., 2.]], dtype=torch.float64, requires_grad=True)
    laplacian = torch.diag(weights.sum(1)) - weights
    expected = 2 * torch.trace(output.T @ laplacian @ output) / len(output)
    loss = _spectralnet_loss(weights, output)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert torch.count_nonzero(output.grad) > 0
    torch.testing.assert_close(weights, weights.T)
    assert torch.count_nonzero(weights.diag()) == 0


def test_spectralnet_fixed_gaussian_scale():
    """A fixed fitted bandwidth gives the prescribed Gaussian weight for a pair."""
    points = torch.tensor([[0.], [2.], [5.]], dtype=torch.float64)
    weights = _spectralnet_affinity(points, 2, scale=2.)
    expected = torch.exp(-torch.tensor(4. / 8., dtype=torch.float64))
    torch.testing.assert_close(weights[0, 1], expected)


@pytest.mark.parametrize("use_siamese,use_autoencoder", [(False, False), (True, False), (True, True)])
def test_spectralnet_fit_predict_and_repeatability(use_siamese, use_autoencoder):
    """Public fit/transform/predict remains reproducible with either preprocessing path."""
    data, _ = make_blobs(90, n_features=5, centers=3, random_state=12)
    estimator = SpectralNet(3, batch_size=45, clustering_epochs=3, siamese_epochs=2,
                            spectral_hidden_layers=(16, 16), siamese_hidden_layers=(16, 4),
                            use_siamese=use_siamese, use_autoencoder=use_autoencoder,
                            neural_network=(FeedforwardAutoencoder, {"layers": [5, 8, 3]}) if use_autoencoder else None,
                            pretrain_epochs=2, embedding_size=3, random_state=42, device="cpu")
    with pytest.raises(NotFittedError):
        estimator.transform(data)
    assert estimator.fit(data) is estimator
    assert estimator.labels_.shape == (90,)
    assert estimator.labels_.dtype == np.int32
    assert len(np.unique(estimator.labels_)) == 3
    embedding = estimator.transform(data)
    assert embedding.shape == (90, 3)
    assert np.isfinite(embedding).all()
    np.testing.assert_array_equal(estimator.predict(data), estimator.labels_)
    repeated = clone(estimator).fit(data)
    np.testing.assert_array_equal(repeated.labels_, estimator.labels_)
    np.testing.assert_allclose(repeated.transform(data), embedding)
    assert np.isfinite(estimator.spectral_loss_history_).all()
    assert estimator.affinity_scale_ > 0
    assert np.isfinite(estimator.siamese_loss_history_).all()
    assert estimator.transform(data[:1]).shape == (1, 3)
    # Orthogonalization weights are learned once, not recomputed from prediction batches.
    estimator.batch_size = 13
    np.testing.assert_allclose(estimator.transform(data), embedding, rtol=2e-5, atol=2e-5)
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(restored.transform(data), estimator.transform(data))
    copied = copy.deepcopy(estimator)
    np.testing.assert_array_equal(copied.predict(data), estimator.predict(data))
    if use_autoencoder:
        assert "feature_network" in dict(estimator.neural_network_trained_.named_children())
    assert "orthogonal_weights" in estimator.neural_network_trained_.state_dict()


def test_spectralnet_new_data_and_feature_validation():
    """An unseen point is embedded without retaining or extending the training graph."""
    data, _ = make_blobs(60, n_features=4, centers=2, random_state=4)
    estimator = SpectralNet(2, batch_size=30, spectral_hidden_layers=(8,), use_siamese=False,
                            clustering_epochs=2, random_state=2).fit(data)
    unseen = data[:7] + 0.01
    assert estimator.predict(unseen).shape == (7,)
    with pytest.raises(ValueError, match="features"):
        estimator.transform(np.ones((3, 5)))


def test_spectralnet_variational_feature_network():
    """Variational preprocessing consistently uses the mean, as encode_batchwise does."""
    data, _ = make_blobs(60, n_features=5, centers=2, random_state=13)
    estimator = SpectralNet(2, batch_size=30, use_siamese=False,
                            neural_network=(VariationalAutoencoder, {"layers": [5, 8, 3]}),
                            pretrain_epochs=2, embedding_size=3, clustering_epochs=2,
                            spectral_hidden_layers=(8,), random_state=42, device="cpu").fit(data)
    assert estimator.transform(data).shape == (60, 2)
    np.testing.assert_array_equal(estimator.predict(data), estimator.labels_)


def test_spectralnet_default_step_keeps_digits_training_finite():
    """The default step survives a full training budget for a ten-dimensional map."""
    train, test = train_test_split(load_digits().data / 16, test_size=0.2, random_state=24)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        estimator = SpectralNet(10, batch_size=256,
                                neural_network=(FeedforwardAutoencoder, {"layers": [64, 32, 16]}),
                                embedding_size=16, pretrain_epochs=5, random_state=24, device="cpu").fit(train)
        assert estimator.transform(test).shape == (360, 10)
        assert np.isfinite(estimator.transform(test)).all()
        assert len(estimator.spectral_loss_history_) == 100
        assert np.isfinite(estimator.spectral_loss_history_).all()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("params,message", [
    ({"n_clusters": 0}, "n_clusters"),
    ({"batch_size": 1}, "batch_size"),
    ({"n_neighbors": 0}, "Neighbor"),
    ({"affinity_scale": 0}, "affinity_scale"),
    ({"clustering_epochs": -1}, "Epoch"),
    ({"spectral_hidden_layers": (0,)}, "widths"),
])
def test_spectralnet_parameter_validation(params, message):
    """Invalid configurations fail before beginning neural training."""
    options = {"n_clusters": 2, "use_siamese": False, "random_state": 42}
    options.update(params)
    with pytest.raises(ValueError, match=message):
        SpectralNet(**options).fit(np.arange(60, dtype=float).reshape(20, 3))

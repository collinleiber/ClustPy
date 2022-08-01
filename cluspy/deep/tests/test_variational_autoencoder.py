from cluspy.data import load_optdigits
from cluspy.deep import VariationalAutoencoder


def test_simple_variational_autoencoder_with_optdigits():
    data, labels = load_optdigits()
    autoencoder = VariationalAutoencoder(layers=[data.shape[1], 128, 64, 10])
    autoencoder.fit(n_epochs=5, lr=1e-3, data=data)
    assert autoencoder.fitted is True

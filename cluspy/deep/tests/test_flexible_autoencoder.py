from cluspy.data import load_optdigits
from cluspy.deep import FlexibleAutoencoder


def test_simple_flexible_autoencoder_with_optdigits():
    data, labels = load_optdigits()
    autoencoder = FlexibleAutoencoder(layers=[data.shape[1], 128, 64, 10])
    autoencoder.fit(n_epochs=5, lr=1e-3, data=data)
    print(autoencoder.fitted)
    assert autoencoder.fitted is True

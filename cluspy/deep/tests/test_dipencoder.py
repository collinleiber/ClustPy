from cluspy.data import load_optdigits
from cluspy.deep import DipEncoder


def test_simple_dipencoder_with_optdigits():
    X, labels = load_optdigits()
    dipencoder = DipEncoder(10, pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(dipencoder, "labels_")
    dipencoder.fit(X)
    assert dipencoder.labels_.shape == labels.shape

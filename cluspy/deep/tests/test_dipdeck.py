from cluspy.data import load_optdigits
from cluspy.deep import DipDECK


def test_simple_dipdeck_with_optdigits():
    X, labels = load_optdigits()
    dipdeck = DipDECK(pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(dipdeck, "labels_")
    dipdeck.fit(X)
    assert dipdeck.labels_.shape == labels.shape

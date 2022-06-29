from cluspy.data import load_optdigits
from cluspy.deep import VaDE


def test_simple_vade_with_optdigits():
    X, labels = load_optdigits()
    vade = VaDE(10, pretrain_epochs=10, clustering_epochs=10, n_gmm_initializations=10)
    assert not hasattr(vade, "labels_")
    vade.fit(X)
    assert vade.labels_.shape == labels.shape

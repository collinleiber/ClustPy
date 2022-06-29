from cluspy.data import load_optdigits
from cluspy.deep import DCN


def test_simple_dcn_with_optdigits():
    X, labels = load_optdigits()
    dcn = DCN(10, pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(dcn, "labels_")
    dcn.fit(X)
    assert dcn.labels_.shape == labels.shape

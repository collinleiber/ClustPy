from cluspy.data import load_nrletters
from cluspy.deep import ENRC


def test_simple_enrc_with_nrletters():
    X, labels = load_nrletters()
    enrc = ENRC([6,4,3], pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(enrc, "labels_")
    enrc.fit(X)
    assert enrc.labels_.shape == labels.shape
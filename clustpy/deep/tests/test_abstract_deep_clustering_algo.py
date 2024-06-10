from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.data import create_subspace_data


def test_abstract_deep_clustering_algo():
    fw_ae = FeedforwardAutoencoder([53, 4])
    adca = _AbstractDeepClusteringAlgo(256, fw_ae, 10, 1)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    X_embed = adca.transform(X)
    assert X_embed.shape == (1500, 4)

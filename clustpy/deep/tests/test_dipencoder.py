from clustpy.deep import DipEncoder
from clustpy.deep.dipencoder import plot_dipencoder_embedding
from clustpy.data import create_subspace_data
import numpy as np
import torch


def test_simple_dipencoder():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    dipencoder = DipEncoder(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(dipencoder, "labels_")
    dipencoder.fit(X)
    assert dipencoder.labels_.dtype == np.int32
    assert dipencoder.labels_.shape == labels.shape
    # Test if random state is working
    # TODO Does not work every time -> Check why
    # dipencoder2 = DipEncoder(3, pretrain_epochs=3, clustering_epochs=3, random_state=1, debug=True)
    # dipencoder2.fit(X)
    # assert np.array_equal(dipencoder.labels_, dipencoder2.labels_)
    # assert np.allclose(dipencoder.projection_axes_, dipencoder2.projection_axes_, atol=1e-1)
    # assert dipencoder.index_dict_ == dipencoder2.index_dict_


def test_plot_dipencoder_embedding():
    embedded_data = np.array(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
         [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17],
         [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21],
         [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37],
         [38, 38, 38], [39, 39, 39]])
    n_clusters = 3
    cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    projection_axes = np.array([[11, 11, 11], [30, 30, 30], [19, 19, 19]])
    index_dict = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    plot_dipencoder_embedding(embedded_data, n_clusters, cluster_labels, projection_axes, index_dict, show_plot=False)
    # Only check if error is thrown
    assert True

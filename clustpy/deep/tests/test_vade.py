from clustpy.deep import VaDE
from clustpy.data import create_subspace_data
import numpy as np
import torch


def test_simple_vade():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    X = (X - np.mean(X)) / np.std(X)
    vade = VaDE(3, pretrain_epochs=3, clustering_epochs=3,
                initial_clustering_params={"n_init": 1, "covariance_type": "diag"}, random_state=1)
    assert not hasattr(vade, "labels_")
    vade.fit(X)
    assert vade.labels_.dtype == np.int32
    assert vade.labels_.shape == labels.shape
    # Test if random state is working
    vade2 = VaDE(3, pretrain_epochs=3, clustering_epochs=3,
                 initial_clustering_params={"n_init": 1, "covariance_type": "diag"}, random_state=1)
    vade2.fit(X)
    assert np.array_equal(vade.labels_, vade2.labels_)
    assert np.array_equal(vade.cluster_centers_, vade2.cluster_centers_)
    assert np.array_equal(vade.vade_labels_, vade2.vade_labels_)
    assert np.array_equal(vade.vade_cluster_centers_, vade2.vade_cluster_centers_)
    # Test predict
    labels_predict = vade.predict(X)
    assert np.array_equal(vade.labels_, labels_predict)

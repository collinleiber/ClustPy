from clustpy.deep import VaDE
from clustpy.data import create_subspace_data
import numpy as np
import torch
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep import mean_squared_error


def test_vade_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(VaDE(3, pretrain_epochs=3, clustering_epochs=3, ssl_loss_fn=mean_squared_error),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_vade():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    vade = VaDE(3, pretrain_epochs=3, clustering_epochs=3,
                initial_clustering_params={"n_init": 1, "covariance_type": "diag"}, random_state=1)
    assert not hasattr(vade, "labels_")
    vade.fit(X)
    assert vade.labels_.dtype == np.int32
    assert vade.labels_.shape == labels.shape
    X_embed = vade.transform(X)
    assert X_embed.shape == (X.shape[0], vade.embedding_size)
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

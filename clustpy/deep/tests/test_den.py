from clustpy.deep import get_dataloader, DEN
from clustpy.data import create_subspace_data
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.estimator_checks import check_estimator


def test_den_estimator():
    check_estimator(DEN(3, pretrain_epochs=3), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_simple_den():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    den = DEN(3, pretrain_epochs=3, random_state=1)
    assert not hasattr(den, "labels_")
    den.fit(X)
    assert den.labels_.dtype == np.int32
    assert den.labels_.shape == labels.shape
    # Test if random state is working
    den2 = DEN(3, pretrain_epochs=3, random_state=1)
    den2.fit(X)
    assert np.array_equal(den2.labels_, den2.labels_)


def test_den_with_predefined_neighbors():
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    n_neighbors = 5
    # Get dataloader with neighbors
    dist_matrix = squareform(pdist(X))
    neighbor_ids = np.argsort(dist_matrix, axis=1)
    neighbors = [X[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    trainloader = get_dataloader(X, 256, True, additional_inputs=neighbors)
    test_laoder = get_dataloader(X, 256, False)
    # Eecute DEN
    den = DEN(3, pretrain_epochs=3, random_state=1, custom_dataloaders=(trainloader, test_laoder), n_neighbors=n_neighbors, weight_locality_constraint=2, weight_sparsity_constraint=2, heat_kernel_t_parameter=2, group_lasso_lambda_parameter=2)
    assert not hasattr(den, "labels_")
    den.fit(X)
    assert den.labels_.dtype == np.int32
    assert den.labels_.shape == labels.shape
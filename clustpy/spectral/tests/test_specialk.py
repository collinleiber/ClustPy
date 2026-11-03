import numpy as np
from clustpy.spectral import SpecialK
from clustpy.spectral.specialk import _get_neighborhood_adjacency_matrix
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator
import pytest

def test_specialk_estimator():
    check_clustpy_estimator(SpecialK(), ("check_complex_data"))

"""
Tests regarding the SpecialK object
"""


def test_simple_SpecialK():
    X, labels = make_blobs(250, 4, centers=3, random_state=1)
    n_neighbors = 10
    specialk = SpecialK(random_state=1, n_neighbors=n_neighbors)
    assert not hasattr(specialk, "labels_")
    specialk.fit(X)
    assert specialk.labels_.dtype == np.int32
    assert specialk.labels_.shape == labels.shape
    assert len(np.unique(specialk.labels_)) == specialk.n_clusters_
    assert np.array_equal(np.unique(specialk.labels_), np.arange(specialk.n_clusters_))
    # Test if random state is working
    similarity_matrix = _get_neighborhood_adjacency_matrix(X, n_neighbors=n_neighbors)
    specialk2 = SpecialK(random_state=1, similarity_matrix=similarity_matrix, n_neighbors=n_neighbors)
    specialk2.fit(X)
    assert np.array_equal(specialk.n_clusters_, specialk2.n_clusters_)
    assert np.array_equal(specialk.labels_, specialk2.labels_)
    # Test with parameters
    specialk = SpecialK(significance=0.1, n_dimensions=150, similarity_matrix='SAM', n_neighbors=5,
                        n_cluster_pairs_to_consider=None, max_n_clusters=5, random_state=1, debug=True)
    specialk.fit(X)
    assert specialk.labels_.dtype == np.int32
    assert specialk.labels_.shape == labels.shape
    assert len(np.unique(specialk.labels_)) == specialk.n_clusters_
    assert np.array_equal(np.unique(specialk.labels_), np.arange(specialk.n_clusters_))


@pytest.mark.parametrize("similarity_matrix", ["NAM", "SAM"])
@pytest.mark.parametrize("state_type", [int, np.random.RandomState])
def test_specialk_reproducible_probability_bounds(monkeypatch, similarity_matrix, state_type):
    # Disconnected blob neighborhoods have repeated eigenvalues. Arbitrary
    # eigensolver starts can rotate their eigenvectors and change the bounds,
    # even when the final clustering happens to remain the same.
    import clustpy.spectral.specialk as specialk_module

    original_bound = specialk_module._log_zz_top_bound
    bounds = []

    def record_bound(*args, **kwargs):
        value = original_bound(*args, **kwargs)
        bounds.append(value)
        return value

    monkeypatch.setattr(specialk_module, "_log_zz_top_bound", record_bound)
    X, _ = make_blobs(250, centers=3, n_features=4, cluster_std=0.5, random_state=2)
    results = []
    for _ in range(2):
        bounds.clear()
        estimator = SpecialK(
            random_state=state_type(1), similarity_matrix=similarity_matrix,
            n_dimensions=200, max_n_clusters=5,
        ).fit(X)
        results.append((estimator.n_clusters_, estimator.labels_.copy(), bounds.copy()))

    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert len(results[0][2]) == len(results[1][2])
    np.testing.assert_allclose(results[0][2], results[1][2], rtol=1e-12, atol=1e-12)

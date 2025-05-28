from clustpy.density import MultiDensityDBSCAN
from clustpy.density.multi_density_dbscan import _sort_neighbors_by_densities, _add_neighbors_to_neighbor_list
from sklearn.datasets import make_blobs
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator


def test_multi_density_dbscan_estimator():
    check_clustpy_estimator(MultiDensityDBSCAN(k=5), ("check_complex_data"))


def test_add_neighbors_to_neighbor_list():
    densities = np.array([3, 5, 7, 4, 1, 6, 2, 9, 8, 0, 10, 11, 0.5])
    neighbors = [6, 5, 2, 7, 11]  # densities: [2, 6, 7, 9, 11]
    new_neighbors = np.array(
        [0, 4, 1, 8, 10, 11, 12, 2])  # Sorted: [12, 4, 0, 1, 2, 8, 10, 11] / densities: [0.5, 1, 3, 5, 7, 8, 10, 11]
    labels = np.array([0] + [-1] * 8 + [0] + [-1] * 3)
    neighbors = _add_neighbors_to_neighbor_list(densities, labels, neighbors, new_neighbors)
    assert np.array_equal([12, 4, 6, 1, 5, 2, 8, 7, 10, 11], neighbors)
    assert np.array_equal([0.5, 1, 2, 5, 6, 7, 8, 9, 10, 11], densities[neighbors])
    # Check if order of samples with same density is correct
    densities = np.array([1, 2, 3, 2, 1, 2, 2, 0, 4, 2])
    neighbors = [4, 1, 6]
    new_neighbors = np.array([3, 0, 2, 6, 5, 9, 1, 7])
    labels = np.array([-1] * 9 + [0])
    neighbors = _add_neighbors_to_neighbor_list(densities, labels, neighbors, new_neighbors)
    assert np.array_equal([7, 0, 4, 1, 3, 5, 6, 2], neighbors)
    assert np.array_equal([0, 1, 1, 2, 2, 2, 2, 3], densities[neighbors])


def test_sort_neighbors_by_densities():
    densities = np.array([0, 3, 2, 6, 4, 7, 8, 10, 1, 5, 3, 7, 9, 1])
    neighbors = [2, 5, 8, 10, 12]  # densities = [2, 7, 1, 3, 9]
    sorted_neighbors = _sort_neighbors_by_densities(neighbors, densities)
    assert np.array_equal([8, 2, 10, 5, 12], sorted_neighbors)
    densities = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    neighbors = [0, 1, 2, 4, 7, 6]  # densities = [0, 0, 1, 3, 4, 4]
    sorted_neighbors = _sort_neighbors_by_densities(neighbors, densities)
    assert np.array_equal([0, 1, 2, 4, 6, 7], sorted_neighbors)


def test_simple_MutliDensityDBSCAN():
    X, labels = make_blobs(200, 4, centers=3, random_state=1)
    # With additional queue sorting
    md_dbscan = MultiDensityDBSCAN()
    assert not hasattr(md_dbscan, "labels_")
    md_dbscan.fit(X)
    assert md_dbscan.labels_.dtype == np.int32
    assert md_dbscan.labels_.shape == labels.shape

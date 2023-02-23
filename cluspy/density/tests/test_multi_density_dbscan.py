from cluspy.data import load_wine
from cluspy.density import MultiDensityDBSCAN
from cluspy.density.multi_density_dbscan import _sort_neighbors_by_densities, _add_neighbors_to_neighbor_list
import numpy as np


def test_add_neighbors_to_neighbor_list():
    densities = np.array([3, 5, 7, 4, 1, 6, 2, 9, 8, 0, 10, 11, 0.5])
    neighbors = [6, 5, 2, 7, 11]  # densities: [2, 6, 7, 9, 11]
    new_neighbors = np.array(
        [0, 4, 1, 8, 10, 11, 12, 2])  # Sorted: [12, 4, 0, 1, 2, 8, 10, 11] / densities: [0.5, 1, 3, 5, 7, 8, 10, 11]
    labels = np.array([0] + [-1] * 8 + [0] + [-1] * 3)
    neighbors = _add_neighbors_to_neighbor_list(densities, labels, neighbors, new_neighbors)
    assert np.array_equal([12, 4, 6, 1, 5, 2, 8, 7, 10, 11], neighbors)
    assert np.array_equal([0.5, 1, 2, 5, 6, 7, 8, 9, 10, 11], densities[neighbors])


def test_sort_neighbors_by_densities():
    densities = np.array([0, 3, 2, 6, 4, 7, 8, 10, 1, 5, 3, 7, 9, 1])
    neighbors = [2, 5, 8, 10, 12]  # densities = [2, 7, 1, 3, 9]
    sorted_neighbors = _sort_neighbors_by_densities(neighbors, densities)
    assert np.array_equal([8, 2, 10, 5, 12], sorted_neighbors)


def test_simple_MutliDensityDBSCAN_with_wine():
    X, labels = load_wine()
    # With additional queue sorting
    md_dbscan = MultiDensityDBSCAN()
    assert not hasattr(md_dbscan, "labels_")
    md_dbscan.fit(X)
    assert md_dbscan.labels_.dtype == np.int32
    assert md_dbscan.labels_.shape == labels.shape

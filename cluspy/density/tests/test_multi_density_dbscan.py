from cluspy.data import load_wine
from cluspy.density import MultiDensityDBSCAN
from cluspy.density.multi_density_dbscan import _sort_neighbors_by_densities
import numpy as np


def test_simple_MutliDensityDBSCAN_with_wine():
    X, labels = load_wine()
    # With additional queue sorting
    md_dbscan = MultiDensityDBSCAN()
    assert not hasattr(md_dbscan, "labels_")
    md_dbscan.fit(X)
    assert md_dbscan.labels_.shape == labels.shape
    # Without additional queue sorting
    md_dbscan = MultiDensityDBSCAN(always_sort_densities=False)
    assert not hasattr(md_dbscan, "labels_")
    md_dbscan.fit(X)
    assert md_dbscan.labels_.shape == labels.shape


def test_sort_neighbors_by_densities():
    densities = [0, 3, 2, 6, 4, 7, 8, 10, 1, 5, 3, 7, 9, 1]
    neighbors = [2, 5, 8, 10, 12]  # densities = [2, 7, 1, 3, 9]
    sorted_neighbors = _sort_neighbors_by_densities(neighbors, densities)
    assert np.array_equal([8, 2, 10, 5, 12], sorted_neighbors)

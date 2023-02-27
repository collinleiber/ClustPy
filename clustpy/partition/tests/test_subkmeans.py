import numpy as np
from clustpy.partition import SubKmeans
from clustpy.partition.subkmeans import _transform_subkmeans_P_to_nrkmeans_P, \
    _transform_subkmeans_scatter_to_nrkmeans_scatters, _transform_subkmeans_centers_to_nrkmeans_centers, \
    _transform_subkmeans_m_to_nrkmeans_m
from clustpy.data import load_wine
from unittest.mock import patch


def test_transform_subkmeans_scatters_to_nrkmeans_scatters():
    X = np.c_[np.ones(25), np.ones(25), np.ones(25)]
    scatter_matrix = np.array([[30, 30, 30], [34, 34, 34], [35, 35, 35]])
    transformed_scatter_matrices = _transform_subkmeans_scatter_to_nrkmeans_scatters(X, scatter_matrix)
    assert len(transformed_scatter_matrices) == 2
    assert np.array_equal(transformed_scatter_matrices[0], scatter_matrix)
    assert np.array_equal(transformed_scatter_matrices[1], np.array([[0., 0., 0.],
                                                                     [0., 0., 0.],
                                                                     [0., 0., 0.]]))


def test_transform_subkmeans_P_to_nrkmeans_P():
    m = 5
    dims = 8
    transformed_P = _transform_subkmeans_P_to_nrkmeans_P(m, dims)
    assert len(transformed_P) == 2
    assert np.array_equal(transformed_P[0], np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(transformed_P[1], np.array([5, 6, 7]))


def test_transform_subkmeans_centers_to_nrkmeans_centers():
    X = np.c_[np.zeros(30), np.r_[np.zeros(10), np.zeros(10) + 1, np.zeros(10) + 2], np.zeros(30) + 3]
    centers = np.zeros((3, 8))
    transformed_centers = _transform_subkmeans_centers_to_nrkmeans_centers(X, centers)
    assert len(transformed_centers) == 2
    assert np.array_equal(transformed_centers[0], centers)
    assert np.array_equal(transformed_centers[1], np.array([[0, 1, 3]]))


def test_transform_subkmeans_m_to_nrkmeans_m():
    m = 5
    dims = 8
    transfomed_m = _transform_subkmeans_m_to_nrkmeans_m(m, dims)
    assert np.array_equal(transfomed_m, [5, 3])


"""
Tests regarding the SubKmeans object
"""


def test_simple_subkmeans_with_wine():
    X, labels = load_wine()
    subkm = SubKmeans(3)
    assert not hasattr(subkm, "labels_")
    subkm.fit(X)
    assert subkm.labels_.dtype == np.int32
    assert subkm.labels_.shape == labels.shape
    # Check if input parameters are working
    subkm_2 = SubKmeans(3, random_state=1, m=4, cluster_centers=np.ones((3, X.shape[1])))
    assert not hasattr(subkm_2, "labels_")
    subkm_2.fit(X)
    assert subkm_2.labels_.shape == labels.shape


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_subkmeans_result_with_wine(mock_fig):
    X, labels = load_wine()
    subkm = SubKmeans(3, max_iter=1)
    subkm.fit(X)
    assert None == subkm.plot_clustered_space(X, None, True, labels, True)


def test_transform_full_space():
    X = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3]
    subkm = SubKmeans(3, max_iter=1)
    subkm.fit(X)
    # Overwrite V
    subkm.V = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    X_transformed = subkm.transform_full_space(X)
    assert np.array_equal(X_transformed, np.c_[np.zeros(25) + 1, np.zeros(25), np.zeros(25) + 2, np.zeros(25) + 3])


def test_transform_clustered_space():
    X = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3]
    subkm = SubKmeans(3, max_iter=1)
    subkm.fit(X)
    # Overwrite V and m
    subkm.V = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    subkm.m = 3
    X_transformed = subkm.transform_clustered_space(X)
    assert np.array_equal(X_transformed, np.c_[np.zeros(25) + 1, np.zeros(25), np.zeros(25) + 2])


def test_calculate_cost_function():
    X = np.c_[np.zeros(30), np.r_[np.zeros(10), np.zeros(10) + 1, np.zeros(10) + 2], np.zeros(30) + 3]
    subkm = SubKmeans(4, max_iter=1)
    subkm.fit(X)
    subkm.V = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    subkm.m = 2
    subkm.scatter_matrix_ = np.array([[30, 30, 30], [34, 34, 34], [38, 38, 38]])
    costs = subkm.calculate_cost_function(X)
    assert costs == 68 + 20

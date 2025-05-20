import numpy as np
from clustpy.alternative import NrKmeans
from clustpy.alternative.nrkmeans import _assign_labels, _are_labels_equal, _is_matrix_orthogonal, _is_matrix_symmetric, \
    _create_full_rotation_matrix, _update_projections, _update_centers_and_scatter_matrix, _remove_empty_cluster, \
    _get_cost_function_of_subspace, _get_total_cost_function, _remove_empty_subspace, _get_precision, check_n_clusters_for_nr
from clustpy.data import create_nr_data
from unittest.mock import patch
from clustpy.utils.checks import check_clustpy_estimator


def test_nrkmeans_estimator():
    # Ignore check_clustering as it does not accept multiple sets of labels
    check_clustpy_estimator(NrKmeans([3, 3]), ("check_complex_data", "check_clustering"))


def test_check_n_clusters_for_nr():
    n_clusters = check_n_clusters_for_nr(3)
    assert n_clusters == [3]
    n_clusters = check_n_clusters_for_nr((3, 3))
    assert n_clusters == [3, 3]
    n_clusters = check_n_clusters_for_nr([3, 3])
    assert n_clusters == [3, 3]

def test_update_centers_and_scatter_matrices():
    X = np.array(
        [[1, 1, 1], [1, 3, 1], [7, 2, 1], [1, 1, 1], [1, 1, 3], [1, 7, 2], [4, 4, 4], [5, 5, 5], [7, 7, 7], [8, 8, 8]])
    n_clusters_subspace = 4
    labels_subspace = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 3])
    expected_centers = np.array([[3, 2, 1], [1, 3, 2], [np.nan, np.nan, np.nan], [6, 6, 6]])
    expected_scatter_matrix = np.array([[34, 10, 10], [10, 36, 10], [10, 10, 12]])
    calculated_centers, calculated_scatter_matrices = _update_centers_and_scatter_matrix(X, n_clusters_subspace,
                                                                                         labels_subspace)
    assert np.array_equal(expected_centers, calculated_centers, equal_nan=True)
    assert np.array_equal(expected_scatter_matrix, calculated_scatter_matrices)


def test_assign_labels():
    X = np.array([[1, 1, 1], [1, 2, 2], [2, 3, 1], [4, 5, 4], [5, 5, 5], [4, 5, 6], [10, 11, 11], [12, 10, 11]])
    V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    P_subspace = np.array([0, 1, 2])
    centers_subspace = np.array([[1, 1, 1], [4, 4, 4], [10, 10, 10]])
    labels = _assign_labels(X, V, centers_subspace, P_subspace)
    expected = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    assert np.array_equal(labels, expected)


def test_are_labels_equal():
    # First Test
    labels_1 = np.array([[0, 0, 1, 2, 1, 1, 0, 2]]).reshape((-1, 1))
    labels_2 = np.array([[0, 0, 1, 2, 1, 1, 0, 2]]).reshape((-1, 1))
    assert _are_labels_equal(labels_1, labels_2)
    # Second Test
    labels_3 = np.array([[0, 0, 1, 2, 1, 1, 2, 2]]).reshape((-1, 1))
    assert not _are_labels_equal(labels_1, labels_3)
    # Third Test
    labels_4 = np.array([[1, 1, 2, 0, 2, 2, 1, 0]]).reshape((-1, 1))
    assert _are_labels_equal(labels_1, labels_4)
    # Fourth test
    assert not _are_labels_equal(None, labels_1)
    assert not _are_labels_equal(labels_1, None)


def test_is_matrix_orthogonal():
    # First Test
    orthogonal_matrix = np.array([[-1.0, 0.0], [0.0, 1.0]])
    assert _is_matrix_orthogonal(orthogonal_matrix)
    # Second Test
    orthogonal_matrix = np.array([[0.0, -0.8, -0.6],
                                  [0.8, -0.36, 0.48],
                                  [0.6, 0.48, -0.64]])
    assert _is_matrix_orthogonal(orthogonal_matrix)
    # Third Test
    orthogonal_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert _is_matrix_orthogonal(orthogonal_matrix)
    # ----------- Test equals to False
    # First test - wrong dimensionality
    not_orthogonal_matrix = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert not _is_matrix_orthogonal(not_orthogonal_matrix)
    # Second test
    not_orthogonal_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert not _is_matrix_orthogonal(not_orthogonal_matrix)
    # Third test
    not_orthogonal_matrix = np.array([[-0.85616, 0.46933], [0.46933, 0.96236]])
    assert not _is_matrix_orthogonal(not_orthogonal_matrix)


def test_is_matrix_symmetric():
    # First test
    symmetric_matrix = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]])
    assert _is_matrix_symmetric(symmetric_matrix)
    # Second Test
    symmetric_matrix = np.array(
        [[0.234234, 0.87564, 0.123414, 0.74573],
         [0.87564, 0.5436346, 0.456364, 0.123],
         [0.123414, 0.456364, 0.23452, 0.23423],
         [0.74573, 0.123, 0.23423, 0.26]])
    assert _is_matrix_symmetric(symmetric_matrix)
    # ----------- Test equals to False
    # First test - wrong dimensionality
    not_symmetric_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert not _is_matrix_symmetric(not_symmetric_matrix)
    # Second test
    not_symmetric_matrix = np.array([[1.0, 0.32454], [0.32453, 1.0]])
    assert not _is_matrix_symmetric(not_symmetric_matrix)
    # Third test
    not_symmetric_matrix = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 3.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]])
    assert not _is_matrix_symmetric(not_symmetric_matrix)


def test_create_full_rotation_matrix():
    # First test
    dimensionality = 6
    projections = np.array([0, 3, 1, 4])
    V_C = np.array([[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]])
    V_F = _create_full_rotation_matrix(dimensionality, projections, V_C)
    V_F_check = np.array(
        [[1.0, 9.0, 0, 5.0, 13.0, 0], [3.0, 11.0, 0, 7.0, 15.0, 0], [0, 0, 1, 0, 0, 0],
         [2.0, 10.0, 0, 6.0, 14.0, 0],
         [4.0, 12.0, 0, 8.0, 16.0, 0], [0, 0, 0, 0, 0, 1]])
    assert np.array_equal(V_F, V_F_check)
    # Second test
    dimensionality = 6
    projections = np.array([1, 4, 5, 2, 3])
    V_C = np.array([[1.0, 6.0, 11.0, 16.0, 21.0], [2.0, 7.0, 12.0, 17.0, 22.0], [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0], [5.0, 10.0, 15.0, 20.0, 25.0]])
    V_F = _create_full_rotation_matrix(dimensionality, projections, V_C)
    V_F_check = np.array(
        [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 16.0, 21.0, 6.0, 11.0], [0, 4.0, 19.0, 24.0, 9.0, 14.0],
         [0, 5.0, 20.0, 25.0, 10.0, 15.0],
         [0, 2.0, 17.0, 22.0, 7.0, 12.0], [0, 3.0, 18.0, 23.0, 8.0, 13.0]])
    assert np.array_equal(V_F, V_F_check)


def test_update_projections():
    # First test
    transitions = np.array([0, 3, 1, 4])
    n_negative_e = 2
    P_1, P_2 = _update_projections(transitions, n_negative_e)
    assert np.array_equal(P_1, np.array([0, 3]))
    assert np.array_equal(P_2, np.array([4, 1]))
    # Second test
    transitions = np.array([1, 4, 5, 2, 3])
    n_negative_e = 2
    P_1, P_2 = _update_projections(transitions, n_negative_e)
    assert np.array_equal(P_1, np.array([1, 4]))
    assert np.array_equal(P_2, np.array([3, 2, 5]))


def test_remove_empty_cluster():
    n_clusters_subspace = 5
    centers_subspace = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [np.nan, np.nan, np.nan], [5, 5, 5]])
    labels_subspace = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5])
    n_clusters_subspace_new, centers_subspace_new, labels_subspace_new = _remove_empty_cluster(
        n_clusters_subspace, centers_subspace, labels_subspace, False)
    assert 4 == n_clusters_subspace_new
    assert np.array_equal(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [5, 5, 5]]), centers_subspace_new)
    assert np.array_equal(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]), labels_subspace_new)


def test_remove_empty_subspace():
    n_clusters = [5, 4, 3, 2, 1]
    m = [1, 0, 2, 0, 5]
    P = [np.array([0]), np.array([]), np.array([1, 2]), np.array([]), np.array([3, 4, 5, 6, 7])]
    centers = [np.zeros((5, 8)), np.zeros((4, 8)) + 1, np.zeros((3, 8)) + 2, np.zeros((2, 8)) + 3, np.zeros((1, 8)) + 4]
    labels = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3, np.zeros(25) + 4]
    scatter_matrices = [np.zeros((5, 8, 8)), np.zeros((4, 8, 8)) + 1, np.zeros((3, 8, 8)) + 2, np.zeros((2, 8, 8)) + 3,
                        np.zeros((1, 8, 8)) + 5]
    n_subspaces_calculated, n_clusters_calculated, m_calculated, P_calculated, centers_calculated, labels_calculated, scatter_matrices_calculated = _remove_empty_subspace(
        n_clusters, m, P, centers, labels, scatter_matrices, False)
    assert n_subspaces_calculated == 3
    assert n_clusters_calculated == [5, 3, 1]
    assert m_calculated == [1, 2, 5]
    assert all([np.array_equal(np.array([0]), P_calculated[0]),
                np.array_equal(np.array([1, 2]), P_calculated[1]),
                np.array_equal(np.array([3, 4, 5, 6, 7]), P_calculated[2])])
    assert all([np.array_equal(np.zeros((5, 8)), centers_calculated[0]),
                np.array_equal(np.zeros((3, 8)) + 2, centers_calculated[1]),
                np.array_equal(np.zeros((1, 8)) + 4, centers_calculated[2])])
    assert np.array_equal(np.c_[np.zeros(25), np.zeros(25) + 2, np.zeros(25) + 4], labels_calculated)
    assert all([np.array_equal(np.zeros((5, 8, 8)), scatter_matrices_calculated[0]),
                np.array_equal(np.zeros((3, 8, 8)) + 2, scatter_matrices_calculated[1]),
                np.array_equal(np.zeros((1, 8, 8)) + 5, scatter_matrices_calculated[2])])


def test_get_cost_function_of_subspace():
    cropped_V = np.array([[0, 1], [0, 0], [1, 0]])
    scatter_matrices_subspace = np.array([[30, 30, 30], [34, 34, 34], [38, 38, 38]])
    costs = _get_cost_function_of_subspace(cropped_V, scatter_matrices_subspace)
    assert 68 == costs


def test_get_total_cost_function():
    V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    P = [np.array([2, 0]), np.array([1])]
    scatter_matrices = [np.array([[30, 30, 30], [34, 34, 34], [38, 38, 38]]),
                        np.array([[32, 33, 34], [36, 37, 38], [40, 41, 42]])]
    costs = _get_total_cost_function(V, P, scatter_matrices)
    assert 68 + 37 == costs


def test_get_precision():
    X = np.array([[0, 0, 0, 0], [1, 2, 3, 0], [10, 11, 12, 0], [20, 21, 22, 0]])
    precision_calculated = _get_precision(X)
    assert precision_calculated == 2


"""
Tests regarding the NrKmeans object
"""


def test_simple_nrkmeans():
    X, labels = create_nr_data(200, random_state=1)
    nrk = NrKmeans([3, 3, 1], random_state=1)
    assert not hasattr(nrk, "labels_")
    nrk.fit(X)
    assert nrk.labels_.dtype == np.int32
    assert nrk.labels_.shape == labels.shape
    # Check if random state is working
    nrk_2 = NrKmeans([3, 3, 1], random_state=1)
    assert not hasattr(nrk_2, "labels_")
    nrk_2.fit(X)
    assert np.array_equal(nrk_2.labels_, nrk.labels_)
    assert np.array_equal(nrk_2.V_, nrk.V_)
    assert all([np.array_equal(nrk_2.cluster_centers_[i], nrk.cluster_centers_[i]) for i in range(2)])
    assert all([np.array_equal(nrk_2.scatter_matrices_[i], nrk.scatter_matrices_[i]) for i in range(2)])
    # Check result with mdl_for_noisespace and n_init=3 and outliers
    nrk_3 = NrKmeans([3, 3, 1], mdl_for_noisespace=True, n_init=3, outliers=True, random_state=1)
    assert not hasattr(nrk_3, "labels_")
    nrk_3.fit(X)
    assert nrk_3.labels_.shape == labels.shape
    # Check result with mdl_for_noisespace and n_init=3 and cost_type = mdl and debug = True
    nrk_4 = NrKmeans([3, 3, 1], mdl_for_noisespace=True, n_init=3, cost_type="mdl", debug=True, random_state=1)
    assert not hasattr(nrk_4, "labels_")
    nrk_4.fit(X)
    assert nrk_4.labels_.shape == labels.shape
    # Test predict
    assert np.array_equal(nrk.labels_, nrk.predict(X))
    assert np.array_equal(nrk_2.labels_[:-1], nrk_2.predict(X[:-1]))
    assert X.shape[0] - np.sum(nrk_3.labels_[:-2] == nrk_3.predict(X[:-2])) < X.shape[
        0] * 0.01  # Some points can change labels when running with outliers=True
    assert np.array_equal(nrk_4.labels_[:-3], nrk_4.predict(X[:-3]))


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_nrkmeans_result(mock_fig):
    X, labels = create_nr_data(200, random_state=1)
    nrk = NrKmeans([3, 3, 1], max_iter=1, random_state=1)
    nrk.fit(X)
    assert None == nrk.plot_subspace(X, 0, None, True, labels[:, 0], True)


def test_have_clusters_been_lost():
    X = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3]
    nrk = NrKmeans([3, 3, 1], max_iter=1, random_state=1)
    nrk.fit(X)
    nrk.n_clusters_final_ = [3, 3, 1]
    assert not nrk.have_clusters_been_lost()
    # Overwrite n_clusters
    nrk.n_clusters_final_ = [3, 1]
    assert nrk.have_clusters_been_lost()
    # Should be true if a whole subspace got lost
    nrk.n_clusters_final_ = [3, 2, 1]
    assert nrk.have_clusters_been_lost()


def test_have_subspaces_been_lost():
    X = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3]
    nrk = NrKmeans([3, 3, 1], max_iter=1, random_state=1)
    nrk.fit(X)
    nrk.n_clusters_final_ = [3, 3, 1]
    assert not nrk.have_subspaces_been_lost()
    # Overwrite n_clusters
    nrk.n_clusters_final_ = [3, 1]
    assert nrk.have_subspaces_been_lost()
    # Should be false if just a cluster got lost
    nrk.n_clusters_final_ = [3, 2, 1]
    assert not nrk.have_subspaces_been_lost()


def test_transform():
    X = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3]
    nrk = NrKmeans([3, 3, 1], max_iter=1, random_state=1)
    nrk.fit(X)
    # Overwrite V
    nrk.V_ = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    X_transformed = nrk.transform(X)
    assert np.array_equal(X_transformed, np.c_[np.zeros(25) + 1, np.zeros(25), np.zeros(25) + 2, np.zeros(25) + 3])


def test_transform_subspace():
    X = np.c_[np.zeros(25), np.zeros(25) + 1, np.zeros(25) + 2, np.zeros(25) + 3]
    nrk = NrKmeans([3, 1], max_iter=1, random_state=1)
    nrk.fit(X)
    # Overwrite V and P
    nrk.V_ = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    nrk.P_ = [np.array([1, 3]), np.array([2, 0])]
    nrk.n_clusters_final_ = [3, 1]
    X_transformed = nrk.transform_subspace(X, 0)
    assert np.array_equal(X_transformed, np.c_[np.zeros(25), np.zeros(25) + 3])
    X_transformed = nrk.transform_subspace(X, 1)
    assert np.array_equal(X_transformed, np.c_[np.zeros(25) + 2, np.zeros(25) + 1])


def test_calculate_mdl_costs():
    X = np.c_[np.zeros(30), np.r_[np.zeros(10), np.zeros(10) + 1, np.zeros(10) + 2], np.zeros(30) + 3]
    nrk = NrKmeans([4, 1], max_iter=1, random_state=1)
    nrk.fit(X)
    nrk.V_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    nrk.P_ = [np.array([2, 0]), np.array([1])]
    nrk.scatter_matrices_ = [np.array([[30, 30, 30], [34, 34, 34], [38, 38, 38]]),
                             np.array([[32, 33, 34], [36, 37, 38], [40, 41, 42]])]
    costs = nrk.calculate_cost_function()
    assert 68 + 37 == costs

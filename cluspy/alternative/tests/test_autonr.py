import numpy as np
from cluspy.alternative import AutoNR
from cluspy.alternative.autonr import _find_two_closest_centers, _merge_nearest_centers, _split_largest_cluster
from cluspy.data import load_fruit
from unittest.mock import patch


def test_find_clostest_centers():
    center_1 = np.array([1, 1, 1, 1])
    center_2 = np.array([3, 3, 3, 3])  # i
    center_3 = np.array([6, 6, 6, 6])
    center_4 = np.array([9, 9, 9, 9])
    center_5 = np.array([4, 4, 4, 4])  # j
    centers = np.array([center_1, center_2, center_3, center_4, center_5])
    i, j = _find_two_closest_centers(centers)
    assert i == 1
    assert j == 4


def test_merge_nearest_centers():
    center_1 = np.array([1, 1, 1, 1])
    center_2 = np.array([3, 3, 3, 3])
    center_3 = np.array([6, 6, 6, 6])
    center_4 = np.array([9, 9, 9, 9])
    center_5 = np.array([4, 4, 4, 4])
    centers = np.array([center_1, center_2, center_3, center_4, center_5])
    new_centers = _merge_nearest_centers(centers)
    center_merged = np.array([3.5, 3.5, 3.5, 3.5])
    assert np.array_equal(np.array([center_1, center_3, center_4, center_merged]), new_centers)


def test_split_largest_cluster():
    m = 20
    V = np.identity(4)
    points_per_cluster = 10
    labels = np.array([0] * points_per_cluster + [1] * points_per_cluster + [2] * points_per_cluster)
    center_1 = np.array([1, 1, 1, 1])
    center_2 = np.array([6, 6, 6, 6])
    center_3 = np.array([3, 3, 3, 3])
    centers = np.array([center_1, center_2, center_3])
    scatter_matrix_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    scatter_matrix_2 = np.array([[3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 5, 0], [0, 0, 0, 6]])
    scatter_matrix_3 = np.array([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    scatter_matrices = np.array([scatter_matrix_1, scatter_matrix_2, scatter_matrix_3])
    P = np.array([0, 1, 2, 3])
    new_centers = _split_largest_cluster(V, m, P, centers, scatter_matrices, labels)
    centers_split_1 = np.array([6 + 3 / 200, 6 + 4 / 200, 6 + 5 / 200, 6 + 6 / 200])
    centers_split_2 = np.array([6 - 3 / 200, 6 - 4 / 200, 6 - 5 / 200, 6 - 6 / 200])
    assert np.array_equal(np.array([center_1, center_3, centers_split_1, centers_split_2]), new_centers)


"""
Tests regarding the AutoNR object
"""


def test_simple_autonr_with_fruit():
    X, labels = load_fruit()
    autonr = AutoNR(random_state=1, mdl_for_noisespace=False)
    assert not hasattr(autonr, "labels_")
    autonr.fit(X)
    assert autonr.labels_.dtype == np.int32
    assert autonr.labels_.shape[0] == labels.shape[0]
    # Check if random state is working
    autonr_2 = AutoNR(random_state=1, mdl_for_noisespace=False)
    assert not hasattr(autonr_2, "labels_")
    autonr_2.fit(X)
    assert np.array_equal(autonr_2.labels_, autonr.labels_)
    assert np.array_equal(autonr_2.nrkmeans_.V, autonr.nrkmeans_.V)
    assert all(
        [np.array_equal(autonr_2.nrkmeans_.cluster_centers[i], autonr.nrkmeans_.cluster_centers[i]) for i in range(2)])
    assert all([np.array_equal(autonr_2.nrkmeans_.scatter_matrices_[i], autonr.nrkmeans_.scatter_matrices_[i]) for i in
                range(2)])


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_autonr_mdl_costs_progress_with_fruit(mock_fig):
    X, labels = load_fruit()
    autonr = AutoNR(max_subspaces=2, max_n_clusters=2)
    autonr.fit(X)
    assert None == autonr.plot_mdl_progress()


def test_dissolve_noise_space_with_fruit():
    X, _ = load_fruit()
    autonr = AutoNR(random_state=4, max_subspaces=4, max_n_clusters=4)
    autonr.fit(X)
    # Save original parameters
    n_clusters_copy = autonr.nrkmeans_.n_clusters.copy()
    m_copy = autonr.nrkmeans_.m.copy()
    P_copy = autonr.nrkmeans_.P.copy()
    scatter_matrices_copy = autonr.nrkmeans_.scatter_matrices_.copy()
    cluster_centers_copy = autonr.nrkmeans_.cluster_centers.copy()
    labels_copy = autonr.nrkmeans_.labels_.copy()
    # Random feature assignment or MDL-based feature assignment
    for strategy in [True, False]:
        if strategy == False:
            # Revert to original parameters
            autonr.nrkmeans_.n_clusters = n_clusters_copy.copy()
            autonr.nrkmeans_.m = m_copy.copy()
            autonr.nrkmeans_.P = P_copy.copy()
            autonr.nrkmeans_.scatter_matrices_ = scatter_matrices_copy.copy()
            autonr.nrkmeans_.cluster_centers = cluster_centers_copy.copy()
            autonr.nrkmeans_.labels_ = labels_copy.copy()
        autonr.dissolve_noise_space(X, random_feature_assignment=strategy)
        assert np.array_equal(autonr.nrkmeans_.n_clusters, n_clusters_copy[:-1])
        assert len(autonr.nrkmeans_.scatter_matrices_) == len(scatter_matrices_copy) - 1
        for i in range(len(autonr.nrkmeans_.scatter_matrices_)):
            assert np.array_equal(autonr.nrkmeans_.scatter_matrices_[i], scatter_matrices_copy[i])
        assert len(autonr.nrkmeans_.cluster_centers) == len(cluster_centers_copy) - 1
        for i in range(len(autonr.nrkmeans_.cluster_centers)):
            assert np.array_equal(autonr.nrkmeans_.cluster_centers[i], cluster_centers_copy[i])
        assert np.array_equal(autonr.nrkmeans_.labels_, labels_copy[:, :-1])
        assert len(autonr.nrkmeans_.m) == len(n_clusters_copy) - 1
        assert np.sum(autonr.nrkmeans_.m) == X.shape[1]
        assert len(autonr.nrkmeans_.P) == len(n_clusters_copy) - 1
        check_P = []
        for i in range(len(autonr.nrkmeans_.P)):
            assert len(autonr.nrkmeans_.P[i]) == autonr.nrkmeans_.m[i]
            check_P += autonr.nrkmeans_.P[i].tolist()
        assert np.array_equal(np.sort(check_P), np.arange(X.shape[1]))

from clustpy.deep import DipDECK
from clustpy.deep.dipdeck import _get_nearest_points_to_optimal_centers, _get_nearest_points, _get_dip_matrix
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_dipdeck_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DipDECK(n_clusters_init=10, pretrain_epochs=3, clustering_epochs=3), 
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_dipdeck():
    dipdeck = DipDECK()
    dipdeck, dipdeck2 = _test_dc_algorithm_simple(dipdeck, check_predict=False)
    assert np.array_equal(dipdeck.cluster_centers_, dipdeck2.cluster_centers_)


def test_dipdeck_augmentation():
    dipdeck = DipDECK()
    _test_dc_algorithm_with_augmentation(dipdeck)


def test_get_nearest_points_to_optimal_centers():
    X = np.array([[10, 10, 10],
                  [20, 20, 20],
                  [30, 30, 30],
                  [40, 40, 40],
                  [50, 50, 50],
                  [60, 60, 60],
                  [70, 70, 70]])
    optimal_centers = np.array([[2], [4.4]])
    embedded_data = np.array([[1], [2], [3], [4], [5], [6], [7]])
    centers, embedded_centers = _get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data)
    assert np.array_equal(centers, np.array([[20, 20, 20], [40, 40, 40]]))
    assert np.array_equal(embedded_centers, np.array([[2], [4]]))


def test_get_nearest_points():
    points_in_larger_cluster = np.array(
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1]])
    center = np.array([4.5, 3])
    size_smaller_cluster = 3
    max_cluster_size_diff_factor = 2
    new_points_in_larger_cluster = _get_nearest_points(points_in_larger_cluster, center, size_smaller_cluster,
                                                       max_cluster_size_diff_factor, 5)
    new_points_in_larger_cluster = np.sort(new_points_in_larger_cluster, axis=0)
    assert new_points_in_larger_cluster.shape == (6, 2)
    assert np.array_equal(new_points_in_larger_cluster, points_in_larger_cluster[2:-2])
    new_points_in_larger_cluster = _get_nearest_points(points_in_larger_cluster, center, size_smaller_cluster,
                                                       max_cluster_size_diff_factor, 11)
    new_points_in_larger_cluster = np.sort(new_points_in_larger_cluster, axis=0)
    assert new_points_in_larger_cluster.shape == (8, 2)
    assert np.array_equal(new_points_in_larger_cluster, points_in_larger_cluster[1:-1])


def test_get_dip_matrix():
    embedded_data = np.array(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
         [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17],
         [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21],
         [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37],
         [38, 38, 38], [39, 39, 39]])
    embedded_centers = np.array([[5, 5, 5], [16, 16, 16], [35, 35, 35]])
    cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    dip_matrix = _get_dip_matrix(embedded_data=embedded_data, embedded_centers_cpu=embedded_centers,
                                 cluster_labels_cpu=cluster_labels,
                                 n_clusters=3, max_cluster_size_diff_factor=2.2, pval_strategy="table", n_boots=1000,
                                 random_state=1)
    assert dip_matrix.shape == (3, 3)
    assert np.array_equal(dip_matrix.diagonal(), np.array([0, 0, 0]))
    dip_matrix_tmp = dip_matrix + np.identity(3) * 0.1
    assert np.max(dip_matrix_tmp) <= 1
    assert np.min(dip_matrix_tmp) >= 0


def test_dipdeck_module_to_device():
    import torch
    from clustpy.deep.dipdeck import _DipDECK_Module

    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    centers = np.array([[1.0, 2.0], [3.0, 4.0]])
    dip_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    module = _DipDECK_Module(
        n_clusters=2, init_np_labels=labels, init_np_centers=centers,
        dip_matrix_cpu=dip_matrix, dip_merge_threshold=0.9,
        max_n_clusters=5, min_n_clusters=1, max_cluster_size_diff_factor=2,
        pval_strategy="table", n_boots=100, random_state=np.random.RandomState(1),
    )
    assert module.to_device(torch.device("cpu")) is module
    torch.testing.assert_close(module.labels, torch.tensor(labels, dtype=torch.int32))
    torch.testing.assert_close(module.cluster_centers, torch.tensor(centers, dtype=torch.float32))
    torch.testing.assert_close(module.dip_matrix, torch.tensor(dip_matrix, dtype=torch.float32))

    # A real non-CPU transfer exposes discarded Tensor.to return values without
    # requiring CUDA hardware. Meta checks placement and metadata, not training.
    assert module.to_device(torch.device("meta")) is module
    assert module.labels.device.type == "meta"
    assert module.cluster_centers.device.type == "meta"
    assert module.dip_matrix.device.type == "meta"
    assert module.labels.dtype == torch.int32
    assert module.cluster_centers.dtype == module.dip_matrix.dtype == torch.float32
    assert module.labels.shape == (4,)
    assert module.cluster_centers.shape == (2, 2)
    assert module.dip_matrix.shape == (2, 2)

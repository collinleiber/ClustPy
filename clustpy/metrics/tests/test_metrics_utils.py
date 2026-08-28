from clustpy.metrics._metrics_utils import (_check_labels_arrays,
                                            _check_length_data_and_labels,
                                            _assign_noise_points_to_one_cluster,
                                             _assign_noise_points_to_singletons,
                                             _remove_noise_points,
                                             _assign_noise_points_to_nearest_cluster)
from clustpy.metrics import handle_noise
import pytest
import numpy as np


def test_check_labels_arrays():
    l1 = np.array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4.])
    assert l1.dtype == float
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    l1, l2 =_check_labels_arrays(l1, l2)
    assert l1.dtype == int and l2.dtype == int
    with pytest.raises(ValueError):
        _check_labels_arrays(l1, l2[1:])
    l3 = np.c_[l1, l2]
    with pytest.raises(ValueError):
        _check_labels_arrays(l1, l3)
    l1, l3 =_check_labels_arrays(l1, l3, allow_2d_labels = True)
    assert l1.shape == (10, ) and l3.shape == (10, 2)
    l3, l1 =_check_labels_arrays(l3, l1, allow_2d_labels = True)
    assert l1.shape == (10, ) and l3.shape == (10, 2)
    l3, l4 =_check_labels_arrays(l3, l3, allow_2d_labels = True)
    assert l3.shape == (10, 2) and l4.shape == (10, 2)


def test_check_length_data_and_labels():
    l1 = np.array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4.])
    assert l1.dtype == float
    X = np.array([[0., 2.], [1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.], [7., 8.], [8., 9.], [9., 10.]])
    print(X.shape)
    X, l1 =_check_length_data_and_labels(X, l1)
    assert X.dtype == float and l1.dtype == int
    with pytest.raises(ValueError):
        _check_length_data_and_labels(X, l1[1:])
    with pytest.raises(ValueError):
        _check_length_data_and_labels(X, np.array([0] * 10))


def test_assign_noise_points_to_one_cluster():
    labels = np.array([0, 1, -1, 2, -1, 1])
    new_labels = _assign_noise_points_to_one_cluster(labels)
    assert np.array_equal(new_labels, [0, 1, 3, 2, 3, 1])
    # No noise case
    labels_no_noise = np.array([0, 1, 1])
    new_labels = _assign_noise_points_to_one_cluster(labels_no_noise)
    assert np.array_equal(new_labels, labels_no_noise)


def test_assign_noise_points_to_singletons():
    labels = np.array([0, 1, -1, 2, -1, 1])
    new_labels = _assign_noise_points_to_singletons(labels)
    assert np.array_equal(new_labels, [0, 1, 3, 2, 4, 1])
    # No noise case
    labels_no_noise = np.array([0, 1, 1])
    new_labels = _assign_noise_points_to_singletons(labels_no_noise)
    assert np.array_equal(new_labels, labels_no_noise)


def test_remove_noise_points():
    labels = np.array([0, 1, -1, 2, -1, 1])
    new_labels, non_noise_ids = _remove_noise_points(labels)
    assert np.array_equal(new_labels, [0, 1, 2, 1])
    assert np.array_equal(non_noise_ids, [0, 1, 3, 5])
    # No noise case
    labels_no_noise = np.array([0, 1, 1])
    new_labels, non_noise_ids = _remove_noise_points(labels_no_noise)
    assert np.array_equal(new_labels, labels_no_noise)
    assert np.array_equal(non_noise_ids, [0, 1, 2])
    # All points are noise -> should raise
    labels_all_noise = np.array([-1, -1])
    with pytest.raises(ValueError):
        _remove_noise_points(labels_all_noise)


def test_assign_noise_points_to_nearest_cluster():
    labels = np.array([0, 1, -1, 2, -1, 1])
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6], [-1, -1], [2, 2]])
    new_labels = _assign_noise_points_to_nearest_cluster(labels, X)
    assert np.array_equal(new_labels, [0, 1, 2, 2, 0, 1])
    # No noise case
    labels_no_noise = np.array([0, 1, 1])
    X = np.array([[0,0], [1, 1], [2,2]])
    new_labels = _assign_noise_points_to_nearest_cluster(labels_no_noise, X)
    assert np.array_equal(new_labels, labels_no_noise)
    assert np.array_equal(X, np.array([[0,0], [1, 1], [2,2]]))
    # All points are noise -> should raise
    labels_all_noise = np.array([-1, -1])
    X_all_noise = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        _assign_noise_points_to_nearest_cluster(labels_all_noise, X_all_noise)
    # No data
    with pytest.raises(ValueError):
        _assign_noise_points_to_nearest_cluster(labels, None)


def test_handle_noise():
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6], [2, 2]])
    labels_compare = np.array([1, 1, 1, 0, 0])
    labels = np.array([0, 2, -1, 1, -1])
    # Keep strategy
    new_labels, new_X, new_labels_compare = handle_noise(labels, "keep", X, labels_compare)
    assert np.array_equal(new_labels, labels)
    assert np.array_equal(new_X, X)
    assert np.array_equal(new_labels_compare, labels_compare)
    # one_cluster
    new_labels, new_X, new_labels_compare = handle_noise(labels, "one_cluster", X, labels_compare)
    assert np.array_equal(new_labels, [0, 2, 3, 1, 3])
    assert np.array_equal(new_X, X)
    assert np.array_equal(new_labels_compare, labels_compare)
    # singletons
    new_labels, new_X, new_labels_compare = handle_noise(labels, "singletons", X, labels_compare)
    assert np.array_equal(new_labels, [0, 2, 3, 1, 4])
    assert np.array_equal(new_X, X)
    assert np.array_equal(new_labels_compare, labels_compare)
    # filter
    new_labels, new_X, new_labels_compare = handle_noise(labels, "filter")
    assert np.array_equal(new_labels, [0, 2, 1])
    assert new_X is None
    assert new_labels_compare is None
    new_labels, new_X, new_labels_compare = handle_noise(labels, "filter", X)
    assert np.array_equal(new_labels, [0, 2, 1])
    assert np.array_equal(new_X, np.array([[0, 0], [1, 1], [6, 6]]))
    assert new_labels_compare is None
    new_labels, new_X, new_labels_compare = handle_noise(labels, "filter", labels_compare=labels_compare)
    assert np.array_equal(new_labels, [0, 2, 1])
    assert new_X is None
    assert np.array_equal(new_labels_compare, [1, 1, 0])
    new_labels, new_X, new_labels_compare = handle_noise(labels, "filter", X, labels_compare)
    assert np.array_equal(new_labels, [0, 2, 1])
    assert np.array_equal(new_X, np.array([[0, 0], [1, 1], [6, 6]]))
    assert np.array_equal(new_labels_compare, [1, 1, 0])
    # nearest_cluster
    new_labels, new_X, new_labels_compare = handle_noise(labels, "nearest_cluster", X, labels_compare)
    assert np.array_equal(new_labels, [0, 2, 1, 1, 2])
    assert np.array_equal(new_X, X)
    assert np.array_equal(new_labels_compare, labels_compare)
    # Error on missing X for nearest-cluster
    with pytest.raises(ValueError):
        handle_noise(labels, "nearest_cluster")
    # Unknown strategy
    with pytest.raises(ValueError):
        handle_noise(labels, "unknown_strategy")

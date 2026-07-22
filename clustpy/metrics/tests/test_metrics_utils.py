from clustpy.metrics._metrics_utils import _check_labels_arrays, _check_length_data_and_labels
from clustpy.metrics._metrics_utils import (
    _assign_all_noise_points_to_one_cluster,
    _assign_each_noise_point_to_singleton_cluster,
    _remove_noise_point,
    _assign_all_noise_points_to_nearest_cluster,
    handle_noise,
)
import pytest
import numpy as np


# ============================================================
# Check labels
# ============================================================

def test_check_labels_arrays():
    l1 = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    assert l1.dtype == float
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    l1, l2 = _check_labels_arrays(l1, l2)
    assert l1.dtype == int and l2.dtype == int
    with pytest.raises(ValueError):
        _check_labels_arrays(l1, l2[1:])
    l3 = np.c_[l1, l2]
    with pytest.raises(ValueError):
        _check_labels_arrays(l1, l3)
    l1, l3 = _check_labels_arrays(l1, l3, allow_2d_labels=True)
    assert l1.shape == (10,) and l3.shape == (10, 2)
    l3, l1 = _check_labels_arrays(l3, l1, allow_2d_labels=True)
    assert l1.shape == (10,) and l3.shape == (10, 2)
    l3, l4 = _check_labels_arrays(l3, l3, allow_2d_labels=True)
    assert l3.shape == (10, 2) and l4.shape == (10, 2)


def test_check_length_data_and_labels():
    l1 = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    assert l1.dtype == float
    X = np.array(
        [
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
        ]
    )
    print(X.shape)
    X, l1 = _check_length_data_and_labels(X, l1)
    assert X.dtype == float and l1.dtype == int
    with pytest.raises(ValueError):
        _check_length_data_and_labels(X, l1[1:])
    with pytest.raises(ValueError):
        _check_length_data_and_labels(X, np.array([0] * 10))


# ============================================================
# Noise handling strategies
# ============================================================

def test_assign_all_noise_points_to_one_cluster():
    labels = np.array([0, 1, -1, 1, -1])
    new_labels = _assign_all_noise_points_to_one_cluster(labels)
    assert np.all(new_labels[labels == -1] == 2)  # max label is 1 -> new cluster 2
    assert new_labels[labels >= 0].tolist() == [0, 1, 1]  # non-noise unchanged

    # No noise case
    labels_no_noise = np.array([0, 1, 1])
    new_labels = _assign_all_noise_points_to_one_cluster(labels_no_noise)
    assert np.array_equal(new_labels, labels_no_noise)


def test_assign_each_noise_point_to_singleton_cluster():
    labels = np.array([0, -1, 1, -1])
    new_labels = _assign_each_noise_point_to_singleton_cluster(labels)
    # Original max label = 1, new singleton labels start at 2
    assert set(new_labels) == {0, 1, 2, 3}
    # Noise positions assigned unique labels
    assert new_labels[1] != new_labels[3]

    # No noise case
    labels_no_noise = np.array([0, 1, 1])
    new_labels = _assign_each_noise_point_to_singleton_cluster(labels_no_noise)
    assert np.array_equal(new_labels, labels_no_noise)


def test_remove_noise_point():
    labels = np.array([0, -1, 1, -1, 2])
    new_labels = _remove_noise_point(labels)
    assert -1 not in new_labels
    assert np.array_equal(new_labels, [0, 1, 2])

    # No noise case
    labels_no_noise = np.array([0, 1, 2])
    new_labels = _remove_noise_point(labels_no_noise)
    assert np.array_equal(new_labels, labels_no_noise)


def test_assign_all_noise_points_to_nearest_cluster():
    # Setup points
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6], [2, 2]])
    labels = np.array([0, 0, 1, 1, -1])  # last point is noise

    new_labels = _assign_all_noise_points_to_nearest_cluster(labels, X)
    # The last point [2,2] is closer to cluster 0
    assert new_labels[-1] == 0
    # Other points unchanged
    assert np.array_equal(new_labels[:-1], labels[:-1])

    # All points are noise -> should raise
    labels_all_noise = np.array([-1, -1])
    X_all_noise = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        _assign_all_noise_points_to_nearest_cluster(labels_all_noise, X_all_noise)


def test_handle_noise():
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6], [2, 2]])
    labels = np.array([0, 0, 1, 1, -1])

    # Keep strategy
    out = handle_noise(labels, "keep", X)
    new_labels, new_X = out
    assert np.array_equal(new_labels, labels)
    assert np.array_equal(new_X, X)

    # as_one_cluster
    out = handle_noise(labels, "as_one_cluster", X)
    new_labels, new_X = out
    assert new_labels[-1] == 2  # new cluster after max label 1
    assert np.array_equal(new_labels[:4], labels[:4])
    assert np.array_equal(new_X, X)

    # as_singletons
    out = handle_noise(labels, "as_singletons", X)
    new_labels, new_X = out
    assert new_labels[-1] == 2  # first singleton label after max 1
    # Non-noise labels unchanged
    assert np.array_equal(new_labels[:4], labels[:4])
    assert np.array_equal(new_X, X)

    # filter
    out = handle_noise(labels, "filter", X)
    new_labels, new_X = out
    assert -1 not in new_labels
    # Length reduced
    assert new_labels.shape[0] == new_X.shape[0] == 4
    # Non-noise labels preserved
    assert np.array_equal(new_labels, [0, 0, 1, 1])

    # to_nearest_cluster
    out = handle_noise(labels, "to_nearest_cluster", X)
    new_labels, new_X = out
    # Noise point [2,2] should be assigned cluster 0
    assert new_labels[-1] == 0
    assert np.array_equal(new_labels[:4], labels[:4])
    # X unchanged
    assert np.array_equal(new_X, X)

    # Error on missing X for nearest-cluster
    with pytest.raises(ValueError):
        handle_noise(labels, "to_nearest_cluster")

    # Unknown strategy
    with pytest.raises(ValueError):
        handle_noise(labels, "unknown_strategy")

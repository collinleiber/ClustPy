from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import create_subspace_data, create_nr_data
import numpy as np


def test_create_subspace_data():
    data, labels = create_subspace_data(1500, 5, [4, 3])
    _helper_test_data_loader((data, labels), 1500, 7, 5)
    # Test with different cluster sizes and outliers
    data, labels = create_subspace_data([200, 200, 300, 300, 500], 5, [4, 3], [10, 10])
    _helper_test_data_loader((data, labels), 1510, 7, 5, outliers=True)


def test_create_nr_data():
    data, labels = create_nr_data(1000, [3, 3, 5], [4, 3, 2], [100, 0, 0], rotate_space=False, random_state=0)
    _helper_test_data_loader((data, labels), 1100, 9, [3, 3, 5], outliers=[True, False, False])
    _, cluster_sizes = np.unique(labels[:, 0], return_counts=True)
    assert cluster_sizes[0] == 100  # Checks number of outliers
    # Check if random state is working
    data2, labels2 = create_nr_data(1000, [3, 3, 5], [4, 3, 2], [100, 0, 0], rotate_space=False, random_state=0)
    assert np.array_equal(data, data2)  # Should be the same
    assert np.array_equal(labels, labels2)  # Should be the same
    # Check rotation
    data2, labels2 = create_nr_data(1000, [3, 3, 5], [4, 3, 2], [100, 0, 0], rotate_space=True, random_state=0)
    assert data2.shape == (1100, 9)  # Shape should be the same
    assert np.array_equal(labels, labels2)  # Labels should also be the same
    assert not np.array_equal(data, data2)  # data should be different
    # Check different cluster sizes
    data3, labels3 = create_nr_data([[300, 400, 500], [400, 400, 450], [400, 400, 300, 100, 100]], [3, 3, 5], [4, 3, 2],
                                    [100, 50, 0], rotate_space=True, random_state=0)
    _helper_test_data_loader((data3, labels3), 1300, 9, [3, 3, 5], outliers=[True, True, False])
    for i in range(3):
        _, cluster_sizes = np.unique(labels3[:, i], return_counts=True)
        if i == 0:
            assert np.array_equal(cluster_sizes, [100, 300, 400, 500])
        elif i == 1:
            assert np.array_equal(cluster_sizes, [50, 400, 400, 450])
        else:
            assert np.array_equal(cluster_sizes, [400, 400, 300, 100, 100])

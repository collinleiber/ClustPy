from clustpy.metrics._metrics_utils import _check_labels_arrays, _check_length_data_and_labels
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
    l1, l3 =_check_labels_arrays(l1, l3)
    assert l1.shape == (10, ) and l3.shape == (10, 2)
    l3, l1 =_check_labels_arrays(l3, l1)
    assert l1.shape == (10, ) and l3.shape == (10, 2)
    l3, l4 =_check_labels_arrays(l3, l3)
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

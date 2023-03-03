import numpy as np
from clustpy.partition import DipExt, DipInit
from clustpy.partition.dipext import _dip_scaling, _n_starting_vectors_default, _angle, _get_random_modal_triangle
from clustpy.data import load_wine


def test_angle():
    assert _angle(np.array([1, 1, 1]), np.array([1, 1, 1])) == 0
    assert _angle(np.array([1, 0, 0]), np.array([0, 1, 0])) == 90
    assert _angle(np.array([1, 0, 0]), np.array([-1, 0, 0])) == 180


def test_n_starting_vectors_default():
    assert _n_starting_vectors_default(2) == 1
    assert _n_starting_vectors_default(3) == 2
    assert _n_starting_vectors_default(50) == 4


def test_dip_scaling():
    X = np.array([[0, 0, 10],
                  [2, 5, 12],
                  [4, 5, 14],
                  [6, 10, 16],
                  [8, 10, 18]])
    dip_values = np.array([0.5, 0.1, 0.2])
    subspace = _dip_scaling(X, dip_values)
    X_result = np.array([[0, 0, 0],
                         [0.25 * 0.5, 0.5 * 0.1, 0.25 * 0.2],
                         [0.5 * 0.5, 0.5 * 0.1, 0.5 * 0.2],
                         [0.75 * 0.5, 0.1, 0.75 * 0.2],
                         [0.5, 0.1, 0.2]])
    assert np.array_equal(subspace, X_result)


def test_get_random_modal_triangle():
    sorted_data = np.array([0, 1, 1, 1, 1, 2, 2, 3, 4])
    modal_triangle_orig = (1, 4, 6)
    for i in range(10):
        random_state = np.random.RandomState(i)
        modal_triangle = _get_random_modal_triangle(sorted_data, modal_triangle_orig, random_state)
        assert modal_triangle[0] > 0 and modal_triangle[0] < 4
        assert modal_triangle[1] > 1 and modal_triangle[0] < 5
        assert modal_triangle[2] > 4 and modal_triangle[2] < 7
        assert modal_triangle[0] < modal_triangle[1]
        assert modal_triangle[1] < modal_triangle[2]


"""
Tests regarding the DipExt object
"""


def test_simple_DipExt_with_wine():
    X, labels = load_wine()
    dipext = DipExt(random_state=1)
    subspace = dipext.fit_transform(X)
    assert subspace.shape[0] == X.shape[0]
    # Check if fit + transform equals fit_transform
    dipext2 = DipExt(random_state=1)
    dipext2.fit(X)
    subspace2 = dipext2.transform(X)
    assert subspace.shape == subspace2.shape
    assert np.array_equal(subspace, subspace2)
    X2 = X[:15]
    subspace = dipext.transform(X2)
    assert subspace.shape == (15, dipext.n_components)
    subspace2 = dipext2.transform(X2)
    assert subspace2.shape == (15, dipext2.n_components)
    assert np.array_equal(subspace, subspace2)


def test_DipExt_with_parameters_with_wine():
    X, labels = load_wine()
    # Check if input parameters are working
    dipext = DipExt(n_components=5, do_dip_scaling=False, consider_duplicates=True,
                    n_starting_vectors=5, random_state=2)
    subspace = dipext.fit_transform(X)
    assert subspace.shape == (X.shape[0], 5)
    # Check if random_state is working
    dipext = DipExt(consider_duplicates=True, random_state=1)
    subspace = dipext.fit_transform(X)
    dipext2 = DipExt(consider_duplicates=True, random_state=1)
    subspace2 = dipext2.fit_transform(X)
    assert np.array_equal(subspace, subspace2)


"""
Tests regarding the DipInit object
"""


def test_simple_DipInit_with_wine():
    X, labels = load_wine()
    dipinit = DipInit(3, random_state=1)
    assert not hasattr(dipinit, "labels_")
    dipinit.fit(X)
    assert dipinit.labels_.dtype == np.int32
    assert dipinit.labels_.shape == labels.shape

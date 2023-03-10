import numpy as np
from clustpy.partition import DipExt, DipInit
from clustpy.partition.dipext import _dip_scaling, _n_starting_vectors_default, _angle, \
    _ambiguous_modal_triangle_random, _get_ambiguous_modal_triangle_possibilities, _ambiguous_modal_triangle_all
from clustpy.data import create_subspace_data


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


def test_get_ambiguous_modal_triangle_possibilities():
    sorted_data = np.array([0, 1, 1, 1, 1, 2, 2, 3, 4])
    modal_triangle = (1, 4, 6)
    possibilities = _get_ambiguous_modal_triangle_possibilities(sorted_data, modal_triangle)
    assert possibilities[0] == [1, 2, 3, 4]
    assert possibilities[1] == [1, 2, 3, 4]
    assert possibilities[2] == [5, 6]


def test_consider_ambiguous_modal_triangle():
    data = np.array([1, 4, 1, 1, 1, 2, 0, 2, 3])
    sorted_indices_orig = np.array([6, 0, 2, 3, 4, 5, 7, 8, 1])
    sorted_data = np.array([0, 1, 1, 1, 1, 2, 2, 3, 4])
    modal_triangle = (1, 4, 6)
    for i in range(100):
        random_state = np.random.RandomState(i)
        sorted_indices = _ambiguous_modal_triangle_random(sorted_data, sorted_indices_orig.copy(),
                                                          modal_triangle, random_state)
        assert data[sorted_indices_orig[modal_triangle[0]]] == data[sorted_indices[modal_triangle[0]]]
        assert data[sorted_indices_orig[modal_triangle[1]]] == data[sorted_indices[modal_triangle[1]]]
        assert data[sorted_indices_orig[modal_triangle[2]]] == data[sorted_indices[modal_triangle[2]]]
        assert sorted_indices[modal_triangle[0]] != sorted_indices[modal_triangle[1]]
        assert data[sorted_indices[modal_triangle[0]]] == data[sorted_indices[modal_triangle[1]]]
        assert sorted_indices[modal_triangle[1]] != sorted_indices[modal_triangle[2]]
        assert data[sorted_indices[modal_triangle[1]]] != data[sorted_indices[modal_triangle[2]]]


def test_ambiguous_modal_triangle_all():
    X = np.array(
        [[1, 1, 0], [4, 1.1, 2], [1, 1.5, 4], [4, 1.7, 6], [1, 1.2, 8], [2, 1.9, 10], [0, 1.1, 12], [2, 2.9, 14], [3, 2.1, 16]])
    X_proj = np.array([1, 4, 1, 4, 1, 2, 0, 2, 3])
    X_proj_argsort = np.array([6, 0, 2, 4, 5, 7, 8, 1, 3])
    X_proj_sorted = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4])
    modal_triangle = (2, 4, 8)
    gradients = _ambiguous_modal_triangle_all(X, X_proj, X_proj_sorted, X_proj_argsort, modal_triangle)
    assert gradients.shape[0] == 2*2*3
    for i in range(gradients.shape[0] - 1):
        assert gradients[i].shape == (3,)
        for j in range(i + 1, gradients.shape[0]):
            assert gradients[j].shape == (3,)
            assert not np.array_equal(gradients[i], gradients[j])


"""
Tests regarding the DipExt object
"""


def test_simple_DipExt():
    X, labels = create_subspace_data(200, subspace_features=(3, 5), random_state=1)
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


def test_DipExt_with_parameters():
    X, labels = create_subspace_data(200, subspace_features=(3, 5), random_state=1)
    # Check if input parameters are working
    dipext = DipExt(n_components=3, do_dip_scaling=False, ambiguous_triangle_strategy="random",
                    n_starting_vectors=2, random_state=1)
    subspace = dipext.fit_transform(X)
    assert subspace.shape == (X.shape[0], 3)
    # Test all gradients
    dipext = DipExt(n_components=3, do_dip_scaling=False, ambiguous_triangle_strategy="all",
                    n_starting_vectors=2, random_state=1)
    subspace = dipext.fit_transform(X)
    assert subspace.shape == (X.shape[0], 3)
    # Check if random_state is working
    dipext = DipExt(ambiguous_triangle_strategy="random", random_state=1)
    subspace = dipext.fit_transform(X)
    dipext2 = DipExt(ambiguous_triangle_strategy="random", random_state=1)
    subspace2 = dipext2.fit_transform(X)
    assert np.array_equal(subspace, subspace2)


"""
Tests regarding the DipInit object
"""


def test_simple_DipInit():
    X, labels = create_subspace_data(200, subspace_features=(3, 5), random_state=1)
    dipinit = DipInit(3, random_state=1)
    assert not hasattr(dipinit, "labels_")
    dipinit.fit(X)
    assert dipinit.labels_.dtype == np.int32
    assert dipinit.labels_.shape == labels.shape

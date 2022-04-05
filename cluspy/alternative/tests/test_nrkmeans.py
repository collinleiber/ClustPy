import numpy as np
import cluspy.alternative.nrkmeans as nrk


def test_assign_labels():
    X = np.array([[1, 1, 1], [1, 2, 2], [2, 3, 1], [4, 5, 4], [5, 5, 5], [4, 5, 6], [10, 11, 11], [12, 10, 11]])
    V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    P_subspace = [0, 1, 2]
    centers_subspace = np.array([[1, 1, 1], [4, 4, 4], [10, 10, 10]])
    labels = nrk._assign_labels(X, V, centers_subspace, P_subspace)
    expected = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    assert np.array_equal(labels, expected) == True


def test_are_labels_equal():
    # First Test
    labels_1 = np.array([[0, 0, 1, 2, 1, 1, 0, 2]]).reshape((-1, 1))
    labels_2 = np.array([[0, 0, 1, 2, 1, 1, 0, 2]]).reshape((-1, 1))
    assert nrk._are_labels_equal(labels_1, labels_2) == True
    # Second Test
    labels_3 = np.array([[0, 0, 1, 2, 1, 1, 2, 2]]).reshape((-1, 1))
    assert nrk._are_labels_equal(labels_1, labels_3) == False
    # Third Test
    labels_4 = np.array([[1, 1, 2, 0, 2, 2, 1, 0]]).reshape((-1, 1))
    assert nrk._are_labels_equal(labels_1, labels_4) == True
    # Fourth test
    assert nrk._are_labels_equal(None, labels_1) == False
    assert nrk._are_labels_equal(labels_1, None) == False


def test_is_matrix_orthogonal():
    # First Test
    orthogonal_matrix = np.array([[-1.0, 0.0], [0.0, 1.0]])
    assert nrk._is_matrix_orthogonal(orthogonal_matrix) == True
    # Second Test
    orthogonal_matrix = np.array([[0.0, -0.8, -0.6],
                                  [0.8, -0.36, 0.48],
                                  [0.6, 0.48, -0.64]])
    assert nrk._is_matrix_orthogonal(orthogonal_matrix) == True
    # Third Test
    orthogonal_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert nrk._is_matrix_orthogonal(orthogonal_matrix) == True


def test_is_matrix_orthogonal_false():
    # First test - wrong dimensionality
    not_orthogonal_matrix = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert nrk._is_matrix_orthogonal(not_orthogonal_matrix) == False
    # Second test
    not_orthogonal_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert nrk._is_matrix_orthogonal(not_orthogonal_matrix) == False
    # Third test
    not_orthogonal_matrix = np.array([[-0.85616, 0.46933], [0.46933, 0.96236]])
    assert nrk._is_matrix_orthogonal(not_orthogonal_matrix) == False


def test_is_matrix_symmetric():
    # First test
    symmetric_matrix = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]])
    assert nrk._is_matrix_symmetric(symmetric_matrix) == True
    # Second Test
    symmetric_matrix = np.array(
        [[0.234234, 0.87564, 0.123414, 0.74573],
         [0.87564, 0.5436346, 0.456364, 0.123],
         [0.123414, 0.456364, 0.23452, 0.23423],
         [0.74573, 0.123, 0.23423, 0.26]])
    assert nrk._is_matrix_symmetric(symmetric_matrix) == True


def test_is_matrix_symmetric_false():
    # First test - wrong dimensionality
    not_symmetric_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert nrk._is_matrix_symmetric(not_symmetric_matrix) == False
    # Second test
    not_symmetric_matrix = np.array([[1.0, 0.32454], [0.32453, 1.0]])
    assert nrk._is_matrix_symmetric(not_symmetric_matrix) == False
    # Third test
    not_symmetric_matrix = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 3.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]])
    assert nrk._is_matrix_symmetric(not_symmetric_matrix) == False


def test_create_full_rotation_matrix():
    # First test
    dimensionality = 6
    transitions = [0, 3, 1, 4]
    V_C = np.array([[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]])
    V_F = nrk._create_full_rotation_matrix(dimensionality, transitions, V_C)
    V_F_check = np.array(
        [[1.0, 9.0, 0, 5.0, 13.0, 0], [3.0, 11.0, 0, 7.0, 15.0, 0], [0, 0, 1, 0, 0, 0],
         [2.0, 10.0, 0, 6.0, 14.0, 0],
         [4.0, 12.0, 0, 8.0, 16.0, 0], [0, 0, 0, 0, 0, 1]])
    assert np.array_equal(V_F, V_F_check) == True
    # Second test
    dimensionality = 6
    transitions = [1, 4, 5, 2, 3]
    V_C = np.array([[1.0, 6.0, 11.0, 16.0, 21.0], [2.0, 7.0, 12.0, 17.0, 22.0], [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0], [5.0, 10.0, 15.0, 20.0, 25.0]])
    V_F = nrk._create_full_rotation_matrix(dimensionality, transitions, V_C)
    V_F_check = np.array(
        [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 16.0, 21.0, 6.0, 11.0], [0, 4.0, 19.0, 24.0, 9.0, 14.0],
         [0, 5.0, 20.0, 25.0, 10.0, 15.0],
         [0, 2.0, 17.0, 22.0, 7.0, 12.0], [0, 3.0, 18.0, 23.0, 8.0, 13.0]])
    assert np.array_equal(V_F, V_F_check) == True


def test_update_projections():
    # First test
    transitions = np.array([0, 3, 1, 4])
    n_negative_e = 2
    P_1, P_2 = nrk._update_projections(transitions, n_negative_e)
    assert np.array_equal(P_1, np.array([0, 3])) == True
    assert np.array_equal(P_2, np.array([4, 1])) == True
    # Second test
    transitions = np.array([1, 4, 5, 2, 3])
    n_negative_e = 2
    P_1, P_2 = nrk._update_projections(transitions, n_negative_e)
    assert np.array_equal(P_1, np.array([1, 4])) == True
    assert np.array_equal(P_2, np.array([3, 2, 5])) == True

from cluspy.data import *
import numpy as np


def test_create_subspace_data():
    data, labels = create_subspace_data(1500, 5, [4, 3])
    assert data.shape == (1500, 7)
    assert labels.shape == (1500,)
    assert np.array_equal(np.unique(labels), range(5))


def test_create_nr_data():
    data, labels = create_nr_data(1000, [3, 3, 5], [4, 3, 2], [100, 0, 0], rotate_space=False, random_state=0)
    assert data.shape == (1000, 9)
    assert labels.shape == (1000, 3)
    unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
    assert [len(l) for l in unique_labels] == [4, 3, 5]
    for i, ul in enumerate(unique_labels):
        if i == 0:
            assert np.array_equal(ul, range(-1, len(ul) - 1))
        else:
            assert np.array_equal(ul, range(len(ul)))
    _, n_outliers = np.unique(labels[:, 0], return_counts=True)
    assert n_outliers[0] == 100
    # Check if random state is working
    data2, labels2 = create_nr_data(1000, [3, 3, 5], [4, 3, 2], [100, 0, 0], rotate_space=False, random_state=0)
    assert np.array_equal(data, data2) # Should be the same
    assert np.array_equal(labels, labels2) # Should be the same
    # Check rotation
    data2, labels2 = create_nr_data(1000, [3, 3, 5], [4, 3, 2], [100, 0, 0], rotate_space=True, random_state=0)
    assert data2.shape == (1000, 9) # Shape should be the same
    assert np.array_equal(labels, labels2) # Labels should also be the same
    assert not np.array_equal(data, data2) # data should be different

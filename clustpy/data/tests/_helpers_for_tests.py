import numpy as np


def _helper_test_data_loader(data, labels, N, d, k, outliers=False):
    """
    Test loading of datasets.

    Parameters
    ----------
    data : np.ndarray
        The data array
    labels : np.ndarray
        The labels array
    N : int
        The number of data objects
    d : int
        The dimensionality of the dataset
    k : int
        The number of clusters. Should be a list for datasets with multiple labelings
    outliers : bool
        Defines if outliers are contained in the dataset. Should be a list for datasets with multiple labelings (default: False)
    """
    assert data.shape == (N, d)
    assert np.issubdtype(labels.dtype, np.integer)
    if type(k) is int:
        assert labels.shape == (N,)
        assert np.array_equal(np.unique(labels), range(k) if not outliers else range(-1, k))
    else:
        if type(outliers) is bool:
            outliers = [outliers] * len(k)
        # In case of datasets for alternative clusterings
        assert labels.shape == (N, len(k))
        unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
        assert [len(l) for l in unique_labels] == [k[i] if not outliers[i] else k[i] + 1 for i in
                                                   range(labels.shape[1])]  # Checks that number of labels is correct
        for i, ul in enumerate(unique_labels):
            assert np.array_equal(ul, range(k[i]) if not outliers[i] else range(-1, k[
                i]))  # Checks that labels go from 0 to k


def _check_normalized_channels(data, channels, should_be_normalized=True):
    imprecision = 1e-4
    # Check is simple if we only have a single channel, i.e. a grayscale image
    if channels == 1:
        if should_be_normalized:
            assert np.mean(data) < imprecision
            assert abs(np.std(data) - 1) < imprecision
        else:
            assert np.mean(data) > imprecision
            assert abs(np.std(data) - 1) > imprecision
    else:
        # Else we have to check each channel separately
        for i in range(channels):
            if should_be_normalized:
                assert np.mean(data[:, np.arange(data.shape[1]) % channels == i]) < imprecision
                assert abs(np.std(data[:, np.arange(data.shape[1]) % channels == i]) - 1) < imprecision
            else:
                assert np.mean(data[:, np.arange(data.shape[1]) % channels == i]) > imprecision
                assert abs(np.std(data[:, np.arange(data.shape[1]) % channels == i]) - 1) > imprecision

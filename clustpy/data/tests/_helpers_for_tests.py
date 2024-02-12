import numpy as np
from collections.abc import Callable
from sklearn.datasets._base import Bunch


def _helper_test_data_loader(dataloader: Callable, N: int, d: int, k: int, outliers: bool = False,
                             dataloader_params: dict = None) -> Bunch:
    """
    Test loading of datasets.

    Parameters
    ----------
    dataloader : Callable
        The data loader function. Can also be a tuple containing (data, labels)
    N : int
        The number of data objects
    d : int
        The dimensionality of the dataset
    k : int
        The number of clusters. Should be a list for datasets with multiple labelings
    outliers : bool
        Defines if outliers are contained in the dataset. Should be a list for datasets with multiple labelings (default: False)
    dataloader_params : dict
        The parameters for the dataloader function

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the loaded dataset
    """
    if callable(dataloader):
        if dataloader_params is None:
            dataloader_params = {}
        # Check if Bunch object and data, labels contain the same arrays
        dataset = dataloader(**dataloader_params)
        dataloader_params["return_X_y"] = True
        data, labels = dataloader(**dataloader_params)
        assert np.array_equal(data, dataset.data)
        assert np.array_equal(labels, dataset.target)
        assert type(dataset.dataset_name) is str
    else:
        data, labels = dataloader
        dataset = None
    # Check actual data
    assert (N is None and data.shape[1] == d) or (
            N is not None and data.shape == (N, d)), "data shape should be {0} but is {1}".format((N, d),
                                                                                                  data.shape)
    assert np.issubdtype(labels.dtype, np.integer)
    if type(k) is int:
        assert labels.shape == (data.shape[0],), "labels shape should be {0} but is {1}".format((data.shape[0],),
                                                                                                labels.shape)
        assert np.array_equal(np.unique(labels), range(k) if not outliers else range(-1, k))
    else:
        if type(outliers) is bool:
            outliers = [outliers] * len(k)
        # In case of datasets for alternative clusterings
        assert labels.shape == (data.shape[0], len(k))
        unique_labels = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
        assert [len(l) for l in unique_labels] == [k[i] if not outliers[i] else k[i] + 1 for i in
                                                   range(labels.shape[1])]  # Checks that number of labels is correct
        for i, ul in enumerate(unique_labels):
            assert np.array_equal(ul, range(k[i]) if not outliers[i] else range(-1, k[
                i]))  # Checks that labels go from 0 to k
    # Return the loaded dataset
    return dataset

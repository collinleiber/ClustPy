from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.utils import check_X_y, check_array, check_random_state


def check_clustpy_estimator(estimator_obj: BaseEstimator, checks_to_ignore: tuple | list = ("check_complex_data")):
    """
    Run the check_estimator function from sklearn ignoring the check for complex data.
    For more information, check: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py

    Parameters
    ----------
    estimator_obj : BaseEstimator
        Initialization of the tested BaseEstimator
    checks_to_ignore : tuple | list
        List containing the names of checks to ignore (default: ("check_complex_data"))
    """
    all_checks = check_estimator(estimator_obj, True)
    for estimator, check in all_checks:
        check_name = check.func.__name__
        if not check_name in checks_to_ignore:
            try:
                check(estimator)
            except Exception as e:
                print("Check", check_name, "failed.")
                raise e
        else:
            print("Skip check:", check_name)


def check_parameters(X: np.ndarray, *, y: np.ndarray=None, random_state: np.random.RandomState | int=None,
                     allow_nd: bool=False, allow_size_1: bool=False, n_features_in: int=None) -> (np.ndarray, np.ndarray, np.random.RandomState):
    """
    Check if parameters for X, y and random_state are defined in accordance with the sklearn standard.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    y : np.ndarray
        the labels (can usually be ignored) (default: None)
    random_state : np.random.RandomState | int
        the random state (default: None)
    allow_nd : bool
        allow n-dimensional arrays instead of only allowing 2d arrays (default: False)
    allow_size_1 : bool
        allow a dataset with a single sample
    n_features_in : int
        compare this value with the number of features in X

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.random.RandomState)
        the checked data set,
        the checked labels
        the checked random_state
    """
    ensure_2d = not allow_nd
    if y is None:
        X = check_array(X, accept_sparse=False, allow_nd=allow_nd, ensure_2d=ensure_2d)
    else:
        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=allow_nd, ensure_2d=ensure_2d)
        class_labels = np.unique(y)
        if np.min(class_labels) == 1 and np.max(class_labels) == len(class_labels):
            y -= 1
            class_labels -= 1
            print("WARNING: labels in y were within [1, {0}], changed to be within [0, {1}] instead".format(len(class_labels), len(class_labels) - 1))
        assert np.array_equal(class_labels, np.arange(len(class_labels))), "y is not defined as expected. Should only contain labels within [0, n_classes - 1]. Labels in y: {0}".format(class_labels)
    if X.ndim == 1:
        raise ValueError("Data can not be a 1d array.")
    if not allow_size_1 and X.shape[0] == 1:
        raise ValueError("Model cannot be fitted if n_samples = 1. X shape =", X.shape)
    if n_features_in is not None and n_features_in != X.shape[1]:
        raise ValueError("Number of features in X is not correct. X has {0} features but {1} is expceted.".format(X.shape[1], n_features_in))
    random_state = check_random_state(random_state)
    return X, y, random_state

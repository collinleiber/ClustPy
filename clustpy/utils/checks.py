from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.utils import check_X_y, check_array, check_random_state


def check_estimator_wo_complex_data(estimator_obj: BaseEstimator):
    """
    Run the check_estimator function from sklearn ignoring the check for complex data.

    Parameters
    ----------
    estimator_obj : BaseEstimator
        Initialization of the tested BaseEstimator
    """
    all_checks = check_estimator(estimator_obj, True)
    for estimator, check in all_checks:
        check_name = check.func.__name__
        if check_name != "check_complex_data":
            try:
                check(estimator)
            except Exception as e:
                print("Check", check_name, "failed.")
                raise e


def check_parameters(X: np.ndarray, *, y: np.ndarray=None, random_state: np.random.RandomState | int=None,
                     allow_nd: bool=False) -> (np.ndarray, np.ndarray, np.random.RandomState):
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
    random_state = check_random_state(random_state)
    return X, y, random_state

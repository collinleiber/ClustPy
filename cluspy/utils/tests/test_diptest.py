from cluspy.utils import dip_test, dip_pval, dip_boot_samples, plot_dip, dip_gradient, dip_pval_gradient
from cluspy.utils.diptest import _dip_c_impl, _dip_python_impl, _dip_pval_function, _dip_pval_table, \
    _get_dip_table_values
import numpy as np
from unittest.mock import patch


def test_dip_test():
    # Multimodal Example
    X = np.array([1] * 10 + [2] * 10)
    dip = dip_test(X)
    assert 0.25 == np.round(dip, 7)
    # Unimodal Example
    X = np.array([1] * 20)
    dip = dip_test(X)
    assert 0.00 == np.round(dip, 7)
    # More complex example (see: https://stats.stackexchange.com/questions/203556/hartigans-dip-test-significant-for-bimodality-with-clearly-unimodal-data)
    X = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2.5, 2.5, 2.5])
    dip = dip_test(X)
    assert 0.1363636 == np.round(dip, 7)
    # Even more complex example (statfaculty data)
    X = np.array(
        [1, 0, 0, 1, 0, 1, 1, 2, 0, 5, 4, 1, 1, 3, 2, 2, 2, 2, 3, 1, 2, 1, 2, 5, 2, 0, 0, 2, 0, 1, 3, 4, 4, 1, 0, 0, 1,
         0, 0, 0, 1, 0, 2])
    dip = dip_test(X)
    assert 0.1395349 == np.round(dip, 7)
    # Second more complex example
    X = np.array([-2, 0, 0.25, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 4.5])
    dip = dip_test(X)
    assert 0.0535714 == np.round(dip, 7)
    # Reverse
    dip_reverse = dip_test(X[::-1])
    assert dip == dip_reverse
    # Shifted and scaled
    dip_shifted = dip_test((X - 5) / 3)
    assert dip == dip_shifted
    # Test modal interval and modal triangle
    dip_param, modal_interval, modal_triangle = dip_test(np.sort(X), just_dip=False, is_data_sorted=True)
    assert dip == dip_param
    assert modal_interval == (4, 9)
    assert modal_triangle == (1, 3, 4)
    # Test also gcm, lcm, mn and mj
    dip_param, modal_interval, modal_triangle, gcm, lcm, mn, mj = dip_test(np.sort(X), just_dip=False,
                                                                           is_data_sorted=True,
                                                                           return_gcm_lcm_mn_mj=True, use_c=True)
    assert dip == dip_param
    assert modal_interval == (4, 9)
    assert modal_triangle == (1, 3, 4)
    assert np.array_equal(gcm, np.array([9, 5, 4, 1, 0]))
    assert np.array_equal(lcm, np.array([4, 8, 9, 10, 13]))
    assert np.array_equal(mn, np.array([0, 0, 1, 1, 1, 4, 5, 5, 7, 5, 4, 4, 1, 1]))
    assert np.array_equal(mj, np.array([9, 9, 9, 9, 8, 6, 8, 8, 9, 10, 13, 13, 13, 13]))
    # Test if python implementation returns the same result
    dip_param2, modal_interval2, modal_triangle2, gcm2, lcm2, mn2, mj2 = dip_test(np.sort(X), just_dip=False,
                                                                                  is_data_sorted=True,
                                                                                  return_gcm_lcm_mn_mj=True,
                                                                                  use_c=False)
    assert dip == dip_param2
    assert modal_interval2 == modal_interval
    assert modal_triangle2 == modal_triangle
    assert np.array_equal(gcm2, gcm)
    assert np.array_equal(lcm2, lcm)
    assert np.array_equal(mn2, mn)
    assert np.array_equal(mj2, mj)


def test_diptest_python_impl_matches_c_impl_with_random_data():
    # Random data
    X = np.sort(np.random.rand(50))
    dip_py, modal_interval_py, modal_triangle_py, gcm_py, lcm_py, mn_py, mj_py = _dip_python_impl(X, debug=False)
    dip_c, modal_interval_c, modal_triangle_c, gcm_c, lcm_c, mn_c, mj_c = _dip_c_impl(X, debug=False)
    # Are results sensible?
    assert dip_py >= 0 and dip_py <= 0.25
    assert len(modal_interval_py) == 2
    assert modal_interval_py[0] < modal_interval_py[1] and modal_interval_py[0] >= 0 and modal_interval_py[1] < X.shape[
        0]
    assert len(modal_triangle_py) == 3
    assert modal_triangle_py[0] <= modal_triangle_py[1] and modal_triangle_py[1] <= modal_triangle_py[2] and \
           modal_triangle_py[0] >= 0 and modal_triangle_py[1] < X.shape[0]
    assert gcm_py.shape[0] <= X.shape[0] and np.min(gcm_py) >= 0 and np.max(gcm_py) < X.shape[0]
    assert lcm_py.shape[0] <= X.shape[0] and np.min(lcm_py) >= 0 and np.max(lcm_py) < X.shape[0]
    assert mn_py.shape[0] == X.shape[0] and np.min(mn_py) >= 0 and np.max(mn_py) < X.shape[0]
    assert mj_py.shape[0] == X.shape[0] and np.min(mj_py) >= 0 and np.max(mj_py) < X.shape[0]
    # Are all results equal?
    assert dip_py == dip_c
    assert modal_interval_py == modal_interval_c
    assert modal_triangle_py == modal_triangle_c
    assert np.array_equal(gcm_py, gcm_c)
    assert np.array_equal(lcm_py, lcm_c)
    assert np.array_equal(mn_py, mn_c)
    assert np.array_equal(mj_py, mj_c)


def test_dip_pval():
    random_state = np.random.RandomState(1)
    # Multimodal Example
    X = np.array([1] * 10 + [2] * 10)
    dip = dip_test(X)
    pval_table = dip_pval(dip, X.shape[0], "table")
    pval_function = dip_pval(dip, X.shape[0], "function")
    pval_boot = dip_pval(dip, X.shape[0], "bootstrap", random_state=random_state)
    assert 0 == np.round(pval_table, 5)
    assert 0 == np.round(pval_function, 5)
    assert 0 == np.round(pval_boot, 5)
    # Unimodal Example
    X = np.array([1] * 20)
    dip = dip_test(X)
    pval_table = dip_pval(dip, X.shape[0], "table")
    pval_function = dip_pval(dip, X.shape[0], "function")
    pval_boot = dip_pval(dip, X.shape[0], "bootstrap", random_state=random_state)
    assert 1 == np.round(pval_table, 5)
    assert 1 == np.round(pval_function, 5)
    assert 1 == np.round(pval_boot, 5)
    # More complex example (see: https://stats.stackexchange.com/questions/203556/hartigans-dip-test-significant-for-bimodality-with-clearly-unimodal-data)
    X = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2.5, 2.5, 2.5])
    dip = dip_test(X)
    pval_table = dip_pval(dip, X.shape[0])
    assert 0.04419 == np.round(pval_table, 5)
    # Even more complex example (statfaculty data)
    X = np.array(
        [1, 0, 0, 1, 0, 1, 1, 2, 0, 5, 4, 1, 1, 3, 2, 2, 2, 2, 3, 1, 2, 1, 2, 5, 2, 0, 0, 2, 0, 1, 3, 4, 4, 1, 0, 0, 1,
         0, 0, 0, 1, 0, 2])
    dip = dip_test(X)
    pval_table = dip_pval(dip, X.shape[0], "table")
    pval_function = dip_pval(dip, X.shape[0], "function")
    pval_boot = dip_pval(dip, X.shape[0], "bootstrap", random_state=random_state)
    assert pval_table < 1e-5
    assert pval_function < 1e-4
    assert pval_boot < 1e-5


def test_pval_table_matches_function_matches_bootstrap_with_random_data():
    random_state = np.random.RandomState(1)
    # Random data
    X = np.sort(np.random.rand(50))
    dip = dip_test(X)
    pval_table = _dip_pval_table(dip, X.shape[0])
    pval_function = _dip_pval_function(dip, X.shape[0])
    pval_boot = dip_pval(dip, X.shape[0], "bootstrap", random_state=random_state)
    assert 1 >= np.round(pval_table, 5) and 0 <= np.round(pval_table, 5)
    assert 1 >= np.round(pval_function, 5) and 0 <= np.round(pval_function, 5)
    assert 1 >= np.round(pval_boot, 5) and 0 <= np.round(pval_boot, 5)
    assert np.abs(pval_table - pval_function) < 0.1
    assert np.abs(pval_table - pval_boot) < 0.1
    assert np.abs(pval_function - pval_boot) < 0.1


def test_dip_gradient():
    n_dims = 3
    X = np.random.rand(50, n_dims)
    proj = np.random.rand(n_dims)
    X_proj = np.matmul(X, proj)
    argsorted = np.argsort(X_proj)
    dip, modal_interval, modal_triangle = dip_test(X_proj[argsorted], just_dip=False, is_data_sorted=True, use_c=False)
    grad = dip_gradient(X, X_proj, argsorted, modal_triangle)
    assert grad.shape == (n_dims,)


def test_dip_pval_gradient():
    n_dims = 3
    X = np.random.rand(50, n_dims)
    proj = np.random.rand(n_dims)
    X_proj = np.matmul(X, proj)
    argsorted = np.argsort(X_proj)
    dip, modal_interval, modal_triangle = dip_test(X_proj[argsorted], just_dip=False, is_data_sorted=True, use_c=False)
    grad = dip_pval_gradient(X, X_proj, argsorted, modal_triangle, dip)
    assert grad.shape == (n_dims,)


def test_dip_boot_samples():
    random_state = np.random.RandomState(1)
    n_boots = 100
    dips = dip_boot_samples(50, n_boots, random_state)
    assert dips.shape[0] == n_boots
    assert np.all(dips >= 0) and np.all(dips <= 0.25)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_dip(mock_fig):
    X = np.sort(np.r_[np.random.rand(50), np.random.rand(50) + 1.3])
    L = np.array([-1] + [0] * 49 + [1] * 49 + [-1])
    dip, modal_interval, modal_triangle, gcm, lcm, mn, mj = dip_test(X, is_data_sorted=True, just_dip=False,
                                                                     return_gcm_lcm_mn_mj=True)
    assert None == plot_dip(X, False, dip, modal_interval, modal_triangle, gcm, lcm, True, True, L, True, True, 20,
                            (1, 1), True)


def test_dip_table_values():
    N, SIG, CV = _get_dip_table_values()
    assert N.shape == (21,)
    assert SIG.shape == (26,)
    assert CV.shape == (21, 26)

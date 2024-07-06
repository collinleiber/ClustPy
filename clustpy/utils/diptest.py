try:
    from clustpy.utils.dipModule import c_diptest  # noqa - Import from C file (could be marked as unresolved)
except:
    print("[WARNING] Could not import c_diptest in clustpy.utils.dipModule")
import numpy as np
import matplotlib.pyplot as plt
from clustpy.utils.plots import plot_histogram
from sklearn.utils import check_random_state


def dip_test(X: np.ndarray, just_dip: bool = True, is_data_sorted: bool = False, return_gcm_lcm_mn_mj: bool = False,
             use_c: bool = True, debug: bool = False) -> (
        float, tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the Dip-value. This can either be done using the C implementation or the python version.
    In addition to the Dip-value additional values can be returned.
    These are e.g. the modal interval (indices of the beginning and end of the steepest slop of the ECDF) and the modal interval (used to calculate the gradient of the Dip-value) if just_dip is False.
    Further, the indices of the Greatest Convex Minorant (gcm), Least Concave Majorant (lcm), minorant and majorant values can be returned by setting return_gcm_lcm_mn_mj to True.
    Note that modal_triangle can be (-1,-1,-1) if the triangle could not be determined correctly.

    Parameters
    ----------
    X : np.ndarray
        the given univariate data set
    just_dip : bool
        Defines whether only the Dip-value should be returned or also the modal interval and modal triangle (default: True)
    is_data_sorted : bool
        Should be True if the data set is already sorted (default: False)
    return_gcm_lcm_mn_mj : bool
        Defines whether the gcm, lcm, mn and mj arrays should be returned. In this case just_dip must be False (default: False)
    use_c : bool
        Defines whether the C implementation should be used (defualt: True)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Returns
    -------
    tuple: (float, tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The resulting Dip-value,
        The indices of the modal_interval - corresponds to the steepest slope in the ECDF (if just_dip is False),
        The indices of the modal triangle (if just_dip is False),
        The indices of points that are part of the Greatest Convex Minorant (gcm) (if just_dip is False and return_gcm_lcm_mn_mj is True),
        The indices of points that are part of the Least Concave Majorant (lcm) (if just_dip is False and return_gcm_lcm_mn_mj is True),
        The minorant values (if just_dip is False and return_gcm_lcm_mn_mj is True),
        The majorant values (if just_dip is False and return_gcm_lcm_mn_mj is True)

    References
    ----------
    Hartigan, John A., and Pamela M. Hartigan.
    "The dip test of unimodality." The annals of Statistics (1985): 70-84.

    and

    Hartigan, P. M. "Computation of the dip statistic to test for unimodality: Algorithm as 217."
    Applied Statistics 34.3 (1985): 320-5.
    """
    assert X.ndim == 1, "Data must be 1-dimensional for the dip-test. Your shape:{0}".format(X.shape)
    assert just_dip or is_data_sorted == True, "Data must be sorted if modal interval and/or modal triangle should be returned (else indices will not match)"
    assert not return_gcm_lcm_mn_mj or not just_dip, "If GCM, LCM, mn and mj should be returned, 'just_dip' must be False"
    if not is_data_sorted:
        X = np.sort(X)
    # Obtain results
    if X.shape[0] < 4 or X[0] == X[-1]:
        dip_value = 0.0
        modal_interval = (0, X.shape[0] - 1)
        modal_triangle = (-1, -1, -1)
        mn, mj = None, None
    elif use_c:
        try:
            dip_value, modal_interval, modal_triangle, _, _, mn, mj = _dip_c_impl(X, debug)
        except Exception as e:
            print("[WARNING] C implementation can not be used for dip calculation.")
            print(e)
            dip_value, modal_interval, modal_triangle, _, _, mn, mj = _dip_python_impl(X, debug)
    else:
        dip_value, modal_interval, modal_triangle, _, _, mn, mj = _dip_python_impl(X, debug)
    # Return results
    if just_dip:
        return dip_value
    elif return_gcm_lcm_mn_mj:
        if mn is not None and mj is not None:
            gcm, lcm = _get_complete_gcm_lcm(mn, mj, modal_interval)
        else:
            gcm, lcm = None, None
        return dip_value, modal_interval, modal_triangle, gcm, lcm, mn, mj
    else:
        return dip_value, modal_interval, modal_triangle


def _dip_c_impl(X: np.ndarray, debug: bool) -> (float, tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calls the Dip C implementation by Martin Maechler.

    Parameters
    ----------
    X : np.ndarray
        the given univariate data set
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (float, tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The resulting Dip-value,
        The indices of the modal_interval - corresponds to the steepest slope in the ECDF,
        The indices of the modal triangle
        The indices of points that are part of the Greatest Convex Minorant (gcm),
        The indices of points that are part of the Least Concave Majorant (lcm),
        The minorant values,
        The majorant values
    """
    # Create reference numpy arrays
    modal_interval = np.zeros(2, dtype=np.int32)
    modal_triangle = -np.ones(3, dtype=np.int32)
    gcm = np.zeros(X.shape, dtype=np.int32)
    lcm = np.zeros(X.shape, dtype=np.int32)
    mj = np.zeros(X.shape, dtype=np.int32)
    mn = np.zeros(X.shape, dtype=np.int32)
    # Execute C function
    dip_value = c_diptest(X.astype(np.float64), modal_interval, modal_triangle, gcm, lcm, mn, mj, X.shape[0],
                          1 if debug else 0)
    return dip_value, (modal_interval[0], modal_interval[1]), (
        modal_triangle[0], modal_triangle[1], modal_triangle[2]), gcm, lcm, mn, mj


def _dip_python_impl(X: np.ndarray, debug: bool) -> (
        float, tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    A python version of the Dip C implementation by Martin Maechler.

    Parameters
    ----------
    X : np.ndarray
        the given univariate data set
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (float, tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The resulting Dip-value,
        The indices of the modal_interval - corresponds to the steepest slope in the ECDF,
        The indices of the modal triangle
        The indices of points that are part of the Greatest Convex Minorant (gcm),
        The indices of points that are part of the Least Concave Majorant (lcm),
        The minorant values,
        The majorant values
    """
    assert X.ndim == 1, "Data must be 1-dimensional for the dip-test. Your shape:{0}".format(X.shape)
    N = len(X)
    if N < 4 or X[0] == X[-1]:
        d = 0.0
        return d, (0, 0), (-1, -1, -1), None, None, None, None
    low = 0
    high = N - 1
    dip_value = 0.0
    # Create modal triangle
    modaltriangle_i1 = -1
    modaltriangle_i2 = -1
    modaltriangle_i3 = -1
    # Establish the indices mn[0..n-1] over which combination is necessary for the convex MINORANT (GCM) fit.
    mn = np.zeros(N, dtype=np.int32)
    mn[0] = 0
    for j in range(1, N):
        mn[j] = j - 1
        while True:
            mnj = mn[j]
            mnmnj = mn[mnj]
            if mnj == 0 or (X[j] - X[mnj]) * (mnj - mnmnj) < (X[mnj] - X[mnmnj]) * (j - mnj):
                break
            mn[j] = mnmnj
    # Establish the indices   mj[0..n-1]  over which combination is necessary for the concave MAJORANT (LCM) fit.
    mj = np.zeros(N, dtype=np.int32)
    mj[N - 1] = N - 1
    for k in range(N - 2, -1, -1):
        mj[k] = k + 1
        while True:
            mjk = mj[k]
            mjmjk = mj[mjk]
            if mjk == N - 1 or (X[k] - X[mjk]) * (mjk - mjmjk) < (X[mjk] - X[mjmjk]) * (k - mjk):
                break
            mj[k] = mjmjk
    gcm = np.zeros(N, dtype=int)  # np.arange(N)
    lcm = np.zeros(N, dtype=int)  # np.arange(N, -1, -1)
    while True:
        # GCM
        gcm[0] = high
        i = 0
        while gcm[i] > low:
            gcm[i + 1] = mn[gcm[i]]
            i += 1
        ig = i
        l_gcm = i
        ix = ig - 1
        # LCM
        lcm[0] = low
        i = 0
        while lcm[i] < high:
            lcm[i + 1] = mj[lcm[i]]
            i += 1
        ih = i
        l_lcm = i
        iv = 1
        if debug:
            print("'dip': LOOP-BEGIN: 2n*D= {0}  [low,high] = [{1},{2}]:".format(dip_value, low, high))
            print("gcm[0:{0}] = {1}".format(l_gcm, gcm[:l_gcm + 1]))
            print("lcm[0:{0}] = {1}".format(l_lcm, lcm[:l_lcm + 1]))
        d = 0.0
        if l_gcm != 1 or l_lcm != 1:
            if debug:
                print("  while(gcm[ix] != lcm[iv])")
            while True:
                gcmix = gcm[ix]
                lcmiv = lcm[iv]
                if gcmix > lcmiv:
                    gcmil = gcm[ix + 1]
                    dx = (lcmiv - gcmil + 1) - (X[lcmiv] - X[gcmil]) * (gcmix - gcmil) / (X[gcmix] - X[gcmil])
                    iv += 1
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv - 1
                        if debug:
                            print("L({0},{1})".format(ig, ih))
                else:
                    lcmivl = lcm[iv - 1]
                    dx = (X[gcmix] - X[lcmivl]) * (lcmiv - lcmivl) / (X[lcmiv] - X[lcmivl]) - (gcmix - lcmivl - 1)
                    ix -= 1
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv
                        if debug:
                            print("G({0},{1})".format(ig, ih))
                if ix < 0:
                    ix = 0
                if iv > l_lcm:
                    iv = l_lcm
                if debug:
                    print("  --> ix = {0}, iv = {1}".format(ix, iv))
                if gcm[ix] == lcm[iv]:
                    break
        else:
            d = 0.0
            if debug:
                print("  ** (l_lcm,l_gcm) = ({0},{1}) ==> d := {2}".format(l_lcm, l_gcm, d))
        if d < dip_value:
            break
        if debug:
            print("  calculating dip ..")
        j_l = -1
        j_u = -1
        lcm_modalTriangle_i1 = -1
        lcm_modalTriangle_i3 = -1
        gcm_modalTriangle_i1 = -1
        gcm_modalTriangle_i3 = -1
        # The DIP for the convex minorant
        dip_l = 0
        for j in range(ig, l_gcm):
            max_t = 1
            j_ = -1
            jb = gcm[j + 1]
            je = gcm[j]
            if je - jb > 1 and X[je] != X[jb]:
                C = (je - jb) / (X[je] - X[jb])
                for jj in range(jb, je + 1):
                    t = (jj - jb + 1) - (X[jj] - X[jb]) * C
                    if max_t < t:
                        max_t = t
                        j_ = jj
            if dip_l < max_t:
                dip_l = max_t
                j_l = j_
                gcm_modalTriangle_i1 = jb
                gcm_modalTriangle_i3 = je
        # The DIP for the concave majorant
        dip_u = 0
        for j in range(ih, l_lcm):
            max_t = 1
            j_ = -1
            jb = lcm[j]
            je = lcm[j + 1]
            if je - jb > 1 and X[je] != X[jb]:
                C = (je - jb) / (X[je] - X[jb])
                for jj in range(jb, je + 1):
                    t = (X[jj] - X[jb]) * C - (jj - jb - 1)
                    if max_t < t:
                        max_t = t
                        j_ = jj
            if dip_u < max_t:
                dip_u = max_t
                j_u = j_
                lcm_modalTriangle_i1 = jb
                lcm_modalTriangle_i3 = je
        if debug:
            print(" (dip_l, dip_u) = ({0}, {1})".format(dip_l, dip_u))
        if dip_u > dip_l:
            dip_new = dip_u
            j_best = j_u
            if debug:
                print(" -> new larger dip {0} (j_best = {1}) gcm-centred triple ({2},{3},{4})".format(dip_new, j_best,
                                                                                                      lcm_modalTriangle_i1,
                                                                                                      j_best,
                                                                                                      lcm_modalTriangle_i3))
        else:
            dip_new = dip_l
            j_best = j_l
            if debug:
                print(" -> new larger dip {0} (j_best = {1}) lcm-centred triple ({2},{3},{4})".format(dip_new, j_best,
                                                                                                      gcm_modalTriangle_i1,
                                                                                                      j_best,
                                                                                                      gcm_modalTriangle_i3))
        if dip_value < dip_new:
            dip_value = dip_new
            if dip_u > dip_l:
                modaltriangle_i1 = lcm_modalTriangle_i1
                modaltriangle_i2 = j_best
                modaltriangle_i3 = lcm_modalTriangle_i3
            else:
                modaltriangle_i1 = gcm_modalTriangle_i1
                modaltriangle_i2 = j_best
                modaltriangle_i3 = gcm_modalTriangle_i3
        if low == gcm[ig] and high == lcm[ih]:
            if debug:
                print("No improvement in  low = {0}  nor  high = {1} --> END".format(low, high))
            break
        low = gcm[ig]
        high = lcm[ih]
    dip_value /= (2 * N)
    return dip_value, (low, high), (modaltriangle_i1, modaltriangle_i2, modaltriangle_i3), gcm, lcm, mn, mj


def dip_pval(dip_value: float, n_points: int, pval_strategy: str = "table", n_boots: int = 1000,
             random_state: np.random.RandomState | int = None) -> float:
    """
    Get the p-value of a corresponding Dip-value.
    P-values depend on the input Dip-value and the sample size.
    There are several strategies to calculate the p-value. These are:
    'table' (most common), 'function' (available for all sample sizes) and 'bootstrap' (slow for large sample sizes)

    Parameters
    ----------
    dip_value : flaat
        The Dip-value
    n_points : int
        The number of samples
    pval_strategy : str
        Specifies the strategy that should be used to calculate the p-value (default: 'table')
    n_boots : int
        Number of random data sets that should be created to calculate Dip-values. Only relevant if pval_strategy is 'bootstrap' (default: 1000)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int. Only relevant if pval_strategy is 'bootstrap' (default: None)

    Returns
    -------
    pval : float
        The resulting p-value

    References
    ----------
    Hartigan, John A., and Pamela M. Hartigan.
    "The dip test of unimodality." The annals of Statistics (1985): 70-84.

    and

    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """
    assert type(pval_strategy) is str, "pval_stratgegy must be of type string"
    pval_strategy = pval_strategy.lower()
    assert pval_strategy in ["bootstrap", "table",
                             "function"], "pval_strategy must match 'bootstrap', 'table' or 'function'. " \
                                          "Your input: {0}".format(pval_strategy)
    if n_points < 4:
        pval = 1.0
    elif pval_strategy == "bootstrap":
        boot_dips = dip_boot_samples(int(n_points), n_boots, random_state)
        pval = np.mean(dip_value <= boot_dips)
    elif pval_strategy == "table":
        pval = _dip_pval_table(dip_value, n_points)
    elif pval_strategy == "function":
        pval = _dip_pval_function(dip_value, n_points)
    else:
        raise Exception(
            "pval_strategy must match 'bootstrap', 'table' or 'function. Your input: {0}".format(pval_strategy))
    return pval


def dip_boot_samples(n_points: int, n_boots: int = 1000,
                     random_state: np.random.RandomState | int = None) -> np.ndarray:
    """
    Sample random data sets and calculate corresponding Dip-values.
    E.g. used to determine p-values.

    Parameters
    ----------
    n_points : int
        The number of samples
    n_boots : int
        Number of random data sets that should be created to calculate Dip-values (default: 1000)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Returns
    -------
    boot_dips : np.ndarray
        Array of Dip-values
    """
    # random uniform vectors
    random_state = check_random_state(random_state)
    boot_samples = random_state.rand(n_boots, n_points)
    boot_dips = np.array([dip_test(boot_s, just_dip=True, is_data_sorted=False) for boot_s in boot_samples])
    return boot_dips


def _get_complete_gcm_lcm(mn: np.ndarray, mj: np.ndarray, modal_interval: tuple) -> (np.ndarray, np.ndarray):
    """
    Complete the GCM and LCM returned by the Dip-test.
    Adapted from: https://github.com/samhelmholtz/skinny-dip/blob/master/code/skinny-dip/RPackageDipTestCustom/diptest/R/dip.R

    Parameters
    ----------
    mn : np.ndarray
        The minorant values
    mj : np.ndarray
        The majorant values
    modal_interval : tuple
        Indices of the modal interval - corresponds to the steepest slope in the ECDF (default: None)

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The adjusted indices of points that are part of the Greatest Convex Minorant,
        The adjusted indices of points that are part of the Least Concave Majorant
    """
    low = modal_interval[0]
    high = modal_interval[1]
    gcm = np.zeros(mn.shape[0], dtype=np.int32)
    lcm = np.zeros(mj.shape[0], dtype=np.int32)
    # Collect the gcm points from 0 to the upper end of the modal interval
    i = 0
    gcm[0] = high
    while gcm[i] > 0:
        gcm[i + 1] = mn[gcm[i]]
        i += 1
    gcm = gcm[:i + 1]
    # Collect the lcm points from the lower end of the modal interval to the last point
    i = 0
    lcm[i] = low
    while lcm[i] < mj.shape[0] - 1:
        lcm[i + 1] = mj[lcm[i]]
        i += 1
    lcm = lcm[:i + 1]
    return gcm, lcm


"""
Dip Gradient
"""


def dip_gradient(X: np.ndarray, X_proj: np.ndarray, sorted_indices: np.ndarray, modal_triangle: tuple) -> np.ndarray:
    """
    Calculate the gradient of the Dip-value regarding the projection axis.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    X_proj : np.ndarray
        The univariate projected data set
    sorted_indices : np.ndarray
        The indices of the sorted univariate data set
    modal_triangle : tuple
        Indices of the modal triangle

    Returns
    -------
    gradient : np.ndarray
        The gradient of the Dip-value regarding the projection axis

    References
    ----------
    Krause, Andreas, and Volkmar Liebscher.
    "Multimodal projection pursuit using the dip statistic." (2005).
    """
    if modal_triangle is None or modal_triangle[0] == -1:
        return np.zeros(X.shape[1])
    data_index_i1 = sorted_indices[modal_triangle[0]]
    data_index_i2 = sorted_indices[modal_triangle[1]]
    data_index_i3 = sorted_indices[modal_triangle[2]]
    # Get A and c
    A = modal_triangle[0] - modal_triangle[1] + \
        (modal_triangle[2] - modal_triangle[0]) * (X_proj[data_index_i2] - X_proj[data_index_i1]) / (
                X_proj[data_index_i3] - X_proj[data_index_i1])
    constant = (modal_triangle[2] - modal_triangle[0]) / (2 * X.shape[0])
    # Check A
    if A < 0:
        constant = -constant
    # Calculate gradient
    quotient = (X_proj[data_index_i3] - X_proj[data_index_i1])
    gradient = (X[data_index_i2] - X[data_index_i1]) / quotient - \
               (X[data_index_i3] - X[data_index_i1]) * (
                       X_proj[data_index_i2] - X_proj[data_index_i1]) / quotient ** 2
    gradient = gradient * constant
    return gradient


def dip_pval_gradient(X: np.ndarray, X_proj: np.ndarray, sorted_indices: np.ndarray, modal_triangle: tuple,
                      dip_value: float) -> np.ndarray:
    """
    Calculate the gradient of the Dip p-value function regarding the projection axis.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    X_proj : np.ndarray
        The univariate projected data set
    sorted_indices : np.ndarray
        The indices of the sorted univariate data set
    modal_triangle : tuple
        Indices of the modal triangle
    dip_value : float
        The Dip-value

    Returns
    -------
    pval_grad : np.ndarray
        The gradient of the Dip p-value function regarding the projection axis

    References
    ----------
    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """
    if modal_triangle is None or modal_triangle[0] == -1:
        return np.zeros(X.shape[1])
    # Calculate gradient of dip-value
    dip_grad = dip_gradient(X, X_proj, sorted_indices, modal_triangle)
    # Get factor for that gradient from p-value gradeint
    b = _dip_pval_function_get_b(X.shape[0])
    exponent = np.exp(-b * dip_value + 6.5)
    quotient = (0.6 * (1 + 1.6 * exponent) ** (0.625) + 0.4 * (1 + 0.2 * exponent) ** 5) ** 2
    grad_factor = (0.6 * (1 + 1.6 * exponent) ** (-0.375) + 0.4 * (1 + 0.2 * exponent) ** 4) / quotient
    grad_factor *= exponent * -b
    # Combine values
    pval_grad = grad_factor * dip_grad
    return pval_grad


"""
Plot
"""


def plot_dip(X: np.ndarray, is_data_sorted: bool = False, dip_value: float = None, modal_interval: tuple = None,
             modal_triangle: tuple = None, gcm: np.ndarray = None, lcm: np.ndarray = None, linewidth_ecdf: float = 1,
             linewidth_extra: float = 2, show_legend: bool = True, add_histogram: bool = True,
             histogram_labels: np.ndarray = None, histogram_show_legend: bool = True, histogram_density: bool = True,
             histogram_n_bins: int = 100, height_ratio: tuple = (1, 2), show_plot: bool = True) -> None:
    """
    Plot a visual representation of the computational process of the Dip.
    Upper part shows an optional histogram of the data and the lower part shows the corresponding ECDF.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    is_data_sorted : bool
        Should be True if the data set is already sorted (default: False)
    dip_value : float
        The Dip-value (default: None)
    modal_interval : tuple
        Indices of the modal interval - corresponds to the steepest slope in the ECDF (default: None)
    modal_triangle : tuple
        Indices of the modal triangle (default: None)
    gcm : np.ndarray
        The indices of points that are part of the Greatest Convex Minorant (gcm) (default: None)
    lcm : np.ndarray
        The indices of points that are part of the Least Concave Majorant (lcm) (default None)
    linewidth_ecdf : flaot
        The linewidth for the eCDF (default: 1)
    linewidth_extra : float
        The linewidth for the visualization of the dip, modal interval, modal triangle, gcm and lcm (default: 2)
    show_legend : bool
        Defines whether the legend of the ECDF plot should be added (default: True)
    add_histogram : bool
        Defines whether the histogram should be shown above the ECDF plot (default: True)
    histogram_labels : np.ndarray
        Labels used to color parts of the histogram (default: None)
    histogram_show_legend : bool
        Defines whether the legend of the histogram should be added (default: True)
    histogram_density : bool
        Defines whether a kernel density should be added to the histogram plot (default: True)
    histogram_n_bins : int
        Number of bins used for the histogram (default: 100)
    height_ratio : tuple
        Defines the height ratio between histogram and ECDF plot. Only relevant if add_histogram is True.
        First value in the tuple defines the height of the histogram and the second value the height of the ECDF plot (default: (1, 2))
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)
    """
    assert X.ndim == 1, "Data must be 1-dimensional for the dip-test. Your shape:{0}".format(X.shape)
    N = len(X)
    if not is_data_sorted:
        argsorted_X = np.argsort(X)
        X = X[argsorted_X]
        if histogram_labels is not None:
            histogram_labels = histogram_labels[argsorted_X]
    if add_histogram:
        # Add histogram at the top of the plot (uses plot_histogram from clustpy.utils.plots)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': height_ratio})
        plot_histogram(X, histogram_labels, density=histogram_density, n_bins=histogram_n_bins,
                       show_legend=histogram_show_legend,
                       container=ax1, show_plot=False)
        if modal_interval is not None:
            y_axis_limit = ax1.get_ylim()
            ax1.plot([X[modal_interval[0]], X[modal_interval[0]]], y_axis_limit, "g--", linewidth=linewidth_extra)
            ax1.plot([X[modal_interval[1]], X[modal_interval[1]]], y_axis_limit, "g--", linewidth=linewidth_extra)
        dip_container = ax2
        # Remove spacing between the two plots
        fig.subplots_adjust(hspace=0)
    else:
        dip_container = plt
    # Plot ECDF
    dip_container.plot(X, np.arange(N) / N, "b", label="eCDF", linewidth=linewidth_ecdf)
    if dip_value is not None:
        # Add Dip range around ECDF
        dip_container.plot(X, np.arange(N) / N - dip_value * 2, "k:", alpha=0.7, label="2x dip",
                           linewidth=linewidth_extra)
        dip_container.plot(X, np.arange(N) / N + dip_value * 2, "k:", alpha=0.7, linewidth=linewidth_extra)
    if modal_interval is not None:
        # Add modal interval in green
        dip_container.plot([X[modal_interval[0]], X[modal_interval[1]]], [modal_interval[0] / N, modal_interval[1] / N],
                           linewidth=linewidth_extra, c="g", label="modal interval")
    if modal_triangle is not None:
        # Add modal triangle in red
        dip_container.plot([X[modal_triangle[0]], X[modal_triangle[1]], X[modal_triangle[2]], X[modal_triangle[0]]],
                           [modal_triangle[0] / N, modal_triangle[1] / N, modal_triangle[2] / N, modal_triangle[0] / N],
                           linewidth=linewidth_extra, c="r", label="modal triangle")
    # Add process of gcm and lcm curves
    if gcm is not None:
        dip_container.plot(X[gcm], gcm / N, "y--", linewidth=linewidth_extra, label="gcm")
    if lcm is not None:
        dip_container.plot(X[lcm], lcm / N, "c--", linewidth=linewidth_extra, label="lcm")
    if show_legend:
        dip_container.legend(loc="lower right")
    if show_plot:
        plt.show()


"""
Dip p-value methods
"""


def _dip_pval_table(dip_value: float, n_points: int) -> float:
    """
    Get the p-value of a corresponding Dip-value using a lookup table.
    Depends on the number of samples.
    Source: https://github.com/alimuldal/diptest

    Parameters
    ----------
    dip_value : flaat
        The Dip-value
    n_points : int
        The number of samples

    Returns
    -------
    pval : float
        The resulting p-value

    References
    ----------
    Hartigan, John A., and Pamela M. Hartigan.
    "The dip test of unimodality." The annals of Statistics (1985): 70-84.
    """
    N, SIG, CV = _get_dip_table_values()
    if n_points > max(N):
        print(
            "[dip_pval_table] WARNING: The number of samples is too large for pval_strategy 'table' (max n_points is 72000), 'function' will be used instead.")
        return _dip_pval_function(dip_value, n_points)
    i1 = N.searchsorted(n_points, side='left')
    i0 = i1 - 1
    # if n falls outside the range of tabulated sample sizes, use the
    # critical values for the nearest tabulated n (i.e. treat them as
    # 'asymptotic')
    i0 = max(0, i0)
    i1 = min(N.shape[0] - 1, i1)
    # interpolate on sqrt(n)
    n0, n1 = N[[i0, i1]]
    if i0 != i1:
        fn = float(n_points - n0) / (n1 - n0)
    else:
        fn = 0
    y0 = np.sqrt(n0) * CV[i0]
    y1 = np.sqrt(n1) * CV[i1]
    sD = np.sqrt(n_points) * dip_value
    pval = 1. - np.interp(sD, y0 + fn * (y1 - y0), SIG)
    return pval


def _dip_pval_function(dip_value: float, n_points: int) -> float:
    """
    Get the p-value of a corresponding Dip-value using the sigmoid function as described by Bauer et al..
    Depends on the number of samples.

    Parameters
    ----------
    dip_value : flaat
        The Dip-value
    n_points : int
        The number of samples

    Returns
    -------
    pval : float
        The resulting p-value

    References
    ----------
    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """
    b = _dip_pval_function_get_b(n_points)
    exponent = np.exp(-b * dip_value + 6.5)
    pval = 1 - 1 / (0.6 * (1 + 1.6 * exponent) ** (0.625) + 0.4 * (1 + 0.2 * exponent) ** 5)
    return pval


def _dip_pval_function_get_b(n_points: int) -> float:
    """
    Helper function for the Dip-p-value calculation using the sigmoid function by Bauer et al..

    Parameters
    ----------
    n_points : int
        The number of samples

    Returns
    -------
    b : float
        The b value

    References
    ----------
    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """
    b = 17.30784022 * np.sqrt(n_points) + 12.04917889
    return b


def _get_dip_table_values() -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Get the values of the Dip-p-value lookup table.

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray)
        The sample sizes (21 values),
        The probabilities (26 values),
        The Dip-values (21 x 26 values)
    """
    N = np.array([4, 5, 6, 7, 8, 9, 10, 15, 20,
                  30, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
                  20000, 40000, 72000])
    SIG = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.2,
                    0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                    0.9, 0.95, 0.98, 0.99, 0.995, 0.998,
                    0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99998,
                    0.99999, 1.])
    # [len(N), len(SIG)] table of critical values
    CV = np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.132559548782689,
                    0.157497369040235, 0.187401878807559, 0.20726978858736, 0.223755804629222, 0.231796258864192,
                    0.237263743826779, 0.241992892688593, 0.244369839049632, 0.245966625504691, 0.247439597233262,
                    0.248230659656638, 0.248754269146416, 0.249302039974259, 0.249459652323225, 0.24974836247845],
                   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.108720593576329, 0.121563798026414, 0.134318918697053,
                    0.147298798976252, 0.161085025702604, 0.176811998476076, 0.186391796027944, 0.19361253363045,
                    0.196509139798845, 0.198159967287576, 0.199244279362433, 0.199617527406166, 0.199800941499028,
                    0.199917081834271, 0.199959029093075, 0.199978395376082, 0.199993151405815, 0.199995525025673,
                    0.199999835639211],
                   [0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333,
                    0.0833333333333333, 0.0924514470941933, 0.103913431059949, 0.113885220640212, 0.123071187137781,
                    0.13186973390253, 0.140564796497941, 0.14941924112913, 0.159137064572627, 0.164769608513302,
                    0.179176547392782, 0.191862827995563, 0.202101971042968, 0.213015781111186, 0.219518627282415,
                    0.224339047394446, 0.229449332154241, 0.232714530449602, 0.236548128358969, 0.2390887911995,
                    0.240103566436295, 0.244672883617768],
                   [0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.0725717816250742,
                    0.0817315478539489, 0.09405901819225269, 0.103244490800871, 0.110964599995697,
                    0.117807846504335, 0.124216086833531, 0.130409013968317, 0.136639642123068, 0.144240669035124,
                    0.159903395678336, 0.175196553271223, 0.184118659121501, 0.191014396174306, 0.198216795232182,
                    0.202341010748261, 0.205377566346832, 0.208306562526874, 0.209866047852379, 0.210967576933451,
                    0.212233348558702, 0.212661038312506, 0.21353618608817],
                   [0.0625, 0.0625, 0.06569119945032829, 0.07386511360717619, 0.0820045917762512,
                    0.0922700601131892, 0.09967371895993631, 0.105733531802737, 0.111035129847705,
                    0.115920055749988, 0.120561479262465, 0.125558759034845, 0.141841067033899, 0.153978303998561,
                    0.16597856724751, 0.172988528276759, 0.179010413496374, 0.186504388711178, 0.19448404115794,
                    0.200864297005026, 0.208849997050229, 0.212556040406219, 0.217149174137299, 0.221700076404503,
                    0.225000835357532, 0.233772919687683],
                   [0.0555555555555556, 0.0613018090298924, 0.0658615858179315, 0.0732651142535317,
                    0.0803941629593475, 0.0890432420913848, 0.0950811420297928, 0.09993808978110461,
                    0.104153560075868, 0.108007802361932, 0.112512617124951, 0.122915033480817, 0.136412639387084,
                    0.146603784954019, 0.157084065653166, 0.164164643657217, 0.172821674582338, 0.182555283567818,
                    0.188658833121906, 0.194089120768246, 0.19915700809389, 0.202881598436558, 0.205979795735129,
                    0.21054115498898, 0.21180033095039, 0.215379914317625],
                   [0.05, 0.0610132555623269, 0.0651627333214016, 0.0718321619656165, 0.077966212182459,
                    0.08528353598345639, 0.09032041737070989, 0.0943334983745117, 0.0977817630384725,
                    0.102180866696628, 0.109960948142951, 0.118844767211587, 0.130462149644819, 0.139611395137099,
                    0.150961728882481, 0.159684158858235, 0.16719524735674, 0.175419540856082, 0.180611195797351,
                    0.185286416050396, 0.191203083905044, 0.195805159339184, 0.20029398089673, 0.205651089646219,
                    0.209682048785853, 0.221530282182963],
                   [0.0341378172277919, 0.0546284208048975, 0.0572191260231815, 0.0610087367689692,
                    0.06426571373304441, 0.06922341079895911, 0.0745462114365167, 0.07920308789817621,
                    0.083621033469191, 0.08811984822029049, 0.093124666680253, 0.0996694393390689,
                    0.110087496900906, 0.118760769203664, 0.128890475210055, 0.13598356863636, 0.142452483681277,
                    0.150172816530742, 0.155456133696328, 0.160896499106958, 0.166979407946248, 0.17111793515551,
                    0.175900505704432, 0.181856676013166, 0.185743454151004, 0.192240563330562],
                   [0.033718563622065, 0.0474333740698401, 0.0490891387627092, 0.052719998201553,
                    0.0567795509056742, 0.0620134674468181, 0.06601638720690479, 0.06965060750664009,
                    0.07334377405927139, 0.07764606628802539, 0.0824558407118372, 0.08834462700173699,
                    0.09723460181229029, 0.105130218270636, 0.114309704281253, 0.120624043335821, 0.126552378036739,
                    0.13360135382395, 0.138569903791767, 0.14336916123968, 0.148940116394883, 0.152832538183622,
                    0.156010163618971, 0.161319225839345, 0.165568255916749, 0.175834459522789],
                   [0.0262674485075642, 0.0395871890405749, 0.0414574606741673, 0.0444462614069956,
                    0.0473998525042686, 0.0516677370374349, 0.0551037519001622, 0.058265005347493,
                    0.0614510857304343, 0.0649164408053978, 0.0689178762425442, 0.0739249074078291,
                    0.08147913793901269, 0.0881689143126666, 0.0960564383013644, 0.101478558893837,
                    0.10650487144103, 0.112724636524262, 0.117164140184417, 0.121425859908987, 0.126733051889401,
                    0.131198578897542, 0.133691739483444, 0.137831637950694, 0.141557509624351, 0.163833046059817],
                   [0.0218544781364545, 0.0314400501999916, 0.0329008160470834, 0.0353023819040016,
                    0.0377279973102482, 0.0410699984399582, 0.0437704598622665, 0.0462925642671299,
                    0.048851155289608, 0.0516145897865757, 0.0548121932066019, 0.0588230482851366,
                    0.06491363240467669, 0.0702737877191269, 0.07670958860791791, 0.0811998415355918,
                    0.0852854646662134, 0.09048478274902939, 0.0940930106666244, 0.0974904344916743,
                    0.102284204283997, 0.104680624334611, 0.107496694235039, 0.11140887547015, 0.113536607717411,
                    0.117886716865312],
                   [0.0164852597438403, 0.022831985803043, 0.0238917486442849, 0.0256559537977579,
                    0.0273987414570948, 0.0298109370830153, 0.0317771496530253, 0.0336073821590387,
                    0.0354621760592113, 0.0374805844550272, 0.0398046179116599, 0.0427283846799166,
                    0.047152783315718, 0.0511279442868827, 0.0558022052195208, 0.059024132304226,
                    0.0620425065165146, 0.06580160114660991, 0.0684479731118028, 0.0709169443994193,
                    0.0741183486081263, 0.0762579402903838, 0.0785735967934979, 0.08134583568891331,
                    0.0832963013755522, 0.09267804230967371],
                   [0.0111236388849688, 0.0165017735429825, 0.0172594157992489, 0.0185259426032926,
                    0.0197917612637521, 0.0215233745778454, 0.0229259769870428, 0.024243848341112,
                    0.025584358256487, 0.0270252129816288, 0.0286920262150517, 0.0308006766341406,
                    0.0339967814293504, 0.0368418413878307, 0.0402729850316397, 0.0426864799777448,
                    0.044958959158761, 0.0477643873749449, 0.0497198001867437, 0.0516114611801451,
                    0.0540543978864652, 0.0558704526182638, 0.0573877056330228, 0.0593365901653878,
                    0.0607646310473911, 0.0705309107882395],
                   [0.00755488597576196, 0.0106403461127515, 0.0111255573208294, 0.0119353655328931,
                    0.0127411306411808, 0.0138524542751814, 0.0147536004288476, 0.0155963185751048,
                    0.0164519238025286, 0.017383057902553, 0.0184503949887735, 0.0198162679782071,
                    0.0218781313182203, 0.0237294742633411, 0.025919578977657, 0.0274518022761997,
                    0.0288986369564301, 0.0306813505050163, 0.0320170996823189, 0.0332452747332959,
                    0.0348335698576168, 0.0359832389317461, 0.0369051995840645, 0.0387221159256424,
                    0.03993025905765, 0.0431448163617178],
                   [0.00541658127872122, 0.00760286745300187, 0.007949878346447991, 0.008521651834359399,
                    0.00909775605533253, 0.009889245210140779, 0.0105309297090482, 0.0111322726797384,
                    0.0117439009052552, 0.012405033293814, 0.0131684179320803, 0.0141377942603047,
                    0.0156148055023058, 0.0169343970067564, 0.018513067368104, 0.0196080260483234,
                    0.0206489568587364, 0.0219285176765082, 0.0228689168972669, 0.023738710122235,
                    0.0248334158891432, 0.0256126573433596, 0.0265491336936829, 0.027578430100536, 0.0284430733108,
                    0.0313640941982108],
                   [0.00390439997450557, 0.00541664181796583, 0.00566171386252323, 0.00607120971135229,
                    0.0064762535755248, 0.00703573098590029, 0.00749421254589299, 0.007920878896017331,
                    0.008355737247680061, 0.00882439333812351, 0.00936785820717061, 0.01005604603884,
                    0.0111019116837591, 0.0120380990328341, 0.0131721010552576, 0.0139655122281969,
                    0.0146889122204488, 0.0156076779647454, 0.0162685615996248, 0.0168874937789415,
                    0.0176505093388153, 0.0181944265400504, 0.0186226037818523, 0.0193001796565433,
                    0.0196241518040617, 0.0213081254074584],
                   [0.00245657785440433, 0.00344809282233326, 0.00360473943713036, 0.00386326548010849,
                    0.00412089506752692, 0.00447640050137479, 0.00476555693102276, 0.00503704029750072,
                    0.00531239247408213, 0.00560929919359959, 0.00595352728377949, 0.00639092280563517,
                    0.00705566126234625, 0.0076506368153935, 0.00836821687047215, 0.008863578928549141,
                    0.00934162787186159, 0.009932186363240289, 0.0103498795291629, 0.0107780907076862,
                    0.0113184316868283, 0.0117329446468571, 0.0119995948968375, 0.0124410052027886,
                    0.0129467396733128, 0.014396063834027],
                   [0.00174954269199566, 0.00244595133885302, 0.00255710802275612, 0.00273990955227265,
                    0.0029225480567908, 0.00317374638422465, 0.00338072258533527, 0.00357243876535982,
                    0.00376734715752209, 0.00397885007249132, 0.00422430013176233, 0.00453437508148542,
                    0.00500178808402368, 0.00542372242836395, 0.00592656681022859, 0.00628034732880374,
                    0.00661030641550873, 0.00702254699967648, 0.00731822628156458, 0.0076065423418208,
                    0.00795640367207482, 0.008227052458435399, 0.00852240989786251, 0.00892863905540303,
                    0.009138539330002131, 0.009522345795667729],
                   [0.00119458814106091, 0.00173435346896287, 0.00181194434584681, 0.00194259470485893,
                    0.00207173719623868, 0.00224993202086955, 0.00239520831473419, 0.00253036792824665,
                    0.00266863168718114, 0.0028181999035216, 0.00299137548142077, 0.00321024899920135,
                    0.00354362220314155, 0.00384330190244679, 0.00420258799378253, 0.00445774902155711,
                    0.00469461513212743, 0.00499416069129168, 0.00520917757743218, 0.00540396235924372,
                    0.00564540201704594, 0.00580460792299214, 0.00599774739593151, 0.00633099254378114,
                    0.00656987109386762, 0.00685829448046227],
                   [0.000852415648011777, 0.00122883479310665, 0.00128469304457018, 0.00137617650525553,
                    0.00146751502006323, 0.00159376453672466, 0.00169668445506151, 0.00179253418337906,
                    0.00189061261635977, 0.00199645471886179, 0.00211929748381704, 0.00227457698703581,
                    0.00250999080890397, 0.00272375073486223, 0.00298072958568387, 0.00315942194040388,
                    0.0033273652798148, 0.00353988965698579, 0.00369400045486625, 0.00383345715372182,
                    0.00400793469634696, 0.00414892737222885, 0.0042839159079761, 0.00441870104432879,
                    0.00450818604569179, 0.00513477467565583],
                   [0.000644400053256997, 0.000916872204484283, 0.000957932946765532, 0.00102641863872347,
                    0.00109495154218002, 0.00118904090369415, 0.00126575197699874, 0.00133750966361506,
                    0.00141049709228472, 0.00148936709298802, 0.00158027541945626, 0.00169651643860074,
                    0.00187306184725826, 0.00203178401610555, 0.00222356097506054, 0.00235782814777627,
                    0.00248343580127067, 0.00264210826339498, 0.0027524322157581, 0.0028608570740143,
                    0.00298695044508003, 0.00309340092038059, 0.00319932767198801, 0.00332688234611187,
                    0.00339316094477355, 0.00376331697005859]])
    return N, SIG, CV

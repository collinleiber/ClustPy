"""
@authors:
Collin Leiber
"""

from clustpy.utils import dip_test, dip_pval
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


def _skinnydip(X: np.ndarray, significance: float, pval_strategy: str, n_boots: int, add_tails: bool, outliers: bool,
               max_cluster_size_diff_factor: float, random_state: np.random.RandomState, debug: bool) -> (
        int, np.ndarray):
    """
    Start the actual SkinnyDip clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap'
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap'
    add_tails : bool
        Defines if TailoredDip should try to add tails to the surrounding clusters
    outliers : bool
        Defines if outliers should be identified as described by UniDip
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used for merging and assigning tails of distributions if 'add_tails' is True
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only relevant if pval_strategy is 'bootstrap'
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (int, np.ndarray)
        The final number of clusters,
        The labels as identified by SkinnyDip

    References
    ----------
    Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a sea of noise."
    Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
    """
    # Check if we have a multidimensional dataset
    if X.ndim == 1:
        if debug:
            print("[SkinnyDip] The dimensionality of the input data is 1. Execute UniDip")
        n_clusters, labels, _ = _tailoreddip(X, significance, pval_strategy, n_boots, add_tails, outliers,
                                             max_cluster_size_diff_factor, random_state, debug)
        return n_clusters, labels
    n_clusters = 1
    labels = np.zeros(X.shape[0], dtype=np.int32)
    # Iterate over all features
    for dim in range(X.shape[1]):
        n_clusters_old = n_clusters
        # Iterate over all clusters from last iteration
        for i in range(n_clusters_old):
            if debug:
                print("[SkinnyDip] Execute UniDip for dimension {0} and cluster {1}".format(dim, i))
            # Get points in this cluster
            points_in_cluster = (labels == i)
            # Call UniDip
            n_clusters_new, labels_new, _ = _tailoreddip(X[points_in_cluster, dim], significance, pval_strategy,
                                                         n_boots, add_tails, outliers, max_cluster_size_diff_factor,
                                                         random_state, debug)
            # Update labels
            labels_new[labels_new > 0] = labels_new[labels_new > 0] + n_clusters - 1
            labels_new[labels_new == 0] = i
            labels[points_in_cluster] = labels_new
            n_clusters += n_clusters_new - 1
    return n_clusters, labels


def _unidip_original(X_1d: np.ndarray, significance: float, already_sorted: bool, pval_strategy: str, n_boots: int,
                     max_cluster_size_diff_factor: float, random_state: np.random.RandomState, debug: bool) -> (
        int, np.ndarray, np.ndarray, np.ndarray, list):
    """
    Start the actual UniDip clustering procedure on the univariate input data set.

    Parameters
    ----------
    X_1d : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    already_sorted: bool
        Is the input data set already sorted?
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap'
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap'
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used for merging
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only relevant if pval_strategy is 'bootstrap'
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray, np.ndarray, list)
        The final number of clusters,
        The labels as identified by UniDip,
        The sorted input data set,
        The indices of the sorted data set,
        List of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore

    References
    ----------
    Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a sea of noise."
    Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
    """
    assert significance >= 0 and significance <= 1, "[UniDip] significance must be a value in the range [0, 1]"
    assert X_1d.ndim == 1, "[UniDip] Data must be 1-dimensional. Your input has shape: {0}".format(X_1d.shape)
    cluster_boundaries = []
    # Check if data is already sorted
    if already_sorted:
        argsorted = np.arange(X_1d.shape[0])
        X_1d_sorted = X_1d
    else:
        argsorted = np.argsort(X_1d)
        X_1d_sorted = X_1d[argsorted]
    # tmp_borders contains: (start and end value to search. Should be mirrored?. Current search space (equals position of next left and right cluster))
    tmp_borders = [(0, X_1d.shape[0], True, 0, X_1d.shape[0])]
    while len(tmp_borders) > 0:
        start, end, should_mirror, search_space_start, search_space_end = tmp_borders.pop(0)
        if debug:
            print("[UniDip] Checking interval {0} / Current clusters: {1}".format((start, end), cluster_boundaries))
        # Get part of data
        tmp_X_1d = X_1d_sorted[start:end]
        dip_value, modal_interval, _ = dip_test(tmp_X_1d, just_dip=False, is_data_sorted=True)
        dip_pvalue = dip_pval(dip_value, n_points=tmp_X_1d.shape[0], pval_strategy=pval_strategy,
                              n_boots=n_boots, random_state=random_state)
        low = modal_interval[0]
        high = modal_interval[1]
        if debug:
            print("[UniDip] Check resulted in the modal interval {0} with p-value {1}".format(
                (low + start, high + start + 1), dip_pvalue))
        if dip_pvalue < significance:  # Data is multimodal
            # Check area between low and high in next iteration. Area between low and search_space_start as well as high and search_space_end will be checked if a unimodal area is identified
            if low != high:
                tmp_borders.insert(0, (start + low, start + high + 1, False, search_space_start, search_space_end))
        else:  # Data is unimodal
            # If the current cluster wasn't a modal before, mirror data
            if should_mirror:
                _, low, high = _dip_mirrored_data(tmp_X_1d, (low, high))
                cluster_start = start + low
                cluster_end = start + high + 1
                if debug:
                    print(
                        "[UniDip] Data range was no modal before. Therefore, mirrored data resulted in modal interval",
                        (cluster_start, cluster_end))
            else:
                cluster_start = start
                cluster_end = end
            # Save cluster boundaries
            cluster_boundaries.append((cluster_start, cluster_end))
            if debug:
                print("[UniDip] => Add cluster", (cluster_start, cluster_end))
            # Other clusters to the right? (right must be handled before left)
            if cluster_end != search_space_end:
                right_X_1d = X_1d_sorted[cluster_start:search_space_end]
                dip_value, modal_interval, _ = dip_test(right_X_1d, just_dip=False, is_data_sorted=True)
                low, high = modal_interval
                dip_pvalue = dip_pval(dip_value, n_points=right_X_1d.shape[0], pval_strategy=pval_strategy,
                                      n_boots=n_boots, random_state=random_state)
                if debug:
                    print(
                        "[UniDip] -> Check of points to the right {0} resulted in modal interval {1} with p-value {2}".format(
                            (cluster_start, search_space_end),
                            (cluster_start + low, cluster_start + high), dip_pvalue))
                if dip_pvalue < significance:  # Data is multimodal
                    # Search area right of cluster
                    tmp_borders.insert(0, (cluster_end, search_space_end, True, cluster_end, search_space_end))
                else:  # Data is unimodal
                    # Update current cluster boundaries
                    _, low, high = _dip_mirrored_data(right_X_1d, (low, high))
                    cluster_boundaries[-1] = (min(cluster_start + low, cluster_boundaries[-1][0]),
                                              max(cluster_start + 1 + high, cluster_boundaries[-1][1]))
                    if debug:
                        print("[UniDip] -> Update cluster (due to mirroring) to", (cluster_boundaries[-1][0],
                                                                                   cluster_boundaries[-1][1]))
            # Other clusters to the left?
            if cluster_start != search_space_start:
                left_X_1d = X_1d_sorted[search_space_start:cluster_end]
                dip_value, modal_interval, _ = dip_test(left_X_1d, just_dip=False, is_data_sorted=True)
                low, high = modal_interval
                dip_pvalue = dip_pval(dip_value, n_points=left_X_1d.shape[0], pval_strategy=pval_strategy,
                                      n_boots=n_boots, random_state=random_state)
                if debug:
                    print(
                        "[UniDip] -> Check of points to the left {0} resulted in modal interval {1} with p-value {2}".format(
                            (search_space_start, cluster_end),
                            (search_space_start + low, search_space_start + high), dip_pvalue))
                if dip_pvalue < significance:  # Data is multimodal
                    # Search area left of cluster
                    tmp_borders.insert(0, (search_space_start, cluster_start, True, search_space_start, cluster_start))
                else:  # Data is unimodal
                    # Update current cluster boundaries
                    _, low, high = _dip_mirrored_data(left_X_1d, (low, high))
                    cluster_boundaries[-1] = (min(search_space_start + low, cluster_boundaries[-1][0]),
                                              max(search_space_start + 1 + high, cluster_boundaries[-1][1]))
                    if debug:
                        print("[UniDip] -> Update cluster (due to mirroring) to", (cluster_boundaries[-1][0],
                                                                                   cluster_boundaries[-1][1]))
    # Sort clusters by lower boundary
    cluster_boundaries = sorted(cluster_boundaries, key=lambda x: x[0])
    # Create labels array (in the beginning everything is noise)
    n_clusters = len(cluster_boundaries)
    labels = -np.ones(X_1d.shape[0], dtype=np.int32)
    for i, boundary in enumerate(cluster_boundaries):
        cluster_start, cluster_end = boundary
        labels[argsorted[cluster_start:cluster_end]] = i
    if debug:
        print("[UniDip] Clusters before merging:", cluster_boundaries)
    # Merge nearby clusters
    n_clusters, labels, cluster_boundaries = _merge_clusters(X_1d_sorted, argsorted, labels, n_clusters,
                                                             cluster_boundaries, significance, pval_strategy,
                                                             n_boots, max_cluster_size_diff_factor, random_state)
    if debug:
        print("[UniDip] Clusters after merging:", cluster_boundaries)
    return n_clusters, labels, X_1d_sorted, argsorted, cluster_boundaries


def _dip_mirrored_data(X_1d_sorted: np.ndarray, orig_modal_interval: tuple) -> (float, int, int):
    """
    Mirror the data to get a more accurate modal interval.
    For more information see 'The DipEncoder: Enforcing Multimodality in Autoencoders'.

    Parameters
    ----------
    X_1d_sorted : np.ndarray
        the input data set, must be sorted
    orig_modal_interval : tuple
        Tuple containing the starting and ending index of the original modal interval. Can be None

    Returns
    -------
    tuple : (float, int, int)
        The highest obtained Dip-value,
        The new starting index of the modal interval,
        The new ending index of the modal interval

    References
    ----------
    Leiber, Collin, et al. "The DipEncoder: Enforcing Multimodality in Autoencoders."
    Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.
    """
    # Left mirror
    mirrored_addition_left = X_1d_sorted[0] - np.flip(X_1d_sorted[1:] - X_1d_sorted[0])
    X_1d_left_mirrored = np.append(mirrored_addition_left, X_1d_sorted)
    dip_value_left, modal_interval_left, _ = dip_test(X_1d_left_mirrored, just_dip=False, is_data_sorted=True)
    # Right mirror
    mirrored_addition_right = X_1d_sorted[-1] + np.flip(X_1d_sorted[-1] - X_1d_sorted[:-1])
    X_1d_right_mirrored = np.append(X_1d_sorted, mirrored_addition_right)
    dip_value_right, modal_interval_right, _ = dip_test(X_1d_right_mirrored, just_dip=False, is_data_sorted=True)
    # Get interval of larger dip
    if dip_value_left > dip_value_right:
        low = modal_interval_left[0]
        high = modal_interval_left[1]
        if low < X_1d_sorted.shape[0] and high >= X_1d_sorted.shape[0]:
            if orig_modal_interval is None:
                # If no orig_modal_interval input is given, calculate modal_interval of original dataset
                _, orig_modal_interval, _ = dip_test(X_1d_sorted, just_dip=False, is_data_sorted=True)
            return dip_value_left, orig_modal_interval[0], orig_modal_interval[1]
        if low >= X_1d_sorted.shape[0]:
            return dip_value_left, low - (X_1d_sorted.shape[0] - 1), high - (X_1d_sorted.shape[0] - 1)
        else:
            return dip_value_left, (X_1d_sorted.shape[0] - 1) - high, (X_1d_sorted.shape[0] - 1) - low
    else:
        low = modal_interval_right[0]
        high = modal_interval_right[1]
        if low < X_1d_sorted.shape[0] and high >= X_1d_sorted.shape[0]:
            if orig_modal_interval is None:
                # If no orig_modal_interval input is given, calculate modal_interval of original dataset
                _, orig_modal_interval, _ = dip_test(X_1d_sorted, just_dip=False, is_data_sorted=True)
            return dip_value_right, orig_modal_interval[0], orig_modal_interval[1]
        if high < X_1d_sorted.shape[0]:
            return dip_value_right, low, high
        else:
            return dip_value_right, 2 * (X_1d_sorted.shape[0] - 1) - high, 2 * (X_1d_sorted.shape[0] - 1) - low


def _merge_clusters(X_1d_sorted: np.ndarray, argsorted: np.ndarray, labels: np.ndarray, n_clusters: int,
                    cluster_boundaries: list, significance: float, pval_strategy: str, n_boots: int,
                    max_cluster_size_diff_factor: float, random_state: np.random.RandomState) -> (
        int, np.ndarray, list):
    """
    Check for each cluster if it can be merged with the left or right neighboring cluster.
    The first and the last cluster will hereby handled by its more central neighbors.
    The decision of two clusters can be merged will be made using the dip-test by analyzing the transition of one cluster into the other.

    Parameters
    ----------
    X_1d_sorted : np.ndarray
        The input data set, must be sorted
    argsorted : np.ndarray
        The indices of the sorted data set
    labels : np.ndarray
        The current cluster labels
    n_clusters : int
        The current number of clusters
    cluster_boundaries : list
        List of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap'
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap'
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only relevant if pval_strategy is 'bootstrap'

    Returns
    -------
    tuple : (int, np.ndarray, list)
        The final number of clusters,
        The labels after merging,
        Updated list of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore
    """
    i = 1
    while i < len(cluster_boundaries) - 1:
        cluster_size_center = cluster_boundaries[i][1] - cluster_boundaries[i][0]
        # Dip of i combined with left (i - 1)
        cluster_size_left = cluster_boundaries[i - 1][1] - cluster_boundaries[i - 1][0]
        start_left = max(cluster_boundaries[i - 1][0],
                         int(cluster_boundaries[i - 1][1] - max_cluster_size_diff_factor * cluster_size_center))
        end_left = min(cluster_boundaries[i][1],
                       int(cluster_boundaries[i][0] + max_cluster_size_diff_factor * cluster_size_left))
        tmp_X_1d = X_1d_sorted[start_left:end_left]
        # Run dip-test
        dip_value = dip_test(tmp_X_1d, just_dip=True, is_data_sorted=True)
        dip_pvalue_left = dip_pval(dip_value, n_points=tmp_X_1d.shape[0], pval_strategy=pval_strategy,
                                   n_boots=n_boots, random_state=random_state)
        # Dip of i combined with right (i + 1)
        cluster_size_right = cluster_boundaries[i + 1][1] - cluster_boundaries[i + 1][0]
        start_right = max(cluster_boundaries[i][0],
                          int(cluster_boundaries[i][1] - max_cluster_size_diff_factor * cluster_size_right))
        end_right = min(cluster_boundaries[i + 1][1],
                        int(cluster_boundaries[i + 1][0] + max_cluster_size_diff_factor * cluster_size_center))
        tmp_X_1d = X_1d_sorted[start_right:end_right]
        # Run dip-test
        dip_value = dip_test(tmp_X_1d, just_dip=True, is_data_sorted=True)
        dip_pvalue_right = dip_pval(dip_value, n_points=tmp_X_1d.shape[0], pval_strategy=pval_strategy,
                                    n_boots=n_boots, random_state=random_state)
        if dip_pvalue_left >= dip_pvalue_right and dip_pvalue_left >= significance:
            # Merge i - 1 and i. Overwrite labels (beware, outliers can be in between)
            labels[argsorted[cluster_boundaries[i - 1][1]:cluster_boundaries[i][1]]] = i - 1
            labels[labels > i] -= 1
            cluster_boundaries[i - 1] = (cluster_boundaries[i - 1][0], cluster_boundaries[i][1])
            del cluster_boundaries[i]
            n_clusters -= 1
        elif dip_pvalue_right > dip_pvalue_left and dip_pvalue_right >= significance:
            # Merge i and i + 1. Overwrite labels (beware, outliers can be in between)
            labels[argsorted[cluster_boundaries[i][1]:cluster_boundaries[i + 1][1]]] = i
            labels[labels > i + 1] -= 1
            cluster_boundaries[i] = (cluster_boundaries[i][0], cluster_boundaries[i + 1][1])
            del cluster_boundaries[i + 1]
            n_clusters -= 1
        else:
            i += 1
    return n_clusters, labels, cluster_boundaries


"""
TailoredDip (UniDip improvements)
"""


def _tailoreddip(X_1d: np.ndarray, significance: float, pval_strategy: str, n_boots: int, add_tails: bool,
                 outliers: bool, max_cluster_size_diff_factor: float, random_state: np.random.RandomState,
                 debug: bool) -> (int, np.ndarray, list):
    """
    Start the actual TailoredDip clustering procedure on the univariate input data set.
    TailoredDip is an extension of UniDip that is able to better capture the tails of distributions.
    If add_tails is False and outliers True, the process of TailoredDip is equal to that of UniDip.

    Parameters
    ----------
    X_1d : np.ndarray
        the given data set
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap'
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap'
    add_tails : bool
        Defines if TailoredDip should try to add tails to the surrounding clusters
    outliers : bool
        Defines if outliers should be identified as described by UniDip
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used for merging and assigning tails of distributions if 'add_tails' is True
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only relevant if pval_strategy is 'bootstrap'
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (int, np.ndarray, list)
        The final number of clusters,
        The labels as identified by UniDip,
        List of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore

    References
    ----------
    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """
    # Start by executing the original UniDip algorithm
    n_clusters, labels, sorted_X_1d, argsorted, cluster_boundaries = _unidip_original(X_1d, significance, False,
                                                                                      pval_strategy, n_boots,
                                                                                      max_cluster_size_diff_factor,
                                                                                      random_state, debug)
    if add_tails:
        labels, cluster_boundaries = _add_tails(X_1d, labels, sorted_X_1d, argsorted, cluster_boundaries,
                                                significance, pval_strategy, n_boots,
                                                max_cluster_size_diff_factor, random_state, debug)
    if not outliers:
        labels = _assign_outliers(X_1d, labels, n_clusters, sorted_X_1d, argsorted, cluster_boundaries)
    return n_clusters, labels, cluster_boundaries


def _add_tails(X_1d: np.ndarray, labels: np.ndarray, sorted_X_1d: np.ndarray, argsorted: np.ndarray,
               cluster_boundaries_orig: list, significance: float, pval_strategy: str, n_boots: int,
               max_cluster_size_diff_factor: float, random_state: np.random.RandomState, debug: bool) -> (
        np.ndarray, list):
    """
    Add the tails of distributions to the surrounding clusters.
    This happens by mirroring the area between two clusters and checking whether this are is uni- or multimodal.
    If it is multimodal, samples are contained that cen be relevant for clustering.
    Therefore, we must test if this modes can be combined with a neighboring cluster.

    Parameters
    ----------
    X_1d : np.ndarray
        the given data set
    labels : np.ndarray
        The labels as identified by UniDip
    sorted_X_1d : np.ndarray
        The sorted input data set
    argsorted : np.ndarray
        The indices of the sorted data set
    cluster_boundaries_orig : list
        List of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap'
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap'
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used for merging and assigning tails of distributions if 'add_tails' is True
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Only relevant if pval_strategy is 'bootstrap'
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (np.ndarray, list)
        The updated labels after adding the tails,
        Updated list of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore
    """
    # Add cluster tails to the clusters. Check all areas between clusters and the area before the first and after the last cluster
    for i in range(len(cluster_boundaries_orig) + 1):
        while True:
            cluster_was_extended = False
            # Get points between two clusters
            if i != 0:
                # End of cluster before
                start = cluster_boundaries_orig[i - 1][1]
            else:
                start = 0
            if i != len(cluster_boundaries_orig):
                # Start of cluster
                end = cluster_boundaries_orig[i][0]
            else:
                end = X_1d.shape[0]
            if end - start < 4:  # Minimum number of samples for the dip-test is 4
                break
            X_tmp = sorted_X_1d[start:end]
            # Calculate mirrored dip to see if there is some relevant structure left between the clusters
            dip_value_mirror, _, _ = _dip_mirrored_data(X_tmp, (None, None))
            dip_pvalue_mirror = dip_pval(dip_value_mirror, n_points=(X_tmp.shape[0] * 2 - 1),
                                         pval_strategy=pval_strategy, n_boots=n_boots, random_state=random_state)
            if debug:
                print("[UniDip Add Tails] Check if interval {0} is unimodal. P-value is {1}".format((start, end),
                                                                                                    dip_pvalue_mirror))
            if dip_pvalue_mirror < significance:  # samples are multimodal
                # Execute UniDip on the noise data
                n_clusters_new, _, _, _, cluster_boundaries_new = _unidip_original(X_tmp, significance,
                                                                                   True, pval_strategy,
                                                                                   n_boots,
                                                                                   max_cluster_size_diff_factor,
                                                                                   random_state, debug)
                if debug:
                    print("[UniDip Add Tails] -> Identified the clusters {0} in the interval {1}".format(
                        cluster_boundaries_new, (start, end)))
                dip_pvalue_left = -1
                dip_pvalue_right = -1
                # Append first found structure to cluster before
                # Calculate dip of first found structure with cluster before
                if i != 0:
                    cluster_range = cluster_boundaries_new[0][1] - cluster_boundaries_new[0][0]
                    # Use a maximum of cluster_range points of left cluster to see if transition is unimodal
                    start_left = max(cluster_boundaries_orig[i - 1][0],
                                     int(cluster_boundaries_orig[i - 1][1] - max_cluster_size_diff_factor * cluster_range))
                    end_left = start + cluster_boundaries_new[0][1]
                    dip_value_left = dip_test(sorted_X_1d[start_left:end_left], just_dip=True, is_data_sorted=True)
                    dip_pvalue_left = dip_pval(dip_value_left, n_points=end_left - start_left,
                                               pval_strategy=pval_strategy, n_boots=n_boots, random_state=random_state)
                    if debug:
                        print(
                            "[UniDip Add Tails] -> Combination of left original cluster with first new cluster {0} has p-value of {1}".format(
                                (start_left, end_left), dip_pvalue_left))
                # Append last found structure to cluster after
                # Calculate dip of last found structure with cluster after
                if i != len(cluster_boundaries_orig):
                    cluster_range = cluster_boundaries_new[-1][1] - cluster_boundaries_new[-1][0]
                    start_right = start + cluster_boundaries_new[-1][0]
                    # Use a maximum of cluster_range points of right cluster to see if transition is unimodal
                    end_right = min(cluster_boundaries_orig[i][1],
                                    int(cluster_boundaries_orig[i][0] + max_cluster_size_diff_factor * cluster_range))
                    dip_value_right = dip_test(sorted_X_1d[start_right:end_right], just_dip=True,
                                               is_data_sorted=True)
                    dip_pvalue_right = dip_pval(dip_value_right, n_points=end_right - start_right,
                                                pval_strategy=pval_strategy, n_boots=n_boots, random_state=random_state)
                    if debug:
                        print(
                            "[UniDip Add Tails] -> Combination of right original cluster with last new cluster {0} has p-value of {1}".format(
                                (start_right, end_right), dip_pvalue_right))
                # --- Extend clusters
                # Does last found structure fit the cluster after? If so, extend cluster
                if dip_pvalue_right >= significance and (n_clusters_new > 1 or dip_pvalue_right >= dip_pvalue_left):
                    labels[argsorted[start_right:cluster_boundaries_orig[i][0]]] = i
                    cluster_boundaries_orig[i] = (start_right, cluster_boundaries_orig[i][1])
                    cluster_was_extended = True
                    if debug:
                        print("[UniDip Add Tails] -> Update right cluster to",
                              (start_right, cluster_boundaries_orig[i][1]))
                # Does first found structure fit the cluster before? If so, extend cluster
                if dip_pvalue_left >= significance and (n_clusters_new > 1 or dip_pvalue_left > dip_pvalue_right):
                    labels[argsorted[cluster_boundaries_orig[i - 1][1]:end_left]] = i - 1
                    cluster_boundaries_orig[i - 1] = (cluster_boundaries_orig[i - 1][0], end_left)
                    cluster_was_extended = True
                    if debug:
                        print("[UniDip Add Tails] -> Update left cluster to",
                              (cluster_boundaries_orig[i - 1][0], end_left))
            if not cluster_was_extended:
                break
    return labels, cluster_boundaries_orig


def _assign_outliers(X_1d: np.ndarray, labels: np.ndarray, n_clusters: int, sorted_X_1d: np.ndarray,
                     argsorted: np.ndarray, cluster_boundaries_orig: list) -> np.ndarray:
    """
    Add outliers to neighboring clusters.
    This method uses the intersection of the ECDF and the linear connection of two clusters as decision boundary.

    Parameters
    ----------
    X_1d : np.ndarray
        the given data set
    labels : np.ndarray
        The labels as identified by UniDip (and optional added tails)
    n_cluster : int
        The number of clusters
    sorted_X_1d : np.ndarray
        The sorted input data set
    argsorted : np.ndarray
        The indices of the sorted data set
    cluster_boundaries_orig : list
        List of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore

    Returns
    -------
    labels : np.ndarray
        The updated labels after assigning all outliers to neighboring clusters
    """
    # Add outliers to clusters
    for i in range(n_clusters):
        # Convert labels between first position and first cluster
        if i == 0:
            labels[argsorted[:cluster_boundaries_orig[i][0]]] = i
        # Convert labels between current cluster and next cluster
        if i == n_clusters - 1:
            labels[argsorted[cluster_boundaries_orig[i][1]:]] = i
        elif cluster_boundaries_orig[i][1] != cluster_boundaries_orig[i + 1][0]:
            n_points_between_clusters = cluster_boundaries_orig[i + 1][0] - cluster_boundaries_orig[i][1]
            # Naive threshold is center between the two clusters
            threshold_between_clusters = (sorted_X_1d[cluster_boundaries_orig[i][1] - 1] +
                                          sorted_X_1d[cluster_boundaries_orig[i + 1][0]]) / 2
            if n_points_between_clusters > 3:
                # Calculate intercection between ECDF and linear line between end of first cluster and start of next cluster
                ascent = n_points_between_clusters / (
                        sorted_X_1d[cluster_boundaries_orig[i + 1][0]] - sorted_X_1d[cluster_boundaries_orig[i][1]])
                condition = (sorted_X_1d[cluster_boundaries_orig[i][1]:cluster_boundaries_orig[i + 1][0]] -
                             sorted_X_1d[cluster_boundaries_orig[i][1]]) * ascent < np.arange(
                    n_points_between_clusters)
                # We do not want a change position at first or last position
                condition[0] = condition[1]
                condition[-1] = condition[-2]
                # Get interceptions -> one point should be above the ECDF and one below => sum must be equal to 1
                change_points = np.where((condition[:-1] * 1 + condition[1:] * 1) == 1)[0]  # bool must be casted to int
                if len(change_points) != 0:
                    # Get intersection that is nearest to the center between the clusters
                    best_change_point_id = np.argmin(np.abs(
                        sorted_X_1d[cluster_boundaries_orig[i][1] + change_points] - threshold_between_clusters))
                    threshold_between_clusters = sorted_X_1d[
                        cluster_boundaries_orig[i][1] + change_points[best_change_point_id] + 1]
            # Update labels using interception of ECDF and linear connection or use center point if no such point exists
            labels[(X_1d >= sorted_X_1d[cluster_boundaries_orig[i][1]]) & (X_1d < threshold_between_clusters)] = i
            labels[(X_1d >= threshold_between_clusters) & (
                    X_1d < sorted_X_1d[cluster_boundaries_orig[i + 1][0]])] = i + 1
    return labels


class SkinnyDip(BaseEstimator, ClusterMixin):
    """
    Execute the SkinnyDip clustering procedure.
    This approach iteratively executes the univariate clustering algorithm UniDip on each feature.
    The result are clusters formed as hypercubes, which have a much higher density than surrounding noise.
    See UniDip for more information.

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal (default: 0.05)
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap' (default: 'table')
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap' (default: 1000)
    add_tails : bool
        Defines if TailoredDip should try to add tails to the surrounding clusters (default: False)
    outliers : bool
        Defines if outliers should be identified as described by UniDip (default: True)
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used for merging and assigning tails of distributions if 'add_tails' is True (default: 2)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int. Only relevant if pval_strategy is 'bootstrap' (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels

    Examples
    ----------
    >>> from sklearn.datasets import make_blobs
    >>> X, L = make_blobs(1000, 3, centers=5)
    >>> sk = SkinnyDip()
    >>> sk.fit(X)

    References
    ----------
    Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a sea of noise."
    Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

    and

    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """

    def __init__(self, significance: float = 0.05, pval_strategy: str = "table", n_boots: int = 1000,
                 add_tails: bool = False, outliers: bool = True, max_cluster_size_diff_factor: float = 2,
                 random_state: np.random.RandomState = None, debug: bool = False):
        self.significance = significance
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.add_tails = add_tails
        self.outliers = outliers
        self.max_cluster_size_diff_factor = max_cluster_size_diff_factor
        self.random_state = check_random_state(random_state)
        self.debug = debug

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'SkinnyDip':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : SkinnyDip
            this instance of the SkinnyDip algorithm
        """
        n_clusters, labels = _skinnydip(X, self.significance, self.pval_strategy, self.n_boots, self.add_tails,
                                        self.outliers, self.max_cluster_size_diff_factor, self.random_state, self.debug)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        return self


class UniDip(BaseEstimator, ClusterMixin):
    """
    Execute the UniDip clustering procedure.
    This univariate clustering algorithm recursively uses the Dip-test of unimodality to check if a set of samples is distributed uni- or multimodally.
    If the test indicates multimodality, the data will be split into three sets: left of the main mode, the main mode itself and right of the main mode.
    Each set will itself be tested by the Dip-test to see if it has to be further subdivided.

    This implementation includes the extensions made by TailoredDip (see Bauer et al.).

    Parameters
    ----------
    significance : float
        Threshold to decide if the result of the dip-test is unimodal or multimodal (default: 0.05)
    pval_strategy : str
        Defines which strategy to use to receive dip-p-vales. Possibilities are 'table', 'function' and 'bootstrap' (default: 'table')
    n_boots : int
        Number of bootstraps used to calculate dip-p-values. Only necessary if pval_strategy is 'bootstrap' (default: 1000)
    add_tails : bool
        Defines if TailoredDip should try to add tails to the surrounding clusters (default: False)
    outliers : bool
        Defines if outliers should be identified as described by UniDip (default: True)
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used for merging and assigning tails of distributions if 'add_tails' is True (default: 2)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int. Only relevant if pval_strategy is 'bootstrap'  (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_boundaries_ : list
        List of tuples containing the id of the first sample in a cluster and the first sample that is not part of the cluster anymore

    References
    ----------
    Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a sea of noise."
    Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

    and

    Bauer, Lena, et al. "Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering."
    Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2023.
    """

    def __init__(self, significance: float = 0.05, pval_strategy: str = "table", n_boots: int = 1000,
                 add_tails: bool = False, outliers: bool = True, max_cluster_size_diff_factor: float = 2,
                 random_state: np.random.RandomState | int = None, debug: bool = False):
        self.significance = significance
        self.pval_strategy = pval_strategy
        self.n_boots = n_boots
        self.add_tails = add_tails
        self.outliers = outliers
        self.max_cluster_size_diff_factor = max_cluster_size_diff_factor
        self.random_state = check_random_state(random_state)
        self.debug = debug

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'UniDip':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : UniDip
            this instance of the UniDip algorithm
        """
        n_clusters, labels, cluster_boundaries = _tailoreddip(X, self.significance, self.pval_strategy, self.n_boots,
                                                              self.add_tails, self.outliers,
                                                              self.max_cluster_size_diff_factor, self.random_state,
                                                              self.debug)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_boundaries_ = cluster_boundaries
        return self

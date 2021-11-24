"""
Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a
sea of noise." Proceedings of the 22nd ACM SIGKDD
international conference on Knowledge discovery and data
mining. 2016.

@authors Collin Leiber
"""

from cluspy.utils import dip, dip_pval
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _unidip(X_1d, significance, gap_size):
    assert X_1d.ndim == 1, "Data must be 1-dimensional"
    assert gap_size >= 0, "gap_size must not be negative"
    # Create labels array (everything is noise)
    labels = -np.ones(X_1d.shape[0])
    cluster_boundaries = []
    cluster_id = 0

    argsorted = np.argsort(X_1d)
    sorted_X_1d = X_1d[argsorted]

    tmp_borders = [(0, X_1d.shape[0], False)]

    while len(tmp_borders) > 0:
        start, end, is_modal = tmp_borders.pop(0)
        # Get part of data
        tmp_X_1d = sorted_X_1d[start:end]
        dip_value, low_high, _, _ = dip(tmp_X_1d, just_dip=False, is_data_sorted=True)
        dip_pvalue = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
        low = low_high[0]
        high = low_high[1]

        if dip_pvalue < significance:
            # Beware the order! Left -> Center -> Right
            # Other clusters to the right?
            if high + 1 != end - start:
                right_X_1d = sorted_X_1d[start + low:end]
                dip_value = dip(right_X_1d, just_dip=True, is_data_sorted=True)
                dip_pvalue = dip_pval(dip_value, n_points=right_X_1d.shape[0])
                if dip_pvalue < significance:
                    tmp_borders.insert(0, (start + high + 1, end, False))
            # Check area between low and high in next iteration
            if low != high:
                tmp_borders.insert(0, (start + low, start + high + 1, True))
            # Other clusters to the left?
            if low != 0:
                left_X_1d = sorted_X_1d[start:start + high + 1]
                dip_value = dip(left_X_1d, just_dip=True, is_data_sorted=True)
                dip_pvalue = dip_pval(dip_value, n_points=left_X_1d.shape[0])
                if dip_pvalue < significance:
                    tmp_borders.insert(0, (start, start + low, False))
        else:
            # If no modal, mirror data
            if not is_modal:
                _, low, high = _dip_mirrored_data(tmp_X_1d, (low, high), gap_size)
                cluster_start = start + low
                cluster_end = start + high + 1
            else:
                cluster_start = start
                cluster_end = end
            # Set labels
            labels[argsorted[cluster_start:cluster_end]] = cluster_id
            cluster_boundaries.append((cluster_start, cluster_end))
            cluster_id += 1
    n_clusters = cluster_id
    # Merge nearby clusters
    labels, n_clusters, cluster_boundaries = _merge_clusters(sorted_X_1d, argsorted, labels, n_clusters,
                                                             cluster_boundaries, significance)
    return labels, n_clusters, sorted_X_1d, argsorted, cluster_boundaries


def _dip_mirrored_data(X_1d, orig_low_high, gap_size):
    """
    Mirror data to get the correct interval
    :param X_1d sorted data
    :return:
    """
    if gap_size > 0:
        data_range = np.max(X_1d) - np.min(X_1d)  # Needed to create gap with correct scaling
    else:
        data_range = 0
    # Left mirror
    mirrored_addition_left = X_1d[0] - np.flip(X_1d[1:] - X_1d[0]) - gap_size * data_range
    X_1d_left_mirrored = np.append(mirrored_addition_left, X_1d)
    dip_value_left, low_high_left, _, _ = dip(X_1d_left_mirrored, just_dip=False, is_data_sorted=True)
    # Right mirror
    mirrored_addition_right = X_1d[-1] + np.flip(X_1d[-1] - X_1d[:-1]) + gap_size * data_range
    X_1d_right_mirrored = np.append(X_1d, mirrored_addition_right)
    dip_value_right, low_high_right, _, _ = dip(X_1d_right_mirrored, just_dip=False, is_data_sorted=True)
    # Get interval of larger dip
    if dip_value_left > dip_value_right:
        low = low_high_left[0]
        high = low_high_left[1]
        if low < X_1d.shape[0] and high >= X_1d.shape[0]:
            return dip_value_left, orig_low_high[0], orig_low_high[1]
        if low >= X_1d.shape[0]:
            return dip_value_left, low - (X_1d.shape[0] - 1), high - (X_1d.shape[0] - 1)
        else:
            return dip_value_left, (X_1d.shape[0] - 1) - high, (X_1d.shape[0] - 1) - low
    else:
        low = low_high_right[0]
        high = low_high_right[1]
        if low < X_1d.shape[0] and high >= X_1d.shape[0]:
            return dip_value_right, orig_low_high[0], orig_low_high[1]
        if high < X_1d.shape[0]:
            return dip_value_right, low, high
        else:
            return dip_value_right, 2 * (X_1d.shape[0] - 1) - high, 2 * (X_1d.shape[0] - 1) - low


def _merge_clusters(sorted_X_1d, argsorted, labels, n_clusters, cluster_boundaries, significance):
    # For each cluster check left and right partner -> first and last cluster are handled by neighbors
    i = 1
    while i < len(cluster_boundaries) - 1:
        # Dip of i combined with left (i - 1)
        tmp_X_1d = sorted_X_1d[cluster_boundaries[i - 1][0]:cluster_boundaries[i][1]]
        dip_value = dip(tmp_X_1d, just_dip=True, is_data_sorted=True)
        dip_pvalue_left = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
        # Dip of i combined with right (i + 1)
        tmp_X_1d = sorted_X_1d[cluster_boundaries[i][0]:cluster_boundaries[i + 1][1]]
        dip_value = dip(tmp_X_1d, just_dip=True, is_data_sorted=True)
        dip_pvalue_right = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
        if dip_pvalue_left > dip_pvalue_right and dip_pvalue_left >= significance:
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
    return labels, n_clusters, cluster_boundaries


"""
UniDip Plus
"""


def _unidip_plus(X_1d, significance, gap_size):
    # Start by executing UniDip
    labels, n_clusters, sorted_X_1d, argsorted, cluster_boundaries_orig = _unidip(X_1d, significance, gap_size)
    noise_clusters = []
    while True:
        argwhere_outliers = np.where(labels[argsorted] == -1)[0]
        X_rest = sorted_X_1d.copy()
        # Ignore gaps
        for start, end in cluster_boundaries_orig:
            if end != X_rest.shape[0]:
                X_rest[end:] -= X_rest[end] - X_rest[start]
        for start, end in noise_clusters:
            if end != X_rest.shape[0]:
                X_rest[end:] -= X_rest[end] - X_rest[start]
        # Only consider outliers for dip test
        X_rest = X_rest[argwhere_outliers]
        dip_value, low_high, _, _ = dip(X_rest, just_dip=False, is_data_sorted=True)
        dip_pvalue = dip_pval(dip_value, n_points=X_rest.shape[0])
        if dip_pvalue < significance:
            # If structure is multimodal execute UniDip on outliers
            _, _, _, _, cluster_boundaries_new = _unidip(X_rest, significance, gap_size)
            # Add new clusters to existing clusters
            labels, cluster_boundaries_orig, noise_clusters, only_noise_clusters = _add_points_to_existing_cluster(
                sorted_X_1d,
                argsorted,
                cluster_boundaries_new,
                argwhere_outliers,
                cluster_boundaries_orig,
                noise_clusters,
                labels,
                dip_pvalue)
        else:
            # If strucure is unimodal, mirror data to see if a reasonable peak exists
            dip_value_mirror, low_mirror, high_mirror = _dip_mirrored_data(X_rest, low_high, 0)
            dip_pvalue_mirror = dip_pval(dip_value_mirror, n_points=(X_rest.shape[0] * 2 - 1))
            low_high = (low_mirror, high_mirror)
            if dip_pvalue_mirror < significance:
                # Add peak to existing clusters
                labels, cluster_boundaries_orig, noise_clusters, only_noise_clusters = _add_points_to_existing_cluster(
                    sorted_X_1d, argsorted, [low_high], argwhere_outliers, cluster_boundaries_orig, noise_clusters,
                    labels, dip_pvalue)
            else:
                # If distribution is still unimodal -> terminate
                break
        if only_noise_clusters:
            # If only useless noise clusters have been found -> terminate
            break
    if len(noise_clusters) != 0:
        labels[labels == -2] = -1
    return labels, n_clusters


def _add_points_to_existing_cluster(sorted_X_1d, argsorted, cluster_boundaries_new, argwhere_outliers,
                                    cluster_boundaries_orig, noise_clusters, labels, uniform_pval):
    cluster_boundaries_orig_copy = cluster_boundaries_orig.copy()
    only_noise_clusters = True
    for start_new, end_new in cluster_boundaries_new:
        start_new = argwhere_outliers[start_new]
        end_new = argwhere_outliers[end_new - 1] + 1
        matching_cluster_found = False
        # Pvalues should be larger than pvalue of uniform noise
        best_probs = np.zeros(end_new - start_new) * uniform_pval
        for i, boundary_orig in enumerate(cluster_boundaries_orig):
            start_orig, end_orig = boundary_orig
            # Do clusters touch?
            if start_new <= start_orig or end_new >= end_orig:
                # Min start should be end of last cluster or 0
                if i == 0:
                    current_start = 0
                else:
                    current_start = cluster_boundaries_orig[i - 1][1]
                current_start = max(current_start, start_new)
                # Max end should be start of next cluster or last entry
                if i == len(cluster_boundaries_orig) - 1:
                    current_end = len(labels)
                else:
                    current_end = cluster_boundaries_orig[i + 1][0]
                current_end = min(current_end, end_new)
                # Expand cluster to the left
                if current_start != start_orig and start_new < start_orig and end_new >= start_orig:
                    matching_cluster_found = True
                    X_tmp = sorted_X_1d[current_start:end_orig]
                    dip_value = dip(X_tmp, just_dip=True, is_data_sorted=True)
                    dip_pvalue = dip_pval(dip_value, n_points=X_tmp.shape[0])
                    max_dip_pval = np.max(best_probs[current_start - start_new:start_orig - start_new])
                    if dip_pvalue > max_dip_pval:
                        if max_dip_pval > 0:
                            cluster_boundaries_orig_copy[i - 1] = (
                            cluster_boundaries_orig_copy[i - 1][0], cluster_boundaries_orig[i - 1][1])
                        labels[argsorted[current_start:start_orig]] = i
                        best_probs[current_start - start_new:start_orig - start_new] = dip_pvalue
                        cluster_boundaries_orig_copy[i] = (current_start, cluster_boundaries_orig_copy[i][1])
                # Expand cluster to the right
                if current_end != end_orig and start_new <= end_orig and end_new > end_orig:
                    matching_cluster_found = True
                    X_tmp = sorted_X_1d[start_orig:current_end]
                    dip_value = dip(X_tmp, just_dip=True, is_data_sorted=True)
                    dip_pvalue = dip_pval(dip_value, n_points=X_tmp.shape[0])
                    max_dip_pval = np.max(best_probs[end_orig - start_new:current_end - start_new])
                    if dip_pvalue > max_dip_pval:
                        labels[argsorted[end_orig:current_end]] = i
                        best_probs[end_orig - start_new:current_end - start_new] = dip_pvalue
                        cluster_boundaries_orig_copy[i] = (cluster_boundaries_orig_copy[i][0], current_end)
        # If structures matches no cluster ignore structure for next iteration (label = -2)
        if not matching_cluster_found:
            labels[argsorted[start_new:end_new]] = -2
            noise_clusters.append((start_new, end_new))
        else:
            only_noise_clusters = False
    return labels, cluster_boundaries_orig_copy, noise_clusters, only_noise_clusters


class UniDip(BaseEstimator, ClusterMixin):

    def __init__(self, significance=0.05, gap_size=0):
        self.significance = significance
        self.gap_size = gap_size

    def fit(self, X, y=None):
        labels, n_clusters, _, _, _ = _unidip(X, self.significance, self.gap_size)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        return self


class UniDipPlus(UniDip):

    def fit(self, X, y=None):
        labels, n_clusters = _unidip_plus(X, self.significance, self.gap_size)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        return self

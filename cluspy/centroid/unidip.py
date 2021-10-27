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


def _unidip(X_1d, significance, outliers, gap_size):
    assert X_1d.ndim == 1, "Data must be 1-dimensional"
    assert gap_size >= 0, "gap_size must not be negative"
    # Create labels array (everything is noise)
    labels = -np.ones(X_1d.shape[0])
    cluster_id = 0

    argsorted = np.argsort(X_1d)
    sorted_X_1d = X_1d[argsorted]

    tmp_borders = [(0, X_1d.shape[0], True)]

    while len(tmp_borders) > 0:
        start, end, is_modal = tmp_borders.pop(0)
        # Get part of data
        tmp_X_1d = sorted_X_1d[start:end]
        dip_value, low_high, _ = dip(tmp_X_1d, just_dip=False, is_data_sorted=True)
        dip_pvalue = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
        low = low_high[2]
        high = low_high[3]

        if dip_pvalue < significance:
            # Other clusters to the right?
            if end - start != high + 1:
                right_X_1d = sorted_X_1d[start + low:end]
                dip_value = dip(right_X_1d, just_dip=True, is_data_sorted=True)
                dip_pvalue = dip_pval(dip_value, n_points=right_X_1d.shape[0])
                if dip_pvalue < significance:
                    tmp_borders.insert(0, (start + high + 1, end, False))
                elif not outliers:
                    # Model should reach end
                    high = end - start - 1
            # Other clusters to the left?
            if 0 != low:
                left_X_1d = sorted_X_1d[start:start + high + 1]
                dip_value = dip(left_X_1d, just_dip=True, is_data_sorted=True)
                dip_pvalue = dip_pval(dip_value, n_points=left_X_1d.shape[0])
                if dip_pvalue < significance:
                    tmp_borders.insert(0, (start, start + low, False))
                elif not outliers:
                    # Model should reach start
                    low = 0
            if low != high:
                tmp_borders.insert(0, (start + low, start + high + 1, True))
        else:
            # If no modal (and outliers should be identified) mirror data
            if outliers and not is_modal:
                low, high = _dip_mirrored_data(sorted_X_1d[start:end], (low, high), gap_size)
                cluster_start = start + low
                cluster_end = start + high + 1
            else:
                cluster_start = start
                cluster_end = end
            # Set labels
            labels[argsorted[cluster_start:cluster_end]] = cluster_id
            cluster_id += 1
    return labels, cluster_id


def _dip_mirrored_data(X_1d, orig_low_high, gap_size):
    """
    Mirror data to get the correct interval
    :return:
    """
    data_range = np.max(X_1d) - np.min(X_1d) # Needed to create gap with correct scaling
    # Left mirror
    mirrored_addition_left = X_1d[0] - np.flip(X_1d - X_1d[0]) - gap_size * data_range
    X_1d_left_mirrored = np.append(mirrored_addition_left, X_1d)
    dip_value_left, low_high_left, _ = dip(X_1d_left_mirrored, just_dip=False, is_data_sorted=True)
    # Right mirror
    mirrored_addition_right = X_1d[-1] + np.flip(X_1d[-1] - X_1d) + gap_size * data_range
    X_1d_right_mirrored = np.append(X_1d, mirrored_addition_right)
    dip_value_right, low_high_right, _ = dip(X_1d_right_mirrored, just_dip=False, is_data_sorted=True)
    # Get interval of larger dip
    if dip_value_left > dip_value_right:
        low = low_high_left[2]
        high = low_high_left[3]
        if low < X_1d.shape[0] and high >= X_1d.shape[0]:  # Should not happen due to gap
            return orig_low_high
        if low >= X_1d.shape[0]:
            return low - X_1d.shape[0], high - X_1d.shape[0]
        else:
            return X_1d.shape[0] - high, X_1d.shape[0] - low
    else:
        low = low_high_right[2]
        high = low_high_right[3]
        if low < X_1d.shape[0] and high >= X_1d.shape[0]:  # Should not happen due to gap
            return orig_low_high
        if high < X_1d.shape[0]:
            return low, high
        else:
            return 2 * X_1d.shape[0] - high, 2 * X_1d.shape[0] - low


class UniDip(BaseEstimator, ClusterMixin):

    def __init__(self, significance=0.05, outliers=True, gap_size=0.1):
        self.significance = significance
        self.outliers = outliers
        self.gap_size = gap_size

    def fit(self, X, y=None):
        labels, n_clusters = _unidip(X, self.significance, self.outliers, self.gap_size)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        return self
"""
Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a
sea of noise." Proceedings of the 22nd ACM SIGKDD
international conference on Knowledge discovery and data
mining. 2016.

@authors Collin Leiber
"""

from cluspy.utils import dip_test, dip_pval
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _unidip(X_1d, significance, already_sorted):
    assert X_1d.ndim == 1, "Data must be 1-dimensional"
    # Create labels array (everything is noise)
    labels = -np.ones(X_1d.shape[0])
    cluster_boundaries = []
    cluster_id = 0

    if already_sorted:
        argsorted = np.arange(X_1d.shape[0])
    else:
        argsorted = np.argsort(X_1d)
        X_1d = X_1d[argsorted]

    tmp_borders = [(0, X_1d.shape[0], False)]
    while len(tmp_borders) > 0:
        start, end, is_modal = tmp_borders.pop(0)
        # Get part of data
        tmp_X_1d = X_1d[start:end]
        dip_value, low_high, _ = dip_test(tmp_X_1d, just_dip=False, is_data_sorted=True)
        dip_pvalue = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
        low = low_high[0]
        high = low_high[1]
        if dip_pvalue < significance:
            # Beware the order in which entries are added to tmp_borders! right -> center -> left
            # Other clusters to the right?
            if high + 1 != end - start:
                right_X_1d = X_1d[start + low:end]
                dip_value = dip_test(right_X_1d, just_dip=True, is_data_sorted=True)
                dip_pvalue = dip_pval(dip_value, n_points=right_X_1d.shape[0])
                if dip_pvalue < significance:
                    tmp_borders.insert(0, (start + high + 1, end, False))
            # Check area between low and high in next iteration
            if low != high:
                tmp_borders.insert(0, (start + low, start + high + 1, True))
            # Other clusters to the left?
            if low != 0:
                left_X_1d = X_1d[start:start + high + 1]
                dip_value = dip_test(left_X_1d, just_dip=True, is_data_sorted=True)
                dip_pvalue = dip_pval(dip_value, n_points=left_X_1d.shape[0])
                if dip_pvalue < significance:
                    tmp_borders.insert(0, (start, start + low, False))
        else:
            # If no modal, mirror data
            if not is_modal:
                _, low, high = _dip_mirrored_data(tmp_X_1d, (low, high))
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
    labels, n_clusters, cluster_boundaries = _merge_clusters(X_1d, argsorted, labels, n_clusters,
                                                             cluster_boundaries, significance)
    return labels, n_clusters, X_1d, argsorted, cluster_boundaries


def _dip_mirrored_data(X_1d, orig_low_high):
    """
    Mirror data to get the correct interval
    :param X_1d sorted data
    :return:
    """
    # Left mirror
    mirrored_addition_left = X_1d[0] - np.flip(X_1d[1:] - X_1d[0])
    X_1d_left_mirrored = np.append(mirrored_addition_left, X_1d)
    dip_value_left, low_high_left, _ = dip_test(X_1d_left_mirrored, just_dip=False, is_data_sorted=True)
    # Right mirror
    mirrored_addition_right = X_1d[-1] + np.flip(X_1d[-1] - X_1d[:-1])
    X_1d_right_mirrored = np.append(X_1d, mirrored_addition_right)
    dip_value_right, low_high_right, _ = dip_test(X_1d_right_mirrored, just_dip=False, is_data_sorted=True)
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


def _merge_clusters(X_1d, argsorted, labels, n_clusters, cluster_boundaries, significance):
    # For each cluster check left and right partner -> first and last cluster are handled by neighbors
    i = 1
    while i < len(cluster_boundaries) - 1:
        cluster_size_center = cluster_boundaries[i][1] - cluster_boundaries[i][0]
        # Dip of i combined with left (i - 1)
        cluster_size_left = cluster_boundaries[i - 1][1] - cluster_boundaries[i - 1][0]
        start_left = max(cluster_boundaries[i - 1][0], cluster_boundaries[i - 1][1] - 2 * cluster_size_center)
        end_left = min(cluster_boundaries[i][1], cluster_boundaries[i][0] + 2 * cluster_size_left)
        tmp_X_1d = X_1d[start_left:end_left]
        # tmp_X_1d = X_1d[cluster_boundaries[i - 1][0]:cluster_boundaries[i][1]]
        dip_value = dip_test(tmp_X_1d, just_dip=True, is_data_sorted=True)
        dip_pvalue_left = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
        # Dip of i combined with right (i + 1)
        cluster_size_right = cluster_boundaries[i + 1][1] - cluster_boundaries[i + 1][0]
        start_right = max(cluster_boundaries[i][0], cluster_boundaries[i][1] - 2 * cluster_size_right)
        end_right = min(cluster_boundaries[i + 1][1], cluster_boundaries[i + 1][0] + 2 * cluster_size_center)
        tmp_X_1d = X_1d[start_right:end_right]
        # tmp_X_1d = X_1d[cluster_boundaries[i][0]:cluster_boundaries[i + 1][1]]
        dip_value = dip_test(tmp_X_1d, just_dip=True, is_data_sorted=True)
        dip_pvalue_right = dip_pval(dip_value, n_points=tmp_X_1d.shape[0])
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
    return labels, n_clusters, cluster_boundaries


"""
UniDip Plus
"""


def _unidip_plus(X_1d, significance, outliers):
    # Start by executing UniDip
    labels, n_clusters, sorted_X_1d, argsorted, cluster_boundaries_orig = _unidip(X_1d, significance, False)
    cluster_boundaries_orig = [(boundary[0], boundary[1], j) for j, boundary in enumerate(cluster_boundaries_orig)]
    i = 0
    while i <= len(cluster_boundaries_orig):
        while True:
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
            if end - start < 4:
                break
            X_tmp = sorted_X_1d[start:end]
            # Calculate mirrored dip to see if there is some relevant structure left between the clusters
            dip_value_mirror, _, _ = _dip_mirrored_data(X_tmp, (None, None))
            dip_pvalue_mirror = dip_pval(dip_value_mirror, n_points=(X_tmp.shape[0] * 2 - 1))
            if dip_pvalue_mirror < significance:
                # Execute UniDip in the noise data
                labels_tmp, n_clusters_new, _, _, cluster_boundaries_new = _unidip(X_tmp, significance, True)
                dip_pvalue_left = -1
                dip_pvalue_right = -1
                # Append first found structure to cluster before
                # Calculate dip of first found structure with cluster before
                if i != 0 and cluster_boundaries_orig[i - 1][2] != -1:
                    # cluster_range = cluster_boundaries_new[0][1] - cluster_boundaries_new[0][0]
                    cluster_range = (start + cluster_boundaries_new[0][1]) - cluster_boundaries_orig[i - 1][1]
                    # Use a maximum of cluster_range points of left cluster to see if transition is unimodal
                    start_left = max(cluster_boundaries_orig[i - 1][0],
                                     cluster_boundaries_orig[i - 1][1] - 2 * cluster_range)
                    end_left = start + cluster_boundaries_new[0][1]
                    dip_value_left = dip_test(sorted_X_1d[start_left:end_left], just_dip=True, is_data_sorted=True)
                    dip_pvalue_left = dip_pval(dip_value_left, n_points=end_left - start_left)
                # Append last found structure to cluster after
                # Calculate dip of last found structure with cluster after
                if i != len(cluster_boundaries_orig) and cluster_boundaries_orig[i][2] != -1:
                    # cluster_range = cluster_boundaries_new[-1][1] - cluster_boundaries_new[-1][0]
                    cluster_range = cluster_boundaries_orig[i][0] - (start + cluster_boundaries_new[-1][0])
                    start_right = start + cluster_boundaries_new[-1][0]
                    # Use a maximum of cluster_range points of right cluster to see if transition is unimodal
                    end_right = min(cluster_boundaries_orig[i][1], cluster_boundaries_orig[i][0] + 2 * cluster_range)
                    dip_value_right = dip_test(sorted_X_1d[start_right:end_right], just_dip=True, is_data_sorted=True)
                    dip_pvalue_right = dip_pval(dip_value_right, n_points=end_right - start_right)
                # --- Extend clusters; Beware the order in which entries are added as boundaries! right -> center -> left
                # Does last found structure fit the cluster after? If so, extend cluster
                if dip_pvalue_right >= significance and (n_clusters_new > 1 or dip_pvalue_right >= dip_pvalue_left):
                    cluster_id = cluster_boundaries_orig[i][2]
                    labels[argsorted[start_right:cluster_boundaries_orig[i][1]]] = cluster_id
                    cluster_boundaries_orig[i] = (start_right, cluster_boundaries_orig[i][1], cluster_id)
                elif dip_pvalue_right < significance and (n_clusters_new > 1 or dip_pvalue_left < significance):
                    cluster_boundaries_orig.insert(i, (
                        start + cluster_boundaries_new[-1][0], start + cluster_boundaries_new[-1][1], -1))
                # Add all found structures except first and last as noise
                for j in range(n_clusters_new - 2, 0, -1):
                    cluster_boundaries_orig.insert(i, (
                        start + cluster_boundaries_new[j][0], start + cluster_boundaries_new[j][1], -1))
                # Does first found strucure fit the cluster before? If so, extend cluster
                if dip_pvalue_left >= significance and (n_clusters_new > 1 or dip_pvalue_left > dip_pvalue_right):
                    cluster_id = cluster_boundaries_orig[i - 1][2]
                    labels[argsorted[cluster_boundaries_orig[i - 1][0]:end_left]] = cluster_id
                    cluster_boundaries_orig[i - 1] = (cluster_boundaries_orig[i - 1][0], end_left, cluster_id)
                elif dip_pvalue_left < significance and n_clusters_new > 1:
                    cluster_boundaries_orig.insert(i, (
                        start + cluster_boundaries_new[0][0], start + cluster_boundaries_new[0][1], -1))
            else:
                break
        i += 1
    if not outliers:
        # Add outliers to closest cluster
        i = 0
        while i < len(cluster_boundaries_orig):
            if cluster_boundaries_orig[i][2] == -1:
                del cluster_boundaries_orig[i]
                continue
            if i < len(cluster_boundaries_orig) - 1 and cluster_boundaries_orig[i + 1][2] == -1:
                del cluster_boundaries_orig[i + 1]
                continue
            # Convert labels between first position and first cluster
            if i == 0:
                labels[argsorted[:cluster_boundaries_orig[i][0]]] = cluster_boundaries_orig[i][2]
            # Convert labels between current cluster and next cluster
            if i == len(cluster_boundaries_orig) - 1:
                labels[argsorted[cluster_boundaries_orig[i][1]:]] = cluster_boundaries_orig[i][2]
            else:
                border = sorted_X_1d[cluster_boundaries_orig[i][1] - 1] + (
                        sorted_X_1d[cluster_boundaries_orig[i + 1][0]] - sorted_X_1d[cluster_boundaries_orig[i][1] - 1]) / 2
                labels[(X_1d >= sorted_X_1d[cluster_boundaries_orig[i][1]]) & (X_1d < border)] = \
                cluster_boundaries_orig[i][2]
                labels[(X_1d >= border) & (X_1d < sorted_X_1d[cluster_boundaries_orig[i + 1][0]])] = \
                    cluster_boundaries_orig[i + 1][2]
            i += 1
    return labels, n_clusters


class UniDip(BaseEstimator, ClusterMixin):

    def __init__(self, significance=0.05):
        self.significance = significance

    def fit(self, X, y=None):
        labels, n_clusters, _, _, _ = _unidip(X, self.significance, False)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        return self


class UniDipPlus(UniDip):

    def __init__(self, significance=0.05, outliers=True):
        super().__init__(significance)
        self.outliers = outliers

    def fit(self, X, y=None):
        labels, n_clusters = _unidip_plus(X, self.significance, self.outliers)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        return self

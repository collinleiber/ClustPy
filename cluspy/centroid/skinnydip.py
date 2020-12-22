"""
Maurus, Samuel, and Claudia Plant. "Skinny-dip: clustering in a
sea of noise." Proceedings of the 22nd ACM SIGKDD
international conference on Knowledge discovery and data
mining. 2016.

Python version of following implementation in R:
https://github.com/samhelmholtz/skinny-dip

@authors Samuel Maurus (original R implementation), Collin Leiber (Python implementation)
"""

import numpy as np
from cluspy.utils.diptest import dip, dip_pval


def _skinnydip_clustering_full_space(X, significance, debug=False):
    hypercubes = _find_cluster_hypercubes(np.identity(X.shape[1]), np.zeros((0, 2)), X, significance, debug)
    # Object-assignment
    labels = np.full(X.shape[0], -1)
    for i in range(len(hypercubes)):
        hypercube = hypercubes[i]
        # Get the objects that fall within this hypercube
        objects_in_hypercube = [True] * X.shape[0]
        for j in range(hypercube.shape[0]):
            objects_in_hypercube = (objects_in_hypercube) & (X[:, j] >= hypercube[j, 0]) & (X[:, j] <= hypercube[j, 1])
        labels[objects_in_hypercube] = i
    return labels, len(hypercubes)


def _find_cluster_hypercubes(subspace, existing_hypercube, filtered_data, significance, debug):
    subspace_dims = subspace.shape[1]
    if existing_hypercube.shape[0] >= subspace_dims:
        # Our hypercube is complete...return it
        return [existing_hypercube]
    if filtered_data.shape[0] == 0:
        # No objects: no cluster
        return []
    next_dimension = existing_hypercube.shape[0]
    # Get the next direction onto which we'll project our data
    projection_vector = subspace[:, next_dimension]
    # Project the data onto that direction and sort it
    projected_data = np.matmul(filtered_data, projection_vector)
    sorted_projected_data = np.sort(projected_data)
    # Get the modal intervals along this direction. We get a matrix back where the rows are the modes,
    # with the start/end values given in the two columns
    # We always get at least one mode back
    modal_intervals = _extract_modal_intervals(sorted_projected_data, significance, debug)
    num_modes_found = modal_intervals.shape[0]

    hypercubes = []
    for i in range(num_modes_found):
        # Refine our hypercube
        refined_hypercube = np.r_[existing_hypercube, modal_intervals[i, :].reshape(-1, 2)]
        # refined_hypercube = np.r_[existing_hypercube,modal_intervals[i,:].reshape((1, 2))]
        refined_data = filtered_data[(filtered_data[:, next_dimension] >= modal_intervals[i, 0]) & (
                filtered_data[:, next_dimension] <= modal_intervals[i, 1]), :]
        cluster_hypercubes = _find_cluster_hypercubes(subspace, refined_hypercube, refined_data, significance, debug)
        hypercubes = hypercubes + cluster_hypercubes

    return hypercubes


# This method is our new method for finding the modes in a 1d (ordered) sample
def _extract_modal_intervals(X, significance, debug):
    # Find the raw clusters using our recursive approach
    # Note: here we're saying that we're not testing a modal interval. This means that, if the full distribution is not multimodal, it will only return its estimate of the mode (not the full distribution)
    clusters_raw = _get_clusters_in_interval(X, 0, X.shape[0] - 1, "----", False, significance, debug)
    clusters_raw = np.array(clusters_raw)
    # Consolidation
    clusters = _merge_intervals(X, clusters_raw, debug)
    clusters = np.array(clusters)

    cluster_starts = clusters[list(range(0, len(clusters), 2))]
    cluster_ends = clusters[list(range(1, len(clusters), 2))]
    return np.c_[X[cluster_starts], X[cluster_ends]]


# Note that here the indexes are always passed/returned in a global sense
def _get_clusters_in_interval(X, index_start, index_end, prefix, testing_modal_interval_only, significance, debug):
    if debug:
        print("{0}Checking interval [{1},{2}]".format(prefix, index_start, index_end))
    # Subset the data...that is, we want to recursively look at only the data between indexStart and indexEnd and search for modes in that distribution
    data_subset = X[index_start:index_end + 1]

    # Run the dip test on the data subset
    dip_value, low_high, modal_triangle = dip(data_subset, just_dip=False)
    dip_pvalue = dip_pval(dip_value, data_subset.shape[0])
    modal_interval_left = index_start + low_high[0]
    modal_interval_right = index_start + low_high[1]

    # Check for non-significance using our significance threshold. If the result is non-significant, we'll assume we only have one cluster (unimodal)
    if dip_pvalue > significance:
        if testing_modal_interval_only:
            if debug:
                print(
                    "{0}Modal Interval [{1},{2}] is unimodally distributed (p-value {3})...returning it as a cluster...".format(
                        prefix, index_start, index_end, dip_pvalue))
            return [index_start, index_end]
        else:
            # Here we know we're finding the "last" cluster. For the unimodal case where the mode is indeed just a point of a small interval, the dip test has the tendency to
            # return a very small modal interval (makes sense). This is bad in our case because it means that our core cluster is typically going to be very small in relation to
            # the others that are found. For this reason we need a mechanism for finding out what the "full" core cluster is
            # Our mechanism: mirror the data and run the dip on it. We're sure that it's then multimodal, and the dip should find "fully" one of the modes as a larger interval
            if debug:
                print(
                    "{0}Interval [{1},{2}] is unimodally distributed (p-value {3})...the modal interval found in was [{4},{5}]. Proceeding to mirror data to extract a fatter cluster here...".format(
                        prefix, index_start, index_end, dip_pvalue, modal_interval_left, modal_interval_right))

            ## Get left and right-mirrored results
            left_mirrored_dataset = _mirror_dataset(data_subset, True)
            left_mirrored_dip_value, left_mirrored_low_high, left_mirrored_modal_triangle = \
                dip(left_mirrored_dataset, is_data_sorted=True, just_dip=False)
            right_mirrored_dataset = _mirror_dataset(data_subset, False)
            right_mirrored_dip_value, right_mirrored_low_high, right_mirrored_modal_triangle = \
                dip(right_mirrored_dataset, is_data_sorted=True, just_dip=False)
            if left_mirrored_dip_value > right_mirrored_dip_value:
                cluster_range = _map_index_range_to_ordered_mirrored_data_index_range_in_original_ordered_data(
                    left_mirrored_low_high, modal_interval_left, modal_interval_right, data_subset.shape[0], True,
                    index_start)
                if debug:
                    print(
                        "{0}Modal interval on the left-mirrored data was [{1},{2}]...which corresponds to a cluser (which we'll return now) in the original data of [{3},{4}].".format(
                            prefix, left_mirrored_low_high[0], left_mirrored_low_high[1], cluster_range[0],
                            cluster_range[1]))
            else:
                cluster_range = _map_index_range_to_ordered_mirrored_data_index_range_in_original_ordered_data(
                    right_mirrored_low_high, modal_interval_left, modal_interval_right, data_subset.shape[0], False,
                    index_start)
                if debug:
                    print(
                        "{0}Modal interval on the right-mirrored data was [{1},{2}]...which corresponds to a cluser (which we'll return now) in the original data of [{3},{4}].".format(
                            prefix, right_mirrored_low_high[0], right_mirrored_low_high[1], cluster_range[0],
                            cluster_range[1]))
            return cluster_range

    if debug:
        print(
            "{0}Modal interval [{1},{2}], p={3}".format(prefix, modal_interval_left, modal_interval_right, dip_pvalue))

    # Otherwise, expand the modal interval to see if it has more than one cluster
    modal_interval_clusters = _get_clusters_in_interval(X, modal_interval_left, modal_interval_right, prefix + "----",
                                                        True,
                                                        significance, debug)

    # Now we need to look at the various cases.
    # If we only have a left interval, we just need to proceed in it...there must be at least one cluster there
    # If we only have a right interval, we just need to proceed in it...there must be at least one cluster there
    # If we have both, we need to consider both...there COULD be one or more on either side
    left_interval_exists = (index_start < modal_interval_left)
    right_interval_exists = (index_end > modal_interval_right)
    if not left_interval_exists and not right_interval_exists:
        raise Exception(
            "We found a statistical multimodality, but the modal interval is the full interval! This should never happen!")
    if not left_interval_exists and right_interval_exists:
        if debug:
            print(
                "{0}Interval [{1},{2}] is significantly MULTIMODAL. The modal interval [{3},{4}] leaves no other points to the left, so we can continue to the right...".format(
                    prefix, index_start, index_end, modal_interval_left, modal_interval_right))
        right_clusters = _get_clusters_in_interval(X, modal_interval_right + 1, index_end, prefix + "----", False,
                                                   significance, debug)
        return modal_interval_clusters + right_clusters
    if left_interval_exists and not right_interval_exists:
        if debug:
            print(
                "{0}Interval [{1},{2}] is significantly MULTIMODAL. The modal interval [{3},{4}] leaves no other points to the right, so we can continue to the left...".format(
                    prefix, index_start, index_end, modal_interval_left, modal_interval_right))
        left_clusters = _get_clusters_in_interval(X, index_start, modal_interval_left - 1, prefix + "----", False,
                                                  significance, debug)
        return modal_interval_clusters + left_clusters

    # Otherwise, we have the general case of both intervals (left and right)
    # Here we need to check both, including the closest cluster from the modal interval
    if len(modal_interval_clusters) > 2:
        # More than one cluster in the modal interval, so include the closest for the test of each extreme
        left_clusters = _get_clusters_in_interval(X, index_start, modal_interval_clusters[1], prefix + "----", False,
                                                  significance, debug)
        right_clusters = _get_clusters_in_interval(X, modal_interval_clusters[-2], index_end, prefix + "----", False,
                                                   significance, debug)
        return modal_interval_clusters + left_clusters + right_clusters
    else:
        if debug:
            print(
                "{0}Interval [{1},{2}] is significantly MULTIMODAL. The modal interval [{3},{4}] is unimodal with intervals left and right of it. Checking these neighbouring intervals with the modal interval...".format(
                    prefix, index_start, index_end, modal_interval_left, modal_interval_right))
        # Single cluster in modal interval. We hence know that there exists cluster(s) outside the modal interval. Find (them) by just focusing on the extreme intervals
        left_clusters = _get_clusters_in_interval(X, index_start, modal_interval_right, prefix + "----", False,
                                                  significance, debug)
        right_clusters = _get_clusters_in_interval(X, modal_interval_left, index_end, prefix + "----", False,
                                                   significance, debug)
        return modal_interval_clusters + left_clusters + right_clusters


# Shifts to zero start, then mirrors, then returns the mirrored data
# E.g. input c(2,3,4), output c(-2,-1,0,1,2)
# Assumes an ordered input
def _mirror_dataset(X, mirror_left):
    if mirror_left:
        min_value = np.min(X)
        data_shifted = X - min_value
        data_shifted_gt_zero = data_shifted[data_shifted > 0]
        data_shifted_gt_zero_mirrored = - data_shifted_gt_zero
        mirrored_dataset = np.r_[data_shifted_gt_zero_mirrored, 0, data_shifted_gt_zero]
    else:
        max_value = np.max(X)
        data_shifted = X - max_value
        data_shifted_lt_zero = data_shifted[data_shifted < 0]
        data_shifted_lt_zero_mirrored = - data_shifted_lt_zero
        mirrored_dataset = np.r_[data_shifted_lt_zero, 0, data_shifted_lt_zero_mirrored]
    return np.sort(mirrored_dataset)


def _map_index_range_to_ordered_mirrored_data_index_range_in_original_ordered_data(low_high, lower_fallback_index,
                                                                                   upper_fallback_index,
                                                                                   length_of_original_data, mirror_left,
                                                                                   offset_index):
    # Let's say our original data had a length of 2
    # Then the mirrored data will have a length of 3
    # The mirrored data will always have an odd length
    # The zero point will be at lengthOfOriginalData
    lower_index_to_map = low_high[0]
    upper_index_to_map = low_high[1]
    if (lower_index_to_map < length_of_original_data - 1) and (upper_index_to_map > length_of_original_data - 1):
        return [lower_fallback_index, upper_fallback_index]

    lower_index_mapped = _map_index_to_ordered_mirrored_data_index_in_original_ordered_data(lower_index_to_map,
                                                                                            length_of_original_data,
                                                                                            mirror_left)
    upper_index_mapped = _map_index_to_ordered_mirrored_data_index_in_original_ordered_data(upper_index_to_map,
                                                                                            length_of_original_data,
                                                                                            mirror_left)
    if lower_index_mapped > upper_index_mapped:
        return [upper_index_mapped + offset_index, lower_index_mapped + offset_index]
    else:
        return [lower_index_mapped + offset_index, upper_index_mapped + offset_index]


def _map_index_to_ordered_mirrored_data_index_in_original_ordered_data(index_to_map, length_of_original_data,
                                                                       mirror_left):
    if mirror_left:
        if index_to_map >= length_of_original_data - 1:
            return index_to_map - length_of_original_data + 1
        else:
            return length_of_original_data - index_to_map - 1
    else:
        if index_to_map > length_of_original_data - 1:
            return (2 * (length_of_original_data - 1)) - index_to_map
        else:
            return index_to_map


def _merge_intervals(ordered_data, intervals, debug):
    # We first need to find the merged clusters (any overlaps are merged), such that we only have mutually-exclusive clusters
    cluster_starting_indices = intervals[list(range(0, len(intervals), 2))]
    cluster_ending_indices = intervals[list(range(1, len(intervals), 2))]
    cluster_starting_indices_order_mappings = np.argsort(cluster_starting_indices)
    cluster_starting_indices_ordered = cluster_starting_indices[cluster_starting_indices_order_mappings]
    cluster_ending_indices_ordered = cluster_ending_indices[cluster_starting_indices_order_mappings]
    clusters = [cluster_starting_indices_ordered[0], cluster_ending_indices_ordered[0]]

    for i in range(1, len(cluster_starting_indices_ordered)):
        ## Get the current interval
        end_in_question = clusters[len(clusters) - 1]

        if end_in_question < cluster_starting_indices_ordered[i]:
            clusters = clusters + [cluster_starting_indices_ordered[i]] + [cluster_ending_indices_ordered[i]]
        elif end_in_question < cluster_ending_indices_ordered[i]:
            clusters[len(clusters) - 1] = cluster_ending_indices_ordered[i]

    # Now we do our "consolidation" step
    # The idea is that we merge in any "tails", "fringes" or "artefacts" that aren't
    # statistically-significant enough to cause multimodality.
    # How? We know that our clusters are ordered.
    # We iterate though our clusters and perform the dip test on the range defined by successfive pairs
    # If a pair has a non-significant multimodality, we call the entire range defined by that successive pair a single cluster
    consolidated_clusters = _consolidate_clusters(ordered_data, clusters, 0, debug)
    return consolidated_clusters


def _consolidate_clusters(ordered_data, clusters, index, debug):
    # If index > length-1 done
    # do dip
    # If significant
    # increment index and recurse with index++
    # else
    # merge and recurse with index

    num_clusters_left = len(clusters) / 2
    if index > (num_clusters_left - 2):
        return clusters
    starting_index = (index * 2)
    ending_index = (index + 1) * 2 + 1
    starting_point_index = clusters[starting_index]
    ending_point_index = clusters[ending_index]
    dip_value = dip(ordered_data[starting_point_index:ending_point_index], just_dip=True, is_data_sorted=True)
    dip_p_value = dip_pval(dip_value, ending_point_index - starting_point_index)
    if dip_p_value < 0.05:
        if debug:
            print("Range {0} to {1} is significant...we're happy with that cluster!".format(starting_point_index,
                                                                                            ending_point_index))
        # significant multimodality...continue with the next index
        return _consolidate_clusters(ordered_data, clusters, index + 1, debug)
    else:
        if debug:
            print("Range {0} to {1} is not significant: merging".format(starting_point_index, ending_point_index))
        ## not significant...merge and repeat with the same index
        clusters = clusters[0:starting_index + 1] + clusters[ending_index:len(clusters)]
        return _consolidate_clusters(ordered_data, clusters, index, debug)


class SkinnyDip():

    def __init__(self, significance=0.05):
        self.significance = significance

    def fit(self, X):
        labels, n_clusters = _skinnydip_clustering_full_space(X, self.significance)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
      
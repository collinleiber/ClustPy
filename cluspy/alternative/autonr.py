from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
import numpy as np
import cluspy.alternative.nrkmeans as nrkm
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

NOISE_SPACE_SPLIT = 0
CLUSTER_SPACE_SPLIT = 1
CLUSTER_SPACE_MERGE = 2

SIMILARITY_THRESHOLD = 1e-5


def _autonr(X, nrkmeans_repetitions, outliers, max_subspaces, max_n_clusters, mdl_for_noisespace,
            max_distance, precision, random_state, debug):
    """
    Execute the AutoNR algorithm. The algorithm will try different parametrizations of the NrKmeans algorithm
    and rate them by their Minimum Description Length. It will terminate if the parameters with the lowest MDL costs are
    found.
    :param X: input data
    :param nrkmeans_repetitions: number of NrKmeans executions to find the best local minimum
    :param outliers: Default False
    :param max_subspaces: maximum number of subspaces
    :param max_n_clusters: maximum number of clusters for a subspace_nr
    :param random_state: use a fixed random state to get a repeatable solution
    :param debug: print information regarding the execution process
    :return: NrKmeans object, MDL costs, List with all MDL costs
    """
    max_subspaces, max_n_clusters, max_distance, precision, random_state = _check_input_parameters(X, nrkmeans_repetitions,
                                                                          max_subspaces, max_n_clusters, max_distance,
                                                                            precision, random_state)
    all_mdl_costs = []
    # Default number of clusters
    n_clusters = [1]
    # Use first values as benchmark
    best_nrkmeans, best_mdl_overall, best_subspace_costs = _execute_nrkmeans(X, n_clusters, 1,
                                                                             random_state, outliers=outliers,
                                                                             debug=debug,
                                                                             mdl_for_noisespace=mdl_for_noisespace,
                                                                             max_distance=max_distance, precision=precision)
    all_mdl_costs.append(_Nrkmeans_Mdl_Costs(True, best_mdl_overall, NOISE_SPACE_SPLIT))
    better_found_since_merging = False
    # Begin algorithm. Repeat until no enhancement is found
    while len(n_clusters) < max_subspaces:
        if debug:
            print("==================================================")
            print("Start next iteration with: " + str(best_nrkmeans.n_clusters) + " / best costs: " + str(
                best_mdl_overall))
            print("==================================================")
        # Save if a better solution has been found in this iteration
        better_solution_found = False
        # Swap noise space to last position
        if best_nrkmeans.n_clusters[-1] == 1:
            order = list(reversed(np.argsort(best_subspace_costs[:-1])))
            order.append(len(best_nrkmeans.n_clusters) - 1)
        else:
            order = list(reversed(np.argsort(best_subspace_costs)))
        # Go through subspaces and try to improve mdl costs
        for subspace_nr in order:
            if debug:
                print("==================================================")
                print("Try splitting subspace_nr {0} with n_clusters = [{1}] and m = {2}. Costs = {3}".format(subspace_nr,
                                                                                                           best_nrkmeans.n_clusters[
                                                                                                               subspace_nr],
                                                                                                           best_nrkmeans.m[
                                                                                                               subspace_nr],
                                                                                                           best_subspace_costs[
                                                                                                               subspace_nr]))
            # Check if dimensionality is bigger than 1 else subspace_nr can not be splitted any more
            # Noise space can still be converted to cluster space
            split_cluster_count = best_nrkmeans.n_clusters[subspace_nr]
            if best_nrkmeans.m[subspace_nr] == 1 and best_nrkmeans.n_clusters[subspace_nr] > 1:
                continue
            # If there are more than one subspace_nr just search within the turned subspace_nr
            if len(best_nrkmeans.n_clusters) > 1:
                X_subspace = best_nrkmeans.transform_subspace(X, subspace_nr)
            else:
                X_subspace = X.copy()
            # Try to find more structure in the noise space
            if best_nrkmeans.n_clusters[subspace_nr] == 1:
                nrkmeans_split, mdl_total_split, mdl_threshold_split, subspace_costs_split = _split_noise_space(X_subspace, subspace_nr, best_nrkmeans, best_mdl_overall, best_subspace_costs, all_mdl_costs,
                        nrkmeans_repetitions, outliers, max_n_clusters, mdl_for_noisespace, max_distance, precision, random_state, debug)
            # Split existing cluster space
            else:
                nrkmeans_split, mdl_total_split, mdl_threshold_split, subspace_costs_split = _split_cluster_space(X_subspace, subspace_nr, best_nrkmeans, best_mdl_overall, best_subspace_costs, all_mdl_costs,
                         nrkmeans_repetitions, outliers, mdl_for_noisespace, max_distance, precision, random_state, debug)
            # ============================= FULL SPACE =====================================
            # Execute new found n_clusters for full space (except number of subspaces was 1)
            if len(best_nrkmeans.n_clusters) > 1 and mdl_threshold_split < best_subspace_costs[subspace_nr]:
                # Get parameters for full space execution
                n_clusters_full, centers_full, P_full, V_full = _get_full_space_parameters_split(X, best_nrkmeans,
                                                                                                 nrkmeans_split,
                                                                                                 subspace_nr)
                if debug:
                    print("==================================================")
                    print("Full space execution with n_clusters = {0}. Current best costs = {1}".format(n_clusters_full,
                                                                                                        best_mdl_overall))
                nrkmeans, mdl_cost, subspace_costs = _execute_nrkmeans(X,
                                                                       n_clusters_full,
                                                                       1,
                                                                       random_state,
                                                                       centers_full, V_full, P_full,
                                                                       outliers=outliers, debug=debug,
                                                                       mdl_for_noisespace=mdl_for_noisespace,
                                                                        max_distance=max_distance, precision=precision)
                all_mdl_costs.append(_Nrkmeans_Mdl_Costs(True, mdl_cost,
                                                         NOISE_SPACE_SPLIT if split_cluster_count else CLUSTER_SPACE_SPLIT))
                if mdl_cost < best_mdl_overall:
                    if debug:
                        print("!!! Better solution found !!!")
                    best_nrkmeans = nrkmeans
                    best_subspace_costs = subspace_costs
                    best_mdl_overall = mdl_cost
                    better_solution_found = True
                    better_found_since_merging = True
                    # Continue with next iteration
                    break
            # If number of subspaces was 1, check if total mdl split was smaller than best mdl overall
            elif len(best_nrkmeans.n_clusters) == 1 and mdl_total_split < best_mdl_overall:
                if debug:
                    print("!!! Better solution found !!!")
                best_nrkmeans = nrkmeans_split
                best_subspace_costs = subspace_costs_split
                best_mdl_overall = mdl_total_split
                better_solution_found = True
                better_found_since_merging = True
                # Continue with next iteration
                break
        # If better solution has been found, try to merge found subspaces
        if better_found_since_merging:
            if not better_solution_found or len(best_nrkmeans.n_clusters) == max_subspaces:
                best_nrkmeans, best_subspace_costs, best_mdl_overall, better_found_merge = _merge_spaces(X, best_nrkmeans,
                                                                                                   best_mdl_overall,
                                                                                                   best_subspace_costs,
                                                                                                   all_mdl_costs,
                                                                                                   max_n_clusters, outliers,
                                                                                                   random_state,
                                                                                                   mdl_for_noisespace,
                                                                                                   max_distance, precision,
                                                                                                   debug)
                better_found_since_merging = False
                # If merging did not improve the result or max number of subspaces is reached, end program
                if len(best_nrkmeans.n_clusters) == max_subspaces or not better_found_merge:
                    break
        else:
            break
    # Return best found nrkmeans, its mdl costs and list with all mdl costs
    return best_nrkmeans, best_mdl_overall, all_mdl_costs


def _check_input_parameters(X, nrkmeans_repetitions, max_subspaces, max_n_clusters, max_distance,  precision,
                            random_state):
    """
    Check the input parameters for AutoNR. This means that all input values which are None must be defined.
    :param X: input data
    :param nrkmeans_repetitions: number of NrKmeans executions to find the best local minimum
    :param max_subspaces: maximum number of subspaces
    :param max_n_clusters: maximum number of clusters for a subspace_nr
    :param random_state: use a fixed random state to get a repeatable solution
    :return: max_subspaces, max_n_clusters, random_state
    """
    random_state = check_random_state(random_state)
    # Check nrkmeans execution count
    if nrkmeans_repetitions is None or nrkmeans_repetitions < 1:
        raise ValueError(
            "NrKmeans execution count must be specified and larger than 0.\nYour input:\n" + str(
                nrkmeans_repetitions))
    # Check max subspaces
    if max_subspaces is None:
        max_subspaces = X.shape[1]
    if max_subspaces > X.shape[1]:
        raise ValueError(
            "Max subspaces can not be larger than the dimensionality of the data.\nYour max subspaces:\n" + str(
                max_subspaces))
    if max_subspaces < 2:
        raise ValueError(
            "Max subspaces must be at least 2.\nYour max subspaces:\n" + str(
                max_subspaces))
    # Check max number of clusters
    if max_n_clusters is None:
        max_n_clusters = X.shape[0]
    if max_n_clusters > X.shape[0]:
        raise ValueError(
            "Max number of clusters can not be larger than the number of data points.\nYour max number of clusters:\n" + str(
                max_n_clusters))
    if max_n_clusters < 2:
        raise ValueError(
            "Max number of clusters must be at least 2.\nYour max number of clusters:\n" + str(
                max_n_clusters))
    # Check dimensionality of input data
    if X.shape[1] < 2:
        raise ValueError("A minimum of 2 dimensions is needed. You input data has only 1 dimension.")
    if max_distance is None:
        max_distance = np.max(cdist(X, X))
    if precision is None:
        precision = nrkm._get_precision(X)
    return max_subspaces, max_n_clusters, max_distance, precision, random_state


def _execute_nrkmeans(X, n_clusters, nrkmeans_repetitions=10, random_state=None, centers=None, V=None, P=None,
                      outliers=False, mdl_for_noisespace=True, max_distance = None, precision=None, debug=False):
    """
    Execute NrKmeans multiple times to find the best result. If specified, multiple cores can be used for the execution.
    The method will return the best found NrKmeans result and its total MDL cost and every single subspace_nr cost.
    :param X: input data
    :param n_clusters: list containing number of clusters for each subspace_nr
    :param nrkmeans_repetitions: number of nrkmeans executions to find the best local minimum (default = 10)
    :param random_state: use a fixed random state to get a repeatable solution(optional)
    :param centers: list containing the cluster centers for each subspace_nr (optional
    :param V: orthogonal rotation matrix (optional)
    :param P: list containing projections for each subspace_nr (optional)
    :param outliers: 0 = No Outliers, 1 = Default encoding, 2 = Uniform distribution (default = 0)
    :param debug: default: False
    :return: best_nrkmeans, best_mdl_cost, subspace_costs
    """
    if debug:
        print("--------------------------------------------------")
        print(
            "[AutoNR] Next try with n_clusters = {0} ( {1} repetitions )".format(n_clusters, nrkmeans_repetitions))
    # Prepare parameters
    random_state = check_random_state(random_state)
    randoms = random_state.randint(0, 2 ** 31 - 1, nrkmeans_repetitions)
    best_nrkmeans = None
    best_total_mdl_costs = np.inf
    best_subspace_costs = np.inf
    add_random_executions = False
    if nrkmeans_repetitions > 1 and centers is not None and V is not None and P is not None:
        add_random_executions = True
    for i in range(nrkmeans_repetitions):
        if centers is not None and not (i > 0 and add_random_executions):
            input_centers = centers.copy()
        else:
            input_centers = None
        if V is not None and not (i > 0 and add_random_executions):
            input_V = V.copy()
        else:
            input_V = None
        if P is not None and not (i > 0 and add_random_executions):
            input_P = P.copy()
        else:
            input_P = None
        # Execute NrKmeans
        nrkmeans = nrkm.NrKmeans(n_clusters.copy(), random_state=randoms[i], input_centers=input_centers, V=input_V,
                                 P=input_P,
                                 outliers=outliers, mdl_for_noisespace=mdl_for_noisespace,
                                 max_distance=max_distance, precision=precision)
        try:
            nrkmeans.fit(X)
        except (Exception, ValueError) as err:
            print("Error occurred during NrKmeans execution: " + str(err))
            raise err
        # Get MDL Costs
        total_costs, _, all_subspace_costs = nrkmeans.calculate_mdl_costs(X)
        if total_costs < best_total_mdl_costs:
            best_total_mdl_costs = total_costs
            best_subspace_costs = all_subspace_costs
            best_nrkmeans = nrkmeans
    if debug:
        print("[AutoNR] Output n_clusters = {0} / m = {1}".format(best_nrkmeans.n_clusters, best_nrkmeans.m))
        print("[AutoNR] {0} ({1})".format(best_total_mdl_costs, best_subspace_costs))
    return best_nrkmeans, best_total_mdl_costs, best_subspace_costs


def _split_noise_space(X_subspace, subspace_nr, best_nrkmeans, best_mdl_overall, best_subspace_costs, all_mdl_costs,
                       nrkmeans_repetitions, outliers, max_n_clusters, mdl_for_noisespace, max_distance, precision,
                       random_state, debug):
    # Default parameters for the split
    mdl_threshold_split = np.inf
    mdl_total_split = None
    subspace_costs_split = None
    nrkmeans_split = None
    centers = None
    # If noise space stays the same, change strategy: only run once with the largest cluster splitted
    noise_stays_similar = False
    if best_nrkmeans.m[subspace_nr] > 1:
        n_clusters = [2, 1]
    else:
        n_clusters = [2]
    while n_clusters[0] <= max_n_clusters:
        nrkmeans, mdl_cost, subspace_costs = _execute_nrkmeans(X=X_subspace,
                                                               n_clusters=n_clusters,
                                                               nrkmeans_repetitions=1 if noise_stays_similar else nrkmeans_repetitions,
                                                               random_state=random_state,
                                                               centers=centers,
                                                               V=None if centers is None else nrkmeans.V,
                                                               P=None if centers is None else nrkmeans.P,
                                                               outliers=outliers, debug=debug,
                                                               mdl_for_noisespace=mdl_for_noisespace,
                                                               max_distance=max_distance, precision=precision)
        sum_subspace_costs = np.sum(subspace_costs)
        all_mdl_costs.append(_Nrkmeans_Mdl_Costs(len(best_nrkmeans.n_clusters) == 1,
                                                 best_mdl_overall - best_subspace_costs[
                                                     subspace_nr] + sum_subspace_costs,
                                                 NOISE_SPACE_SPLIT))
        if sum_subspace_costs < mdl_threshold_split:
            # Check if noise space stays nearly the same
            if nrkmeans_split is not None and abs(
                    subspace_costs[-1] - subspace_costs_split[-1]) < abs(
                subspace_costs[-1]) * SIMILARITY_THRESHOLD and nrkmeans.m[-1] == nrkmeans_split.m[-1]:
                noise_stays_similar = True
            # Save new values
            nrkmeans_split = nrkmeans
            mdl_threshold_split = sum_subspace_costs
            mdl_total_split = mdl_cost
            subspace_costs_split = subspace_costs
        else:
            break
        # Copy n_clusters from nrkmeans result in case cluster has been lost
        if nrkmeans.have_clusters_been_lost():
            n_clusters = nrkmeans.n_clusters.copy()
        # Continue searching with noise
        if nrkmeans.have_subspaces_been_lost():
            n_clusters.append(1)
            noise_stays_similar = False
            centers = None
        else:
            # Split cluster with largest variance
            if len(nrkmeans.n_clusters) == 2:
                centers = [
                    _split_largest_cluster(nrkmeans.V, nrkmeans.m[0], nrkmeans.P[0],
                                           nrkmeans.cluster_centers_[0],
                                           nrkmeans.scatter_matrices_[0],
                                           nrkmeans.labels_[:, 0]),
                    nrkmeans.cluster_centers_[1]]
            else:
                centers = [
                    _split_largest_cluster(nrkmeans.V, nrkmeans.m[0], nrkmeans.P[0],
                                           nrkmeans.cluster_centers_[0],
                                           nrkmeans.scatter_matrices_[0],
                                           nrkmeans.labels_[:, 0])]
        n_clusters[0] += 1
    return nrkmeans_split, mdl_total_split, mdl_threshold_split, subspace_costs_split


def _split_cluster_space(X_subspace, subspace_nr, best_nrkmeans, best_mdl_overall, best_subspace_costs, all_mdl_costs,
                         nrkmeans_repetitions, outliers, mdl_for_noisespace, max_distance, precision,
                         random_state, debug):
    split_cluster_count = best_nrkmeans.n_clusters[subspace_nr]
    # Default parameters for the split
    mdl_threshold_split = np.inf
    nrkmeans_split = None
    mdl_total_split = None
    n_clusters = [split_cluster_count, split_cluster_count]
    single_change_index = None
    V_split = None
    P_split = None
    centers = None
    while True:
        if n_clusters[0] * n_clusters[1] < split_cluster_count or 1 in n_clusters:
            if single_change_index is not None or nrkmeans_split.have_subspaces_been_lost():
                break
            else:
                mdl_cost_1 = np.inf
                mdl_cost_2 = np.inf
                # First test
                n_clusters = nrkmeans_split.n_clusters.copy()
                n_clusters[0] -= 1
                centers = [_merge_nearest_centers(nrkmeans_split.cluster_centers_[0]),
                           nrkmeans_split.cluster_centers_[1]]
                if n_clusters[0] * n_clusters[1] >= split_cluster_count and not 1 in n_clusters:
                    nrkmeans_1, mdl_cost_1, subspace_costs_1 = _execute_nrkmeans(X_subspace, n_clusters,
                                                                                 nrkmeans_repetitions if centers is None else 1,
                                                                                 random_state,
                                                                                 centers,
                                                                                 None if nrkmeans_split is None else nrkmeans_split.V,
                                                                                 None if nrkmeans_split is None else nrkmeans_split.P,
                                                                                 outliers=outliers,
                                                                                 debug=debug,
                                                                                 mdl_for_noisespace=mdl_for_noisespace,
                                                                                    max_distance=max_distance, precision=precision)
                    sum_subspace_costs_1 = np.sum(subspace_costs_1)
                    all_mdl_costs.append(_Nrkmeans_Mdl_Costs(len(best_nrkmeans.n_clusters) == 1,
                                                             best_mdl_overall - best_subspace_costs[
                                                                 subspace_nr] + sum_subspace_costs_1,
                                                             CLUSTER_SPACE_SPLIT))
                # Second test
                n_clusters = nrkmeans_split.n_clusters.copy()
                n_clusters[1] -= 1
                centers = [nrkmeans_split.cluster_centers_[0],
                           _merge_nearest_centers(nrkmeans_split.cluster_centers_[1])]
                if n_clusters[0] * n_clusters[1] >= split_cluster_count and not 1 in n_clusters:
                    nrkmeans_2, mdl_cost_2, subspace_costs_2 = _execute_nrkmeans(X_subspace, n_clusters,
                                                                                 nrkmeans_repetitions if centers is None else 1,
                                                                                 random_state,
                                                                                 centers,
                                                                                 None if nrkmeans_split is None else nrkmeans_split.V,
                                                                                 None if nrkmeans_split is None else nrkmeans_split.P,
                                                                                 outliers=outliers,
                                                                                 debug=debug,
                                                                                 mdl_for_noisespace=mdl_for_noisespace,
                                                                             max_distance=max_distance, precision=precision)
                    sum_subspace_costs_2 = np.sum(subspace_costs_2)
                    all_mdl_costs.append(_Nrkmeans_Mdl_Costs(len(best_nrkmeans.n_clusters) == 1,
                                                             best_mdl_overall - best_subspace_costs[
                                                                 subspace_nr] + sum_subspace_costs_2,
                                                             CLUSTER_SPACE_SPLIT))
                # Evaluation of the two tests
                if mdl_cost_1 == np.inf and mdl_cost_2 == np.inf:
                    break
                elif mdl_cost_1 < mdl_cost_2:
                    nrkmeans = nrkmeans_1
                    single_change_index = 0
                    n_clusters = nrkmeans_split.n_clusters.copy()
                    if sum_subspace_costs_1 < mdl_threshold_split:
                        nrkmeans_split = nrkmeans_1
                        mdl_threshold_split = sum_subspace_costs_1
                        mdl_total_split = mdl_cost_1
                        subspace_costs_split = subspace_costs_1
                else:
                    nrkmeans = nrkmeans_2
                    single_change_index = 1
                    n_clusters = nrkmeans_split.n_clusters.copy()
                    if sum_subspace_costs_2 < mdl_threshold_split:
                        nrkmeans_split = nrkmeans_2
                        mdl_threshold_split = sum_subspace_costs_2
                        mdl_total_split = mdl_cost_2
                        subspace_costs_split = subspace_costs_2
                # Set number of clusters
                n_clusters[single_change_index] -= 1
        else:
            nrkmeans, mdl_cost, subspace_costs = _execute_nrkmeans(X_subspace, n_clusters,
                                                                   nrkmeans_repetitions if centers is None else 1,
                                                                   random_state,
                                                                   centers,
                                                                   V_split,
                                                                   P_split,
                                                                   outliers=outliers, debug=debug,
                                                                   mdl_for_noisespace=mdl_for_noisespace,
                                                                    max_distance=max_distance, precision=precision)
            sum_subspace_costs = np.sum(subspace_costs)
            all_mdl_costs.append(_Nrkmeans_Mdl_Costs(len(best_nrkmeans.n_clusters) == 1,
                                                     best_mdl_overall - best_subspace_costs[
                                                         subspace_nr] + sum_subspace_costs,
                                                     CLUSTER_SPACE_SPLIT))
            if sum_subspace_costs < mdl_threshold_split:
                nrkmeans_split = nrkmeans
                mdl_threshold_split = sum_subspace_costs
                mdl_total_split = mdl_cost
                subspace_costs_split = subspace_costs
            elif single_change_index is not None:
                n_clusters = [1,1]
                continue
        # Prepare values for next iteration
        if nrkmeans.have_clusters_been_lost() or nrkmeans.have_subspaces_been_lost():
            centers = None
            V_split = None
            P_split = None
        else:
            centers = nrkmeans.cluster_centers_
            V_split = nrkmeans.V
            P_split = nrkmeans.P
        # Change both subspaces
        if single_change_index is None:
            n_clusters[0] -= 1
            n_clusters[1] -= 1
            if centers is not None:
                centers = [_merge_nearest_centers(nrkmeans.cluster_centers_[0]),
                           _merge_nearest_centers(nrkmeans.cluster_centers_[1])]
        else:
            n_clusters[single_change_index] -= 1
            if centers is not None:
                if single_change_index == 0:
                    centers = [_merge_nearest_centers(nrkmeans.cluster_centers_[0]),
                               nrkmeans.cluster_centers_[1]]
                else:
                    centers = [nrkmeans.cluster_centers_[0],
                               _merge_nearest_centers(nrkmeans.cluster_centers_[1])]
    return nrkmeans_split, mdl_total_split, mdl_threshold_split, subspace_costs_split


def _merge_spaces(X, best_nrkmeans, best_mdl_overall, best_subspace_costs, all_mdl_costs, max_n_clusters,
                  outliers, random_state, mdl_for_noisespace, max_distance, precision, debug):
    """
    Merge two subspaces and check if Mdl costs decreases. Repeat merge. If no enhancement is found or only one subspace_nr
    (noise space excluded) is left, break.
    :param X: input data
    :param best_nrkmeans: best found NrKmeans until now
    :param best_mdl_overall: mdl costs of best found NrKmeans
    :param best_subspace_costs: single subspace_nr mdl costs of best found NrKmeans
    :param all_mdl_costs: list to save mdl costs of each NrKmeans execution
    :param max_n_clusters: maximum number of clusters for a subspace_nr
    :param outliers: 0 = No Outliers, 1 = Default encoding, 2 = Uniform distribution
    :param n_used_cores: number of cores used for NrKmeans execution
    :param random_state: use a fixed random state to get a repeatable solution
    :param debug: print information regarding the execution process
    :return: best_nrkmeans, best_subspace_costs, best_mdl_overall
    """
    better_found = False
    # As long as a improvement was made, repeat
    while True:
        # Get number of subspaces (noise space excluded). Must be larger than 1 to be able to execute a merge
        if len([x for x in best_nrkmeans.n_clusters if x > 1]) > 1:
            if debug:
                print("==================================================")
                print("Start merging")
                print("==================================================")
            best_nrkmeans_iteration = None
            best_subspace_costs_iteration = None
            # Go through each combination of subspaces
            for i in range(len(best_nrkmeans.n_clusters) - 1):
                for j in range(i + 1, len(best_nrkmeans.n_clusters)):
                    # Skip noise space
                    if best_nrkmeans.n_clusters[j] == 1:
                        continue
                    if debug:
                        print("==================================================")
                        print(
                            "Try merging subspace_nr {0} with n_clusters = [{1}] and subspace_nr {2} with n_clusters = [{3}]. Combined costs = {4}".format(
                                i, best_nrkmeans.n_clusters[i], j, best_nrkmeans.n_clusters[j],
                                best_subspace_costs[i] + best_subspace_costs[j]
                            ))
                    # If there are more than two subspaces just search within the turned subspaces
                    if len(best_nrkmeans.n_clusters) > 2:
                        X_subspace = np.c_[
                            best_nrkmeans.transform_subspace(X, i), best_nrkmeans.transform_subspace(X, j)]
                    else:
                        X_subspace = X.copy()
                    # Get default subspace_nr costs threshold and V as identity matrix
                    nrkmeans_merge = None
                    mdl_threshold_merge = np.inf
                    # Rotation stays the same for single subspace_nr
                    V = np.identity(X_subspace.shape[1])
                    # Create every combination of centers for the two subspaces
                    centers = []
                    turned_centers_1 = np.matmul(best_nrkmeans.cluster_centers_[i], best_nrkmeans.V)[:,
                                       best_nrkmeans.P[i]]
                    turned_centers_2 = np.matmul(best_nrkmeans.cluster_centers_[j], best_nrkmeans.V)[:,
                                       best_nrkmeans.P[j]]
                    for center_1 in turned_centers_1:
                        for center_2 in turned_centers_2:
                            centers.append(np.append(center_1, center_2))
                    centers = [centers]
                    # Try decreasing number of clusters within merged subspaces
                    for n in reversed(range(max(best_nrkmeans.n_clusters[i], best_nrkmeans.n_clusters[j]),
                                            best_nrkmeans.n_clusters[i] * best_nrkmeans.n_clusters[j] + 1)):
                        # Skip if n is larger than the maximum amount of clusters
                        if n > max_n_clusters:
                            centers = [_merge_nearest_centers(centers[0])]
                            continue
                        # In case clusters have been lost, n_clusters and centers can diverge
                        if nrkmeans_merge is not None and len(centers[0]) < n:
                            continue
                        nrkmeans, mdl_cost, subspace_costs = _execute_nrkmeans(X_subspace, [n], 1,
                                                                               random_state,
                                                                               centers, V,
                                                                               outliers=outliers,
                                                                               debug=debug,
                                                                               mdl_for_noisespace=mdl_for_noisespace,
                                                                             max_distance=max_distance, precision=precision)
                        sum_subspace_costs = subspace_costs[0]
                        all_mdl_costs.append(_Nrkmeans_Mdl_Costs(len(best_nrkmeans.n_clusters) == 2,
                                                                 best_mdl_overall - best_subspace_costs[i] -
                                                                 best_subspace_costs[
                                                                     j] + sum_subspace_costs, CLUSTER_SPACE_MERGE))
                        if sum_subspace_costs < mdl_threshold_merge:
                            # Save new values
                            nrkmeans_merge = nrkmeans
                            mdl_threshold_merge = sum_subspace_costs
                            mdl_total_merge = mdl_cost
                            subspace_costs_merge = subspace_costs
                            # Prepare values for next iteration
                            centers = [_merge_nearest_centers(nrkmeans_merge.cluster_centers_[0])]
                        else:
                            # pass
                            break
                    # ============================= FULL SPACE =====================================
                    # Execute new found n_clusters for full space (except number of subspaces was 2)
                    if len(best_nrkmeans.n_clusters) > 2 and mdl_threshold_merge < best_subspace_costs[i] + \
                            best_subspace_costs[j]:
                        # Get parameters for full space execution
                        n_clusters_full, centers_full, P_full, V_full = _get_full_space_parameters_merge(X,
                                                                                                         best_nrkmeans,
                                                                                                         nrkmeans_merge,
                                                                                                         i, j)
                        if debug:
                            print("==================================================")
                            print("Next full try with: " + str(n_clusters_full))
                        nrkmeans, mdl_cost, subspace_costs = _execute_nrkmeans(X, n_clusters_full, 1,
                                                                               random_state,
                                                                               centers_full, V_full, P_full,
                                                                               outliers=outliers,
                                                                               debug=debug,
                                                                               mdl_for_noisespace=mdl_for_noisespace,
                                                                             max_distance=max_distance, precision=precision)
                        all_mdl_costs.append(_Nrkmeans_Mdl_Costs(True, mdl_cost, CLUSTER_SPACE_MERGE))
                        if mdl_cost < best_mdl_overall:
                            if debug:
                                print("!!! Better solution found !!!")
                            best_nrkmeans_iteration = nrkmeans
                            best_subspace_costs_iteration = subspace_costs
                            best_mdl_overall = mdl_cost
                    # If number of subspaces was 2, check if total mdl merge was smaller than best mdl overall
                    elif len(best_nrkmeans.n_clusters) <= 2 and mdl_total_merge < best_mdl_overall:
                        if debug:
                            print("!!! Better solution found !!!")
                        best_nrkmeans_iteration = nrkmeans_merge
                        best_subspace_costs_iteration = subspace_costs_merge
                        best_mdl_overall = mdl_total_merge
            # Overwrite best NrKmeans with best possible merge
            if best_nrkmeans_iteration is not None:
                better_found = True
                best_nrkmeans = best_nrkmeans_iteration
                best_subspace_costs = best_subspace_costs_iteration
            # If no better NrKmeans was found break
            else:
                break
        # If there is only one subspace_nr is left (noise space excluded) merge is not possible. Break
        else:
            break
    return best_nrkmeans, best_subspace_costs, best_mdl_overall, better_found


def _split_largest_cluster(V, m_subspace, P_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace):
    """
    Split the cluster with the highest variance into two clusters. The new centers will be calculated
    with following formula: old center +- cluster variance
    :param V: orthogonal rotation matrix
    :param m_subspace: dimensionality of the subspace_nr
    :param P_subspace: projecitons of the subspace_nr
    :param centers_subspace: cluster centers of the subspace_nr
    :param scatter_matrices_subspace: scatter matrices of the subspace_nr
    :param labels_subspace: cluster assignments of the subspace_nr
    :return: new centers of the subspace_nr
    """
    # Get number of points for each cluster
    n_points = [len(labels_subspace[labels_subspace == i]) for i in range(len(centers_subspace))]
    # Index of the cluster with the largest variance
    cropped_V = V[:, P_subspace]
    turned_scatter = np.matmul(np.matmul(cropped_V.transpose(), scatter_matrices_subspace),
                               cropped_V)
    variances = [np.trace(s) / n_points[i] / m_subspace for i, s in enumerate(turned_scatter)]
    max_variances_id = np.argmax(variances)
    # For first center add variance, for second remove variance
    push_factor = scatter_matrices_subspace[max_variances_id].diagonal() / n_points[max_variances_id] / m_subspace
    new_center_1 = centers_subspace[max_variances_id] + push_factor
    new_center_2 = centers_subspace[max_variances_id] - push_factor
    # Add newly created centers
    centers_subspace = np.append(centers_subspace, [new_center_1], axis=0)
    centers_subspace = np.append(centers_subspace, [new_center_2], axis=0)
    # Remove old center
    centers_subspace = np.delete(centers_subspace, max_variances_id, axis=0)
    return centers_subspace


def _merge_nearest_centers(centers_subspace):
    """
    Merge the two nearest centers of a subspace_nr.
    :param centers_subspace: cluster centers of the subspace_nr
    :return: new centers of the subspace_nr
    """
    # Get indices of the closest centers
    i, j = _find_two_closest_centers(centers_subspace)
    new_center = (centers_subspace[i] + centers_subspace[j]) / 2
    # Delete merged centers from list
    centers = np.delete(centers_subspace, [i, j], axis=0)
    # Add new created center
    centers = np.append(centers, [new_center], axis=0)
    return centers


def _find_two_closest_centers(centers_subspace):
    """
    Identify the indices of the two nearest clusters of a subspace_nr.
    :param centers_subspace: cluster centers of the subspace_nr
    :return: index_1, index_2
    """
    # Get pairwise distances
    distances = pdist(centers_subspace, "euclidean")
    # Convert condensed distance matrix to squared matrix
    distances_squared = squareform(distances)
    # Fill distances to themselves with infinity
    np.fill_diagonal(distances_squared, np.inf)
    # Get minimum indices
    index_1, index_2 = np.unravel_index(distances_squared.argmin(), distances_squared.shape)
    return index_1, index_2


def _get_full_space_parameters_split(X, best_nrkmeans, nrkmeans_split, subspace):
    """
    Get parameters from the subspace_nr split for the next full space NrKmeans execution.
    :param best_nrkmeans: NrKmeans result form last iteration
    :param nrkmeans_split: best found NrKmeans result from subspace_nr split
    :param subspace: subspace_nr index of the splitted space
    :return: n_clusters_new, centers_new, P_new, V_new
    """
    # Remove the splitted cluster count and add the new ones
    n_clusters_new = best_nrkmeans.n_clusters.copy()
    del n_clusters_new[subspace]
    n_clusters_new += nrkmeans_split.n_clusters
    n_clusters_new = np.array(n_clusters_new)
    # Remove centers from splitted subspace_nr and add the new ones reversed turned
    centers_new = best_nrkmeans.cluster_centers_.copy()
    del centers_new[subspace]
    centers_from_subspace = [
        [np.mean(X[nrkmeans_split.labels_[:, i] == j], axis=0) for j in range(nrkmeans_split.n_clusters[i])] for i in
        range(nrkmeans_split.labels_.shape[1])]
    centers_new += centers_from_subspace
    centers_new = np.array(centers_new)
    # Update the rotation matrix with the rotation from the splitted subspace_nr
    V_F = nrkm._create_full_rotation_matrix(best_nrkmeans.V.shape[0], best_nrkmeans.P[subspace], nrkmeans_split.V)
    V_new = np.matmul(best_nrkmeans.V, V_F)
    # Update the projecitons by replacing the projections from the new subspaces with the ones from the original space
    P_new = best_nrkmeans.P.copy()
    P_from_subspace = [np.array([P_new[subspace][i] for i in p]) for p in nrkmeans_split.P]
    del P_new[subspace]
    P_new += P_from_subspace
    P_new = np.array(P_new)
    # Order the subspaces with the one with the most clusters first
    order = np.argsort(-n_clusters_new)
    n_clusters_new = list(n_clusters_new[order])
    centers_new = list(centers_new[order])
    P_new = list(P_new[order])
    # Remove possible double noise space
    n_clusters_new, centers_new, P_new = remove_double_noise_space(n_clusters_new, centers_new, P_new)
    # Return parameters for the full space execution
    return n_clusters_new, centers_new, P_new, V_new


def _get_full_space_parameters_merge(X, best_nrkmeans, nrkmeans_merge, subspace_1, subspace_2):
    """
    Get parameters from the subspace_nr merge for the next full space NrKmeans execution.
    :param best_nrkmeans: NrKmeans result form last iteration
    :param nrkmeans_merge: best found NrKmeans result from subspace_nr merge
    :param subspace_1: first subspace_nr index of the merged spaces
    :param subspace_2: second subspace_nr index of the merged spaces
    :return: n_clusters_new, centers_new, P_new, V_new
    """
    # Remove the merged cluster counts and add the new one
    n_clusters_new = best_nrkmeans.n_clusters.copy()
    del n_clusters_new[subspace_2]
    del n_clusters_new[subspace_1]
    n_clusters_new += nrkmeans_merge.n_clusters
    n_clusters_new = np.array(n_clusters_new)
    # Update the projecitons by replacing the projections from the new subspace_nr with the ones from the original spaces
    P_new = best_nrkmeans.P.copy()
    P_from_subspace = [np.append(P_new[subspace_1], P_new[subspace_2])]
    del P_new[subspace_2]
    del P_new[subspace_1]
    P_new += P_from_subspace
    P_new = np.array(P_new)
    # Remove centers from splitted subspace_nr and add the new ones reversed turned
    centers_new = best_nrkmeans.cluster_centers_.copy()
    del centers_new[subspace_2]
    del centers_new[subspace_1]
    centers_from_subspace = [
        [np.mean(X[nrkmeans_merge.labels_[:, i] == j], axis=0) for j in range(nrkmeans_merge.n_clusters[i])] for i in
        range(nrkmeans_merge.labels_.shape[1])]
    centers_new += centers_from_subspace
    centers_new = np.array(centers_new)
    # Order the subspaces with the one with the most clusters first
    order = np.argsort(-n_clusters_new)
    n_clusters_new = list(n_clusters_new[order])
    centers_new = list(centers_new[order])
    P_new = list(P_new[order])
    # Remove possible double noise space
    n_clusters_new, centers_new, P_new = remove_double_noise_space(n_clusters_new, centers_new, P_new)
    # Return parameters for the full space execution
    return n_clusters_new, centers_new, P_new, best_nrkmeans.V


def remove_double_noise_space(n_clusters, centers, P):
    while n_clusters[-2] == 1:
        P[-2] = np.append(P[-1], P[-2])
        del P[-1]
        del centers[-1]
        del n_clusters[-1]
    return n_clusters, centers, P


class _Nrkmeans_Mdl_Costs():

    def __init__(self, full_space_execution, costs, type):
        if type not in [CLUSTER_SPACE_SPLIT, NOISE_SPACE_SPLIT, CLUSTER_SPACE_MERGE]:
            raise ValueError("Type must be CLUSTER_SPACE_SPLIT, NOISE_SPACE_SPLIT or CLUSTER_SPACE_MERGE")
        self.full_space_execution = full_space_execution
        self.costs = costs
        self.type = type

    def get_line_color(self):
        if self.type == NOISE_SPACE_SPLIT:
            return "brown"
        elif self.type == CLUSTER_SPACE_SPLIT:
            return "orange"
        elif self.type == CLUSTER_SPACE_MERGE:
            return "magenta"


class AutoNR(BaseEstimator, ClusterMixin):

    def __init__(self, nrkmeans_repetitions=15, outliers=True, max_subspaces=None,
                 max_n_clusters=None, mdl_for_noisespace=True, max_distance=None, precision=None,
                 random_state=None, debug=False):
        # Fixed attributes
        self.nrkmeans_repetitions = nrkmeans_repetitions
        self.outliers = outliers
        self.max_subspaces = max_subspaces
        self.max_n_clusters = max_n_clusters
        self.mdl_for_noisespace = mdl_for_noisespace
        self.max_distance = max_distance
        self.precision = precision
        self.random_state = random_state
        self.debug = debug

    def fit(self, X, y=None):
        """
                Cluster the input dataset with the AutoNR algorithm. This means the best number of subspaces and clusters
                is searched as input for the NrKmeans algorithm. Saves the best NrKmeans result and its MDL costs.
                in the AutoNR object.
                :param X: input data
                :return: the AutoNR object
                """
        nrkmeans, mdl_costs, all_mdl_costs = _autonr(X, self.nrkmeans_repetitions, self.outliers,
                                                     self.max_subspaces,
                                                     self.max_n_clusters, self.mdl_for_noisespace,
                                                     self.max_distance, self.precision,
                                                     self.random_state, self.debug)
        # Output
        self.nrkmeans = nrkmeans
        self.mdl_costs = mdl_costs
        self.all_mdl_costs = all_mdl_costs
        return self

    def plot_mdl_progress(self):
        """
        Plot the progress of the MDL costs during AutoNR. Dots represent full space NrKmeans executions.
        Best found result is displayed as a green dot.
        BEWARE: If the green dot is not the lowest point of the curve, the estimated cost of a split/merge was lower
        than the final costs of a full space NrKmeans execution!
        :return:
        """
        if self.nrkmeans is None:
            raise Exception("The AutoNR algorithm has not run yet. Use the fit() function first.")
        # Plot line with all costs
        mdl_costs = np.array([nrkmeans_mdl.costs for nrkmeans_mdl in self.all_mdl_costs])
        fig, ax = plt.subplots()
        for i in range(len(mdl_costs) - 1):
            line_color = self.all_mdl_costs[i + 1].get_line_color()
            ax.plot((i, i + 1), (mdl_costs[i], mdl_costs[i + 1]), color=line_color, zorder=1)
        # Plot circle for each full execution (leave out best solution)
        full_executions = np.array(
            [i for i, nrkmeans_mdl in enumerate(self.all_mdl_costs) if
             nrkmeans_mdl.full_space_execution])  # indices of full executions
        min_index_full_executions = np.argmin(mdl_costs[full_executions])  # index of min value in full_executions array
        min_index_complete = full_executions[min_index_full_executions]  # index of min value in complete dataset
        full_executions_wo_best = np.delete(full_executions, min_index_full_executions)
        plt.scatter(full_executions_wo_best, mdl_costs[full_executions_wo_best], alpha=0.7, color="blue",
                    marker="o",
                    zorder=2)
        # Plot best full solution as green circle
        plt.scatter(min_index_complete, mdl_costs[min_index_complete], alpha=0.7, color="green", marker="o",
                    zorder=2)
        # Add labels
        plt.xlabel("NrKmeans try")
        plt.ylabel("MDL costs")
        # Add legend
        legend_elements = [Line2D([0], [0], color="brown", lw=2, label="Noise space split"),
                           Line2D([0], [0], color="orange", lw=2, label="Cluster space split"),
                           Line2D([0], [0], color="magenta", lw=2, label="Cluster space merge"),
                           Line2D([0], [0], marker="o", color="blue", markersize=7,
                                  label="Full space run"),
                           Line2D([0], [0], marker="o", color="green", markersize=7, label="Best result")]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.show()

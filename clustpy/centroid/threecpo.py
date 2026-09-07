import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus as kpp
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt
from scipy.special import xlogy
from clustpy.utils.checks import check_random_state, check_parameters
from clustpy.utils._information_theory import integer_costs
from sklearn.utils.validation import check_is_fitted
from pathlib import Path

"""
=============================
Distance Function and Seeding
=============================
"""


def poisson_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate the Poisson-based distances between rows in X and rows in Y.

    Parameters
    ----------
    X : np.ndarray
        The first set of rows
    y : np.ndarray
        The second set of rows

    Returns
    -------
    dist : float
        The distances between the rows
    """
    if X.ndim == 1:
        X_re = X.reshape(1, -1)
    else:
        X_re = X
    if Y.ndim == 1:
        Y_re = Y.reshape(1, -1)
    else:
        Y_re = Y
    X_row_sums = X_re.sum(1)
    Y_row_sums = Y_re.sum(1)
    combined_N = X_row_sums.reshape(-1, 1) + Y_row_sums.reshape(1, -1)
    X_part = xlogy(X_re, X_re).sum(1) - xlogy(X_row_sums, X_row_sums)
    Y_part = xlogy(Y_re, Y_re).sum(1) - xlogy(Y_row_sums, Y_row_sums)
    combined_part = xlogy(combined_N, combined_N)
    # proecss the last part in chunks to avoid memory issues due to a huge n,m,d matrix
    for j in range(Y_re.shape[0]):
        column_sum_x_j = X_re + Y_re[j]
        combined_part[:, j] -= xlogy(column_sum_x_j, column_sum_x_j).sum(1)
    dist = X_part.reshape(-1, 1) + Y_part.reshape(1, -1) + combined_part
    dist = np.maximum(0, dist) # consider numerical issues
    assert dist.shape == (X_re.shape[0], Y_re.shape[0]), f"Shape is {dist.shape}"
    if X.ndim == 1 and Y.ndim == 1:
        dist = dist[0, 0]
    elif Y.ndim == 1:
        dist = dist[:, 0]
    return dist


def poisson_seeding(X: np.ndarray, n_clusters: int, alpha: float = 1., subset_size: int | str | None = None, 
                    n_local_trials: int | None = None, random_state: np.random.RandomState | int | None = None) -> np.ndarray:
    """
    Apply a k-means++ like seeding based on the poisson dist to select rows.

    Parameters
    ----------
    X : np.ndarray
        The data matrix
    n_clusters : int
        The number of clusters
    alpha : float
        Exponential used for the distances (default: 1.)
    subset_size : int | str | None
        The size of the subset used for seeding. Can be 'auto' to select 100 * n_clusters samples.
        If None, the whole data set will be used (default: None)
    n_local_trials : int | None
        The number of local trials used when choosing a new centroid. If None, it will be equal to 2 + int(np.log(n_clusters)) (default: None)
    random_state : np.random.RandomState | int | None
        The random state

    Returns
    -------
    selected_rows : np.ndarray
        The selected rows
    """
    if n_local_trials is None:
        # Strategy taken from sklearn
        n_local_trials = 2 + int(np.log(n_clusters))
    n_local_trials_final = min(n_local_trials, X.shape[0])
    random_state = check_random_state(random_state)
    if subset_size is not None:
        if subset_size == "auto":
            subset_size = 100 * n_clusters
        assert isinstance(subset_size, int), "subset_size has to be an int"
        if X.shape[0] > subset_size:
            ids = random_state.choice(X.shape[0], size=subset_size, replace=False)
            X = X[ids]
    selected_rows = np.zeros((n_clusters, X.shape[1]))
    # Get initial row randomly
    init_row_id = random_state.randint(X.shape[0])
    selected_rows[0] = X[init_row_id]
    distance_closest_row = poisson_dist(X, selected_rows[0])
    # Add new rows
    for i in range(1, n_clusters):
        distance_closest_row_adj = np.nan_to_num(distance_closest_row, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha != 1.:
            distance_closest_row_adj = distance_closest_row_adj ** alpha
        denominator = distance_closest_row_adj.sum()
        probs = distance_closest_row_adj / denominator if denominator > 0 else None
        new_row_ids = random_state.choice(X.shape[0], size=n_local_trials_final, p=probs, replace=False)
        if n_local_trials_final > 1:
            distances_to_new_rows = poisson_dist(X, X[new_row_ids])
            distance_closest_row_tmp = np.minimum(distance_closest_row.reshape((-1, 1)), distances_to_new_rows)
            best_id = distance_closest_row_tmp.sum(0).argmin()
            distance_closest_row = distance_closest_row_tmp[:, best_id]
            best_new_row_id = new_row_ids[best_id]
        else:
            best_new_row_id = new_row_ids[0]
            distances_to_new_row = poisson_dist(X, X[best_new_row_id])
            distance_closest_row = np.minimum(distance_closest_row, distances_to_new_row)
        selected_rows[i] = X[best_new_row_id]
    return selected_rows


def _poisson_labels_through_centroids(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Obtain the labels of X based on a set of centers.

    Parameters
    ----------
    X : np.ndarray
        The data matrix
    centers : np.ndarray
        The rows that act as cluster centers

    Returns
    -------
    labels : np.ndarray
        The cluster labels
    """
    all_column_lambdas = centers / centers.sum(1).reshape((-1, 1))
    row_lambdas = X.sum(1)
    probs = get_log_probs_poisson(X, row_lambdas, all_column_lambdas, "columns")
    labels = np.argmax(probs, axis=1)
    return labels


def initial_poisson_clustering_labels(X: np.ndarray, n_clusters: int, init_strat: str,
                                      random_state: np.random.RandomState | int | None) -> np.ndarray:
    """
    Initialize the labels for Poisson-based clustering.
    The initialization strategy can be:
    - labels obtained uniformly at random (random)
    - labels obtained by a random set of initial center used for assignments with the poisson distance (random-centers-dist)
    - labels obtained by a random set of initial center used for assignments with the poisson log probability (random-centers)
    - labels obtained by a full kmeans run (kmeans)
    - labels obtained by kmeans++ seeding (kmeans++)
    - labels obtained by normalizing the rows considering relative frequencies followed by a full kmaens run (rf+kmeans)
    - labels obtained by normalizing the rows considering relative frequencies followed by kmeans++ seeding (rf+kmeans++)
    - labels obtained by kmeans++ seeding using the poisson distance used for assignments with the poisson distance (poisson++-dist)
    - labels obtained by kmeans++ seeding using the poisson distance used for assignments with the poisson log probability (poisson++)

    Parameters
    ----------
    X : np.ndarray
        The data matrix
    n_clusters : int
        The number of clusters
    init_strat : str
        The initialization strategy. Can be 'random', 'random-centers', 'random-centers-dist', 'kmeans', 'kmeans++', 'rf+kmeans', 'rf+kmeans++', 'poisson++', or 'poisson++-dist'
    random_state : np.random.RandomState | int | None
        The random state

    Returns
    -------
    labels : np.ndarray
        The initial cluster labels
    """
    assert init_strat in ["random", "random-centers", "random-centers-dist", "kmeans", "kmeans++", "rf+kmeans", "rf+kmeans++", "poisson++", "poisson++-dist"]
    random_state = check_random_state(random_state)
    if init_strat.startswith("rf+"):
        # Normalize rows by relative frequencies
        row_sums = X.sum(1).reshape((-1, 1))
        X = X / row_sums
    if init_strat == "random":
        labels = np.zeros(X.shape[0], dtype=np.int32)
        while np.unique(labels).shape[0] != n_clusters:
            new_labels = random_state.randint(0, n_clusters, size=X.shape[0])
            assert isinstance(new_labels, np.ndarray)
            labels = new_labels
    elif init_strat in ["random-centers", "random-centers-dist"]:
        center_row_ids = random_state.choice(X.shape[0], size=n_clusters, replace=False)
        if init_strat == "random-centers":
            labels = _poisson_labels_through_centroids(X, X[center_row_ids])
        else:
            labels = poisson_dist(X, X[center_row_ids]).argmin(1)
    elif init_strat in ["kmeans", "rf+kmeans"]:
        km = KMeans(n_clusters, random_state=random_state)
        km.fit(X)
        labels = km.labels_
    elif init_strat in ["kmeans++", "rf+kmeans++"]:
        centers_kpp, _ = kpp(X, n_clusters, random_state=random_state)
        labels, _ = pairwise_distances_argmin_min(X=X, Y=centers_kpp, metric='euclidean', metric_kwargs={'squared': True})
    elif init_strat in ["poisson++", "poisson++-dist"]:
        centers_poisson = poisson_seeding(X, n_clusters, random_state=random_state)
        if init_strat == "poisson++":
            labels = _poisson_labels_through_centroids(X, centers_poisson)
        else:
            labels = poisson_dist(X, centers_poisson).argmin(1)
    return labels

"""
====
3CPO
====
"""

def get_log_probs_poisson(X: np.ndarray, row_lambdas: np.ndarray, column_lambdas: np.ndarray,
                          reduction: str, column_sums: np.ndarray | None = None) -> np.ndarray:
    """
    Calculate the relevant part of the loss for Poisson-based approaches.
    This is equal to: X_ij * log(lambda_j^C) - lambda_i^R * lambda_j^C.
    Usually includes a reduction over the rows or columns.

    Parameters
    ----------
    X : np.ndarray
        The data matrix
    row_lambdas : np.ndarray
        The row lambdas
    column_lambdas : np.ndarray
        The cluster-specific column lambdas. Can be a 1d array or of shape (k, d), where each row corresonds to a cluster.
    reduction : str
        Reduction of the resulting log probability matrix.
        Can be "row" (result will be of shape (k, d)) or "column" (result will be of shape (n, k)).
        Cal also be 'none' if column_lambdas contains a single dimension
    column_sums : np.ndarray | None
        The columns sums of the data (only relevant if reduction is 'rows'). Can speed up computations in cases where the whole dataset is considered.

    Returns
    -------
    log_probs : np.ndarray
        The log probabilites with respect to the given clustering parameters
    """
    assert not np.any(row_lambdas == 0) and not np.any(column_lambdas == 0), f"row_lambdas: {row_lambdas}, column_lambdas: {column_lambdas}"
    reduction = reduction.lower()
    assert reduction in ["rows", "columns", "none"]
    if reduction == "columns":
        # \sum_j (X_{ij} * \log(\lambda_{j|k}^C)) - \lambda_i^R * \sum_j \lambda_{j|k}^C
        if column_lambdas.ndim == 1:
            term1 = xlogy(X, column_lambdas.reshape((1,-1))).sum(1)
            term2 = row_lambdas * column_lambdas.sum()
        else:
            term1 = X @ np.log(column_lambdas).T
            term2 = row_lambdas[:, None] * column_lambdas.sum(1)[None, :]
    elif reduction == "rows":
        # \log(\lambda_{j|k}^C) * \sum_i X­_{ij} - \lambda_{j|k}^C * \sum_i \lambda_i^R
        if column_sums is None:
            column_sums = X.sum(0)
        if column_lambdas.ndim == 1:
            term1 = xlogy(column_sums, column_lambdas)
        else:
            term1 = xlogy(column_sums.reshape((1, -1)), column_lambdas)
        term2 = column_lambdas * row_lambdas.sum()
    else:
        assert column_lambdas.ndim == 1, "If reduction is 'None', column_lambdas has to be a single set of values, i.e., the values for a single cluster"
        column_lambdas_reshape = column_lambdas.reshape((1,-1))
        term1 = xlogy(X, column_lambdas_reshape)
        term2 = row_lambdas.reshape((-1,1)) @ column_lambdas_reshape
    log_probs = term1 - term2
    return log_probs


class ThreeCPO(BaseEstimator, ClusterMixin):
    """
    The 3CPO algorithm.

    Parameters
    ----------
    n_clusters : int
        number of clusters (default: 8)
    max_iter : int
        Maximum number of iterations during the optimization (default: 300)
    n_init: int
        The number of times the algorithm is executed using different seeds.
        Only the result with the best reward will be returned (default: 10)
    outliers : bool
        Indicate if outltiers should be identified (default: False)
    re_init_empty_clusters : bool
        Re-initialize cluster that got empty during optimization (default: False)
    init_strat_rows : str
        The initialization strategy for the labels. Can be 'random', 'random-centers', 'random-centers-dist',
        'kmeans', 'kmeans++', 'rf+kmeans', 'rf+kmeans++', 'poisson++-dist', or 'poisson++' (default: 'poisson++')
    init_strat_columns : str
        The initialization strategy for the columns in c_minus.
        Can be 'random', 'strategy' or 'none' (default: 'strategy')
    column_bias_type : str
        Type of column bias. Can be 'none', 'mdl' or 'bic' (default: 'mdl')
    ignore_c_minus : bool
        If true, C_minus will always be the empty set. Used for ablation study (default: False)
    ignore_c_zero : bool
        If true, C_zero will always be the empty set. Used for ablation study (default: False)
    track_reward : bool
        track a list of the reward after each operation.
        Makes the execution slightly slower (default: False)
    random_state : np.random.RandomState | int | None
        The random state (default: None)

    Attributes
    ----------
    reward_ : float
        The final reward
    labels_ : np.ndarray
        The cluster labels
    c_minus_ : np.ndarray
        Indicator array for columns in C_minus
    c_zero_ : np.ndarray
        Indicator array for columns in C_zero
    c_one_ : np.ndarray
        Indicator array for columns in C_one
    column_lambdas_minus_ : np.ndarray
        The cluster-specific column lambdas in C_minus
    column_lambdas_zero_ : np.ndarray
        The cluster-specific column lambdas in C_zero
    column_lambdas_one_ : np.ndarray
        The cluster-specific column lambdas in C_one
    n_c_minus_changes_ : int
        The number of changes in C_minus
    n_c_zero_changes_ : int
        The number of changes in C_zero
    n_iter_ : int
        The number of used iterations
    all_rewards_ : tuple
        Tuple containig build as follows: (reward after each operation, iteration of change in C_Minus, iteration of change in C_zero).
        Is None if track_reward is False
    n_features_in_ : int
        the number of features used for the fitting
    """
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, n_init: int = 10, outliers: bool = False, re_init_empty_clusters: bool = False,
                init_strat_rows: str = "poisson++", init_strat_columns: str = "strategy", column_bias_type: str = "mdl", ignore_c_minus: bool = False,
                ignore_c_zero: bool = False, track_reward: bool = False, random_state: np.random.RandomState | int | None = None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.outliers = outliers
        self.re_init_empty_clusters = re_init_empty_clusters
        self.init_strat_rows = init_strat_rows
        self.init_strat_columns = init_strat_columns
        self.column_bias_type = column_bias_type
        self.ignore_c_minus = ignore_c_minus
        self.ignore_c_zero = ignore_c_zero
        self.track_reward = track_reward
        self.random_state = random_state

    def init_column_partitions(self, X: np.ndarray, column_sums: np.ndarray,
                               random_state: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the column partition into c_minus, c_zero and c_one.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        column_sums : np.ndarray
            The column sums
        random_state: np.random.RandomState
            The random state

        Returns
        -------
        tuple : tuple[np.ndarray, np.ndarray, np.ndarray]
            The partitions c_one, c_zero and c_minus
        """
        init_strat_columns = self.init_strat_columns.lower()
        assert init_strat_columns in ["strategy", "random", "none"]
        # Init selection for C_one
        if not self.ignore_c_minus:
            if init_strat_columns == "strategy":
                n_cols = len(column_sums)
                row_sums = X.sum(1)
                global_sum = column_sums.sum()
                denominator = row_sums.reshape((-1, 1)) * column_sums.reshape((1, -1))
                denominator[denominator == 0] = 1
                diff = xlogy(X, X * global_sum / denominator).sum(0)
                mask = column_sums != 0
                diff[mask] = diff[mask] / column_sums[mask]
                c_one = np.zeros(X.shape[1], dtype=bool)
                best_columns = np.argsort(diff)[-n_cols//2:]
                c_one[best_columns] = True
            elif init_strat_columns == "random":
                c_one = np.zeros(X.shape[1], dtype=bool)
                random_columns = random_state.rand(X.shape[1])
                c_one[random_columns >= min(0.5, random_columns.max())] = True
            else:
                c_one = np.ones(X.shape[1], dtype=bool)
        else:
            c_one = np.ones(X.shape[1], dtype=bool)
        c_minus = ~c_one
        c_zero = np.zeros(X.shape[1], dtype=bool)
        
        assert np.any(c_one)
        assert np.all((c_one.astype(int) + c_zero.astype(int) + c_minus.astype(int)) == 1)
        return c_one, c_zero, c_minus

    def get_column_biases(self, column_sums: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Calculate the biases for columns within c_one.

        Parameters
        ----------
        column_sums : np.ndarray
            The column sums
        n_samples : int
            The number of samples in the dataset

        Returns
        -------
        column_biases : np.ndarray
            The column biases
        """
        column_bias_type = self.column_bias_type.lower()
        assert column_bias_type in ["none", "mdl", "bic"]
        if column_bias_type == "none":
            column_biases = np.zeros(column_sums.shape[0])
        elif column_bias_type == "mdl":
            column_biases = (self.n_clusters - 1) * np.log(column_sums)
        else:
            column_biases = np.zeros(column_sums.shape[0]) + 0.5 * self.n_clusters * np.log(n_samples)
        return np.maximum(column_biases, 0)

    def update_lambdas(self, X: np.ndarray, labels: np.ndarray, c_one: np.ndarray, c_zero: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Update row and the cluster-specific column lambdas in c_one.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        labels : np.ndarray
            The cluster labels
        c_one : np.ndarray
            Indicator array for columns in c_one
        c_zero : np.ndarray
            Indicator array for columns in c_zero

        Returns
        -------
        tuple : tuple[np.ndarray, np.ndarray]
            The row lambdas, the column lambdas in c_one
        """
        column_lambdas_one = np.ones((self.n_clusters, X.shape[1]))
        mask_one_zero = (c_one | c_zero)
        X_one_zero = X[:, mask_one_zero] # If no columns are in c_zero, this does not change this parameter 
        row_sums = X_one_zero.sum(1)
        row_lambdas = row_sums.copy()
        if np.any(c_zero):
            X_zero = X[:, c_zero]
            sum_c_zero = X_zero.sum()
            row_lambdas /= sum_c_zero
            column_lambdas_one *= sum_c_zero
            for clust in range(self.n_clusters):
                in_cluster = (labels == clust)
                cluster_subset = X[in_cluster]
                if cluster_subset.shape[0] == 0:
                    continue
                column_sum_cluster = cluster_subset.sum(0)
                sum_c_zero_cluster = column_sum_cluster[c_zero].sum()
                # Update column lambdas in C_one
                column_lambdas_one[clust] *= column_sum_cluster / sum_c_zero_cluster
                # Update row lambdas
                sum_c_one_zero_cluster = column_sum_cluster[mask_one_zero].sum()
                row_lambdas[in_cluster] *= sum_c_zero_cluster / sum_c_one_zero_cluster
        else:
            sum_c_one = row_sums.sum()
            # Update row lambdas
            row_lambdas /= sum_c_one
            for clust in range(self.n_clusters):
                in_cluster = (labels == clust)
                cluster_subset = X[in_cluster]
                if cluster_subset.shape[0] == 0:
                    continue
                column_sum_cluster = cluster_subset.sum(0)
                # Update column lambdas in C_one
                column_lambdas_one[clust] *= column_sum_cluster * sum_c_one / column_sum_cluster[c_one].sum()
        sum_c_one_zero = 0
        if self.outliers and (-1 in labels):
            sum_c_one_zero = row_sums.sum()
            is_outlier = (labels == -1)
            row_lambdas[is_outlier] = row_sums[is_outlier] / sum_c_one_zero
        # Return row and column lambdas
        return row_lambdas, column_lambdas_one
    
    def update_column_partitions(self, X: np.ndarray, labels: np.ndarray, row_lambdas: np.ndarray,
                                 column_lambdas_one: np.ndarray, column_lambdas_zero: np.ndarray,
                                 reward_in_c_minus: np.ndarray, column_biases: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update the column partition into c_minus, c_zero and c_one.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        labels : np.ndarray
            The cluster labels
        row_lambdas : np.ndarray
            The row lambdas
        column_lambdas_one : np.ndarray
            The column lambdas in c_one
        column_lambdas_zero : np.ndarray
            The column lambdas in c_zero
        reward_in_c_minus : np.ndarray
            The reward of a column in c_zero
        column_biases : np.ndarray
            The column biases

        Returns
        -------
        tuple : tuple[np.ndarray, np.ndarray, np.ndarray]
            The partitions c_one, c_zero and c_minus
        """
        reward_for_row_lambda = (X * np.log(row_lambdas).reshape((-1,1))).sum(0)
        # Get reward for c_zero
        if not self.ignore_c_zero:
            reward_in_c_zero = get_log_probs_poisson(X, row_lambdas, column_lambdas_zero, "rows", column_lambdas_zero) + reward_for_row_lambda
        else:
            reward_in_c_zero = -np.ones(X.shape[1]) * np.inf
        # Get reward for c_one
        reward_in_c_one = reward_for_row_lambda - column_biases
        for clust in range(self.n_clusters):
            in_cluster = (labels == clust)
            cluster_subset = X[in_cluster]
            if cluster_subset.shape[0] == 0:
                continue
            reward_in_c_one = reward_in_c_one + get_log_probs_poisson(cluster_subset, row_lambdas[in_cluster], column_lambdas_one[clust], "rows")
        if self.outliers and -1 in labels:
            is_outlier = (labels == -1)
            reward_in_c_one_outliers = get_log_probs_poisson(X[is_outlier], row_lambdas[is_outlier], column_lambdas_zero, "rows")
            reward_in_c_one = reward_in_c_one + reward_in_c_one_outliers
        # Get column partitions
        c_one = (reward_in_c_one > reward_in_c_zero) & (reward_in_c_one > reward_in_c_minus)
        c_zero = (~c_one) & (reward_in_c_zero > reward_in_c_minus)
        c_minus = ~(c_one | c_zero)
        # Check that at least one column is contained in c_one
        if not np.any(c_one):
            if np.any(c_zero):
                c_one = c_zero
                c_zero = np.zeros(X.shape[1], dtype=bool)
            else:
                c_one = c_minus
                c_minus = np.zeros(X.shape[1], dtype=bool)
            print("c_one was empty, use c_zero/c_minus as c_one.")
        #assert np.all((c_one.astype(int) + c_zero.astype(int) + c_minus.astype(int)) == 1)
        return c_one, c_zero, c_minus

    def update_labels_through_prob(self, X: np.ndarray, column_sums: np.ndarray, c_one: np.ndarray,
                                   c_zero: np.ndarray, old_labels: np.ndarray) -> np.ndarray:
        """
        Update the cluster labels using probabilities.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        column_sums : np.ndarray
            The columns sums (i.e., the column lambdas in c_zero). Only used if outliers are considered)
        c_one : np.ndarray
            Indicator array for columns in c_one
        c_zero : np.ndarray
            Indicator array for columns in c_zero
        old_labels : np.ndarray
            The cluster labels from the last iteration
        
        Returns
        -------
        labels : np.ndarray
            The updated labels
        """
        # Calculate probabilites using expected values mu
        mask_one_zero = (c_one | c_zero)
        only_c_one =  c_one[mask_one_zero]
        X_one_zero = X[:, mask_one_zero] # If no columns are in c_zero, this does not change this parameter
        X_one = X_one_zero[:, only_c_one]
        row_sums = X_one_zero.sum(1)

        column_sum_c_one_zero_cluster = np.zeros((self.n_clusters, X_one_zero.shape[1]), dtype=np.float64)
        if self.outliers:
            valid_lables = (old_labels != -1)
            np.add.at(column_sum_c_one_zero_cluster, old_labels[valid_lables], X_one_zero[valid_lables])
        else:
            np.add.at(column_sum_c_one_zero_cluster, old_labels, X_one_zero)
        column_sum_c_one_zero_cluster[column_sum_c_one_zero_cluster == 0] = 1e-5
        sum_c_one_zero_cluster = column_sum_c_one_zero_cluster.sum(1).reshape(-1, 1)
        probs = get_log_probs_poisson(X_one, row_sums, column_sum_c_one_zero_cluster[:, only_c_one] / sum_c_one_zero_cluster, "columns")

        labels = np.argmax(probs, axis=1)
        if self.outliers:
            probs_max = probs[np.arange(X.shape[0]), labels]
            sum_c_one_zero = column_sums[mask_one_zero].sum()
            probs_outlier = get_log_probs_poisson(X_one, row_sums, column_sums[c_one] / sum_c_one_zero, "columns")
            labels[probs_outlier > probs_max] = -1
        # Check for lost clusters
        if self.re_init_empty_clusters:
            labels_wo_outliers = labels[labels != -1] if self.outliers else labels
            unique_labels, cluster_sizes = np.unique(labels_wo_outliers, return_counts=True)
            if not np.array_equal(unique_labels, np.arange(self.n_clusters)):
                # Add random clusters
                missing_clusters = [a for a in np.arange(self.n_clusters) if a not in unique_labels]
                assert len(missing_clusters) > 0
                print("found missing clusters", missing_clusters, " - ", unique_labels, cluster_sizes)
                # Get probability of splitting a cluster based on cluster size
                replacement_probs_per_cluster = np.zeros(self.n_clusters)
                replacement_probs_per_cluster[unique_labels] = cluster_sizes / cluster_sizes.sum()
                replacement_probs = replacement_probs_per_cluster[labels_wo_outliers]
                replacement_probs = replacement_probs / replacement_probs.sum()
                for a in missing_clusters:
                    random_sample = np.random.choice(np.where(labels != -1)[0], p=replacement_probs)
                    print("set label of", random_sample, "with label", labels[random_sample], "to", a)
                    labels[random_sample] = a
        return labels

    def calculate_reward(self, X: np.ndarray, labels: np.ndarray, row_lambdas: np.ndarray,
                         column_lambdas_one: np.ndarray, column_lambdas_zero: np.ndarray,
                         c_one: np.ndarray, c_zero: np.ndarray, c_minus: np.ndarray,
                         reward_in_c_minus: np.ndarray, column_biases: np.ndarray) -> float:
        """
        Calculate the reward of 3CPO.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        labels : np.ndarray
            The cluster labels
        row_lambdas : np.ndarray
            The row lambdas
        column_lambdas_one : np.ndarray
            The column lambdas in c_one
        column_lambdas_zero : np.ndarray
            The column lambdas in c_zero
        c_one : np.ndarray
            Indicator array for columns in c_one
        c_zero : np.ndarray
            Indicator array for columns in c_zero
        c_minus : np.ndarray
            Indicator array for columns in c_minus
        reward_in_c_minus : np.ndarray
            The reward of a column in c_zero
        column_biases : np.ndarray
            The column biases

        Returns
        -------
        reward : float
            The reward with respect to the given clustering parameters
        """
        reward_for_row_lambda = (X[:, ~c_minus] * np.log(row_lambdas).reshape((-1,1))).sum()
        reward_minus = reward_in_c_minus[c_minus].sum()
        reward_zero = get_log_probs_poisson(X[:, c_zero], row_lambdas, column_lambdas_zero[c_zero], "rows", column_lambdas_zero[c_zero]).sum()
        reward_one = 0
        for clust in range(self.n_clusters):
            in_cluster = (labels == clust)
            cluster_subset = X[in_cluster]
            if cluster_subset.shape[0] == 0:
                continue
            reward_one += get_log_probs_poisson(cluster_subset[:, c_one], row_lambdas[in_cluster], column_lambdas_one[clust, c_one], "rows").sum()
        if self.outliers and -1 in labels:
            is_outlier = (labels == -1)
            reward_one += get_log_probs_poisson(X[is_outlier][:, c_one], row_lambdas[is_outlier], column_lambdas_zero[c_one], "rows").sum()
        reward = reward_for_row_lambda + reward_minus + reward_zero + reward_one
        # Add penalty for columns used for clustering
        reward = reward - column_biases[c_one].sum()
        # Add penalty for number of clusters
        reward = reward - integer_costs(int(self.n_clusters), use_log2=False) - X.shape[0] * np.log(self.n_clusters)
        return reward

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'ThreeCPO':
        """
        Fit 3CPO to the given data set.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        y : np.ndarray | None
            The labels

        Returns
        -------
        self : ThreeCPO
            This instance of the ThreeCPO algorithm
        """
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        assert self.n_clusters < X.shape[0]
        assert np.all(X >= 0)
        X = X.astype(float)
        X += 1e-3
        column_sums = X.sum(0)
        column_lambdas_zero = column_sums
        column_lambdas_minus = column_sums / X.shape[0]
        if not self.ignore_c_minus:
            reward_in_c_minus = get_log_probs_poisson(X, np.ones(X.shape[0]), column_lambdas_minus, "rows", column_sums)
        else:
            reward_in_c_minus = -np.ones(X.shape[1]) * np.inf
        column_biases = self.get_column_biases(column_sums, X.shape[0])
        self.reward_ = -np.inf
        for run in range(self.n_init):
            all_rewards: list[float] | None = [] if self.track_reward else None
            c_one_sizes: list[int] | None = [] if self.track_reward else None
            c_one_changes: list[int] | None = [] if self.track_reward else None
            c_zero_changes: list[int] | None = [] if self.track_reward else None
            c_minus_changes: list[int] | None = [] if self.track_reward else None
            # Initialize
            c_one, c_zero, c_minus = self.init_column_partitions(X, column_sums, random_state)
            labels = initial_poisson_clustering_labels(X[:, c_one], self.n_clusters, self.init_strat_rows.lower(), random_state)
            # Start optimization
            for iteration in range(self.max_iter):
                # Update lambdas
                row_lambdas, column_lambdas_one = self.update_lambdas(X, labels, c_one, c_zero)
                if self.track_reward and all_rewards is not None and c_one_sizes is not None:
                    all_rewards.append(self.calculate_reward(X, labels, row_lambdas, column_lambdas_one, column_lambdas_zero, c_one, c_zero, c_minus, reward_in_c_minus, column_biases))
                    c_one_sizes.append(np.sum(c_one))
                    old_c_minus = c_minus
                # Update column partition
                old_c_zero = c_zero
                old_c_one = c_one
                c_one, c_zero, c_minus = self.update_column_partitions(X, labels, row_lambdas, column_lambdas_one, column_lambdas_zero, reward_in_c_minus, column_biases)
                # Update labels
                old_labels = labels
                labels = self.update_labels_through_prob(X, column_sums, c_one, c_zero, old_labels)
                # Check column lambdas changes
                c_zero_not_changed = np.array_equal(old_c_zero, c_zero)
                c_one_not_changed = np.array_equal(old_c_one, c_one)
                if self.track_reward and c_minus_changes is not None and c_zero_changes is not None and c_one_changes is not None:
                    c_minus_not_changed = np.array_equal(old_c_minus, c_minus)
                    # Check for changes and convergence
                    if not c_minus_not_changed:
                        c_minus_changes.append(iteration)
                    if not c_zero_not_changed:
                        c_zero_changes.append(iteration)
                    if not c_one_not_changed:
                        c_one_changes.append(iteration)
                if np.array_equal(labels, old_labels) and c_zero_not_changed and c_one_not_changed:
                    break
            # Save result with best reward
            reward_iter = self.calculate_reward(X, labels, row_lambdas, column_lambdas_one, column_lambdas_zero, c_one, c_zero, c_minus, reward_in_c_minus, column_biases)
            #print(f"Finish run {run} in iteration {iteration} with reward {reward_iter}")
            if self.track_reward and all_rewards is not None and c_one_sizes is not None:
                all_rewards.append(reward_iter) 
                c_one_sizes.append(int(np.sum(c_one)))
            if reward_iter > self.reward_:
                self.reward_ = reward_iter
                self.labels_ = labels.astype(np.int32)
                self.c_minus_ = c_minus
                self.c_zero_ = c_zero
                self.c_one_ = c_one
                self.column_lambdas_minus_ = column_lambdas_minus
                self.column_lambdas_zero_ = column_lambdas_zero
                self.column_lambdas_one_ = column_lambdas_one
                self.n_iter_ = iteration + 1
                if self.track_reward:
                    self.c_one_changes_ = np.array(c_one_changes)
                    self.c_minus_changes_ = np.array(c_minus_changes)
                    self.c_zero_changes_ = np.array(c_zero_changes)
                    self.all_rewards_ = np.array(all_rewards)
                    self.c_one_sizes_ = np.array(c_one_sizes)
        self.n_features_in_ = X.shape[1]
        return self

    def plot_reward(self, formulatea_as_reward: bool = False, add_size_c1: bool = True,
                    add_column_changes: bool = True, save_path: str | Path | None = None) -> None:
        """
        Plot the changes of the reward during optimization.

        Parameters
        ----------
        formulatea_as_reward : bool
            Defines whether the y-axis shows a reward or a loss (loss = -reward) (default: False)
        add_column_changes : bool
            Add the column changes to the plot as vertical dashed lines (default: True)
        save_path : str | Path | None
            Path were the plot should be saved (default: None)
        """
        assert hasattr(self, "all_rewards_"), "Make sure that the algorithm has been fitted with track_reward = True."
        y_values = self.all_rewards_ if formulatea_as_reward else -self.all_rewards_
        y_values = (y_values - y_values.min()) / (y_values.max() - y_values.min())
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(len(y_values)), y_values, c="blue")
        ax1.set_xlabel("Iteration", fontsize=20)
        ax1.set_ylabel("Reward" if formulatea_as_reward else "Loss", fontsize=20, color="blue")
        x_axis_limit = ax1.get_xlim()
        if add_column_changes:
            ax1.plot(x_axis_limit, [-0.02] * 2, c="black", linewidth=0.5)
            ax1.text(x_axis_limit[0] - (x_axis_limit[1] - x_axis_limit[0]) * 0.05, -0.115, "$C_1$", fontsize=12)
            ax1.text(x_axis_limit[0] - (x_axis_limit[1] - x_axis_limit[0]) * 0.05, -0.195, "$C_0$", fontsize=12)
            ax1.text(x_axis_limit[0] - (x_axis_limit[1] - x_axis_limit[0]) * 0.05, -0.275, "$C_-$", fontsize=12)
            ax1.text(3 / 4 * x_axis_limit[0], -0.37, "Changes of column partitions", fontsize=14)
            for i, entry in enumerate(self.c_one_changes_):
                ax1.scatter([entry + 0.5], [-0.1], c="coral", marker="|", s=100, label="change of C_1" if i == 0 else None)
            for i, entry in enumerate(self.c_zero_changes_):
                ax1.scatter([entry + 0.5], [-0.18], c="skyblue", marker="|", s=100, label="change of C_0" if i == 0 else None)
            for i, entry in enumerate(self.c_minus_changes_):
                ax1.scatter([entry + 0.5], [-0.26], c="gray", marker="|", s=100, label="change of C_-" if i == 0 else None)
            ax1.scatter([0], [-0.34], c="white", marker="|")
            #plt.legend(loc="best")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.tick_params(axis="x", labelsize=15)
        ax1.set_yticks([])
        ax1.set_xlim(x_axis_limit)

        if add_size_c1:
            n_dims = self.c_one_.shape[0]
            ax2 = ax1.twinx()
            ax2.set_ylabel("$|C_1|$", fontsize=20, color="green")
            ax2.plot(np.arange(len(y_values)), self.c_one_sizes_, "--", color="green")
            ax2.set_yticks([int(n_dims * i / 4) for i in range(5)])
            ax2.tick_params(axis='y', labelcolor="green", labelsize=15)
            ax2.set_ylim(top=n_dims*1.05, bottom=-n_dims * 0.37)

        fig.tight_layout()

        plt.gca().xaxis.get_major_locator().set_params(integer=True)   # type: ignore[call-arg]
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

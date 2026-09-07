import numpy as np
from clustpy.utils.checks import check_parameters
from sklearn.base import BaseEstimator, ClusterMixin
from clustpy.centroid.threecpo import initial_poisson_clustering_labels, get_log_probs_poisson
from sklearn.utils.validation import check_is_fitted


class PoissonL(BaseEstimator, ClusterMixin):
    """
    Execute the PoissonL/PoissonC clustering procedure.
    The algorithms are specifically designed for count data.
    They model all values of a count matrix as a combination of row- and cluster-specific column-values that are updated through
    an Expectation Maximization (EM) approach.
    While PoissonL updates the labels by directly optimizing the log likelihood, PoissonC uses the Chi-squared test.

    Parameters
    ----------
    n_clusters : int
        the number of clusters (default: 8)
    max_iter : int
        the maximum number of iterations (default: 300)
    n_init: int
        The number of times the algorithm is executed using different seeds. Only the result with the best reward will be returned (default: 10)
    init_strat : str
        The initialization strategy. Can be 'random', 'random-centers', 'random-centers-dist', 
        'kmeans', 'kmeans++', 'rf+kmeans', 'rf+kmeans++', 'poisson++-dist', or 'poisson++' (default: 'random')
    random_state : np.random.RandomState | int | None
        Use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    reward_ : float
        The final reward
    labels_ : np.ndarray
        The cluster labels
    column_lambdas_ : np.ndarray
        The cluster-specific column lambdas
    n_iter_ : int
        The number of used iterations
    n_features_in_ : int
        the number of features used for the fitting

    References
    -------
    Cai, Li, et al. "Clustering analysis of SAGE data using a Poisson approach." 
    Genome biology 5.7 (2004): R51.
    """
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, n_init: int = 10, init_strat: str = "random", 
                 random_state: np.random.RandomState | int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_strat = init_strat.lower()
        self.random_state = random_state

    def update_column_lambdas(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update the cluster-specific column lambdas.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        labels : np.ndarray
            The cluster labels

        Returns
        -------
        column_lambdas : np.ndarray
            The updated column lambdas
        """
        column_lambdas = np.ones((self.n_clusters, X.shape[1]), dtype=float)
        for clust in range(self.n_clusters):
            cluster_subset = X[labels == clust]
            if cluster_subset.shape[0] == 0:
                continue
            cluster_column_sum = cluster_subset.sum(0)
            column_lambdas[clust] = cluster_column_sum / cluster_column_sum.sum()
        return column_lambdas

    def update_labels(self, X: np.ndarray, row_lambdas: np.ndarray, column_lambdas: np.ndarray) -> np.ndarray:
        """
        Update the cluster labels using probabilities.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        row_lambdas : np.ndarray
            The row lambdas
        column_lambdas : np.ndarray
            The cluster-specific column lambdas

        Returns
        -------
        labels : np.ndarray
            The updated labels
        """
        probs = get_log_probs_poisson(X, row_lambdas, column_lambdas, "columns")
        labels = np.argmax(probs, axis=1)
        return labels

    def calculate_reward(self, X: np.ndarray, labels: np.ndarray, row_lambdas: np.ndarray, column_lambdas: np.ndarray) -> float:
        """
        Calculate the reward for PoissonL/PoissonC.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        labels : np.ndarray
            The cluster labels
        row_lambdas : np.ndarray
            The row lambdas
        column_lambdas : np.ndarray
            The cluster-specific column lambdas

        Returns
        -------
        reward : float
            The reward with respect to the given clustering parameters
        """
        reward = 0.
        for clust in range(self.n_clusters):
            in_cluster = (labels==clust)
            cluster_subset = X[in_cluster]
            if cluster_subset.shape[0] == 0:
                continue
            reward += get_log_probs_poisson(cluster_subset, row_lambdas[in_cluster], column_lambdas[clust], "rows").sum()
        return reward

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'PoissonL':
        """
        Execute the actual PoissonL/PoissonC clustering process.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        y : np.ndarray | None
            The labels

        Returns
        -------
        self : PoissonL
            This instance of the PoissonL/PoissonC algorithm
        """
        X, _, random_state = check_parameters(X=X, y=y, random_state=self.random_state)
        assert self.n_clusters < X.shape[0]
        assert np.all(X >= 0)
        X = X.astype(float)
        X += 1e-3
        row_lambdas = X.sum(1)
        self.reward_ = -np.inf
        for _ in range(self.n_init):
            # Initialize
            labels = initial_poisson_clustering_labels(X, self.n_clusters, self.init_strat, random_state)
            # Start procdeure
            for iteration in range(self.max_iter):
                column_lambdas = self.update_column_lambdas(X, labels)
                old_labels = labels 
                labels = self.update_labels(X, row_lambdas, column_lambdas)
                if np.array_equal(labels, old_labels):
                    break
            # Save result with best reward
            reward_iter = self.calculate_reward(X, labels, row_lambdas, column_lambdas)
            if reward_iter > self.reward_:
                self.reward_ = reward_iter
                self.labels_ = labels.astype(np.int32)
                self.column_lambdas_ = column_lambdas
                self.n_iter_ = iteration + 1
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given data set using PoissonL/PoissonC.

        Parameters
        ----------
        X : np.ndarray
            The data matrix

        Returns
        -------
        self : np.ndarray
            The predicted labels
        """
        check_is_fitted(self, ["labels_", "n_features_in_"])
        X, _, _ = check_parameters(X=X, estimator_obj=self, allow_size_1=True)
        X = X.astype(float)
        X += 1e-3
        row_lambdas = X.sum(1)
        labels = self.update_labels(X, row_lambdas, self.column_lambdas_)
        return labels

class PoissonC(PoissonL):
    
    def update_labels(self, X: np.ndarray, row_lambdas: np.ndarray, column_lambdas: np.ndarray) -> np.ndarray:
        """
        Update the cluster labels using Chi-squared.

        Parameters
        ----------
        X : np.ndarray
            The data matrix
        row_lambdas : np.ndarray
            The row lambdas
        column_lambdas : np.ndarray
            The cluster-specific column lambdas

        Returns
        -------
        labels : np.ndarray
            The updated labels
        """
        row_lamdas_reshape = row_lambdas.reshape((-1, 1))
        X_scaled = (X ** 2)  / row_lamdas_reshape
        term1 = X_scaled @ (1. / column_lambdas).T
        term2 = 2 * row_lamdas_reshape
        term3 = row_lamdas_reshape * column_lambdas.sum(1).reshape((1, -1))
        diffs = term1 - term2 + term3
        labels = np.argmin(diffs, axis=1)
        return labels
    
"""
Kaufman, Rousseeuw "Divisive Analysis (Program DIANA)" Chapter six from Finding Groups in Data:
An Introduction to Cluster Analysis. 1990.

@authors Stephan Breimann
"""

import time
import os
import pandas as pd
import operator
from sklearn.metrics import pairwise_distances


# I Helper Functions
class _Diana:
    """Internal class for Divisive clustering algorithm (Top-Down)"""
    def __init__(self, X, n_clusters=None, metric="euclidean", compute_full_tree=False, distance_threshold=0):
        self.distance_threshold = distance_threshold
        self.compute_full_tree = compute_full_tree
        self.X = X
        self.metric = metric
        self.dict_n_clusters = {}
        self.n_clusters = n_clusters

    def _update_dict_n_clusters(self, list_clusters):
        """Update dict for each round of clustering"""
        self.dict_n_clusters[len(list_clusters)] = list_clusters.copy()

    def _average_dissimilarity(self, X=None, list_index=None):
        """Calculate average dissimilarity for every sample in dissimilarity_matrix"""
        if X is not None and len(X) > 1:
            dis_matrix = pairwise_distances(X, metric=self.metric)
            list_mean_dis = [val.sum() / (len(dis_matrix) - 1) for val in dis_matrix]
            dict_id_mean_dis = dict(zip(list_index, list_mean_dis))
            return dict_id_mean_dis
        else:
            return None

    def _diameter(self, X=None):
        """Calculate diameter of cluster, i.e. 'largest dissimilarity between two of its objects'"""
        dis_matrix = pairwise_distances(X, metric=self.metric)
        if X is None:
            return 0
        elif len(X) > 1:
            return dis_matrix.max()
        else:
            return 0

    def _max_diameter_cluster(self, list_index_clusters=None):
        """Get biggest cluster of list, i.e. cluster with largest diameter.
        Diameter of 0 means that all objects are similiar)"""
        list_diameter = [self._diameter(X=self.X.loc[index_cluster]) for index_cluster in list_index_clusters]
        max_diameter = max(list_diameter)
        index_max_diameter = list_diameter.index(max_diameter)
        list_max_diameter_cluster = list_index_clusters[index_max_diameter]
        return list_max_diameter_cluster, max_diameter

    def _split_clust(self, list_cluster_a=None, list_cluster_b=None):
        """Split cluster a in cluster a and cluster b (splinter group) based on average dissimilarity."""
        # Set index_cluster_a and X_a
        if list_cluster_a is None:
            X_a = self.X
            list_cluster_a = X_a.index.tolist()
        else:
            X_a = self.X.loc[list_cluster_a]
        # Get dict with average dissimilarity for each sample (given as index)
        dict_id_mean_dis = self._average_dissimilarity(X=X_a, list_index=list_cluster_a)
        # Recursive call if cluster a contains more than two objects
        if len(list_cluster_a) > 2:
            # Select index with maximum dissimilarity
            index_max_dis = max(dict_id_mean_dis.items(), key=operator.itemgetter(1))[0]
            index_selected_a = [i for i in list_cluster_a if i != index_max_dis]
            if list_cluster_b is None:
                index_selected_b = [index_max_dis]
            else:
                index_selected_b = list_cluster_b + [index_max_dis]
            # Calculate diameter (max dissimilarity between two objects in cluster)
            diameter_a = self._diameter(X=self.X.loc[index_selected_a])
            diameter_b = self._diameter(X=self.X.loc[index_selected_b])
            # Recursive call of split clust if diameter of cluster a is higher than for cluster b
            if diameter_a >= diameter_b and len(X_a) > 1:
                return self._split_clust(list_cluster_a=index_selected_a, list_cluster_b=index_selected_b)
            else:
                return list_cluster_a, list_cluster_b
        # Return both clusters if they exist
        elif len(list_cluster_a) == 2 and list_cluster_b is not None:
            return list_cluster_a, list_cluster_b
        # Split cluster a if no object in cluster b
        elif len(list_cluster_a) == 2:
            return [list_cluster_a[0]], [list_cluster_a[1]]
        else:
            return [list_cluster_a]

    def recursive_clustering(self, list_clusters=None):
        """Recursive clustering algorithm of diana"""
        # Initiate clustering
        if list_clusters is None:
            index_cluster_a, index_cluster_b = self._split_clust()
            list_clusters = [index_cluster_a, index_cluster_b]
            self._update_dict_n_clusters(list_clusters)
        # Split biggest cluster from list of clusters and insert new clusters
        list_max_diameter_cluster, max_diameter = self._max_diameter_cluster(list_index_clusters=list_clusters)
        index_remove = list_clusters.index(list_max_diameter_cluster)
        list_clusters.pop(index_remove)
        list_split_clusters = self._split_clust(list_cluster_a=list_max_diameter_cluster)
        list_clusters.insert(index_remove, list_split_clusters[0])
        if len(list_split_clusters) > 1:
            list_clusters.append(list_split_clusters[1])
        # Recursive call until max_n_clusters is reached or max diameter is 0
        # (i.e. cluster with largest dissimilarity contains just similar objects)
        if self.n_clusters is not None and len(self.dict_n_clusters) + 1 == self.n_clusters:
            return list_clusters
        elif self.compute_full_tree and round(max_diameter, 5) == self.distance_threshold:
            return list_clusters
        else:
            # Save cluster level
            self._update_dict_n_clusters(list_clusters)
            return self.recursive_clustering(list_clusters=list_clusters)


def _get_clusters_from_diana(n_dict_clusters):
    """Convert dict_n_clusters into df with cluster labels for each element and each clustering level"""
    dict_dict_cluster_ids = {}
    for i in n_dict_clusters:
        dict_cluster_ids = {}
        for j, clusters in enumerate(n_dict_clusters[i]):
            for scale in clusters:
                dict_cluster_ids[scale] = j + 1
        dict_dict_cluster_ids[i] = dict_cluster_ids
    df = pd.DataFrame.from_dict(dict_dict_cluster_ids, orient="index")
    return df


# II Main Functions
class Diana:
    """Divisive clustering algorithm (Top-Down) based on pairwise dissimilarity of objects

    Recursively splitting clusters with maximum dissimilarity
    measured by distance metric (e.g., Euclidean distance).

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be None if
        'distance_threshold' is not None.

    metric : str or callable, default='euclidean'
        Metric used to compute the dissimilarity. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed".

    distance_threshold : float, default=None
        Minimum of dissimilarity above which clusters will be split. For the Diana clustering algorithm,
        dissimilarity is given as the maximum diameter, which is the largest dissimilarity between two
        objects within a cluster.
        If None, 'n_clusters' must not be None. If not None, 'compute_full_tree' must be True.

    compute_full_tree : bool, default=False
        If False, stop early the construction of the clustering tree at n_clusters,
        which is the equal to the tree level.
        If True, clustering will stop when 'distance_threshold' (needs to be given) is reached.
        This is useful to decrease computation time if the number of clusters is not small compared to
        the number of samples.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        cluster labels for each data point for given 'n_cluster'

    df_hierarchy_: df (n_samples, n_clusters)
        cluster lables for each data point (columns) for all cluster levels (index)


    """
    def __init__(self, n_clusters=2, metric="euclidean", compute_full_tree=False, distance_threshold=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.distance_threshold = distance_threshold
        self.compute_full_tree = compute_full_tree

    def fit(self, X):
        """Fit hierarchical clustering from features.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster
        """
        # Check parameters
        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))
        if self.n_clusters is not None and self.compute_full_tree:
            raise ValueError("Either 'n_clusters' or 'compute_full_tree' has to be set, "
                             " the other needs to be None or False, respectively.")
        if self.compute_full_tree and self.distance_threshold is None:
            raise ValueError("If 'compute_full_tree' is True, "
                             "'distance_threshold' must be  set.")
        if self.distance_threshold is None or self.distance_threshold < 0:
            # Set distance_threshold to theoretical minimum if not given or below
            self.distance_threshold = 0

        # Hierarchical clustering
        diana = _Diana(X=X, n_clusters=self.n_clusters,
                       metric=self.metric,
                       distance_threshold=self.distance_threshold,
                       compute_full_tree=self.compute_full_tree)
        diana.recursive_clustering()
        df = _get_clusters_from_diana(diana.dict_n_clusters)
        df = df[X.index.to_list()]
        self.df_hierarchy_ = df
        self.labels_ = df.loc[self.n_clusters].values


# III Test/Caller Functions
def diana_caller():
    """"""
    # TODO shift into new example file ?
    folder_data = os.path.dirname(os.path.realpath(__file__)).replace("/hierarchical", "/data/")
    df = pd.read_excel(folder_data + "All_Scales_Norm_Modified.xlsx", index_col=0) #.transpose().sample(300, random_state=1)
    df_t = df[[x for x in list(df) if "Lins" not in x]].transpose().sample(200, random_state=1)
    diana = Diana(n_clusters=20)
    diana.fit(X=df_t)


# IV Main
def main():
    t0 = time.time()
    diana_caller()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()

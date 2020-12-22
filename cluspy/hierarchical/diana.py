"""
Kaufman, Rousseeuw "Divisive Analysis (Program DIANA)" Chapter six from Finding Groups in Data:
An Introduction to Cluster Analysis. 1990.

@authors Stephan Breimann
"""

import time
import os
import pandas as pd
import operator
import numpy as np
from sklearn.metrics import pairwise_distances
import sys

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe
sys.setrecursionlimit(10000)


# I Helper Functions


# II Main Functions
class Diana:
    """Divisive clustering algorithm (Top-Down) based on pairwise dissimilarity of objects.
    Adapted from R package cluster: https://www.rdocumentation.org/packages/cluster/versions/2.1.0/topics/diana"""
    def __init__(self, X=None, metric="euclidean"):
        self.X = X
        self.metric = metric
        self.dict_n_clusters = {}

    def _update_dict_n_clusters(self, list_index_clusters):
        """Update dict for each round of clustering"""
        self.dict_n_clusters[len(list_index_clusters)] = list_index_clusters.copy()

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
        """Get biggest cluster of list, i.e. cluster with largest diameter"""
        list_diameter = [self._diameter(X=self.X.loc[index_cluster]) for index_cluster in list_index_clusters]
        index_max_diameter = list_diameter.index(max(list_diameter))
        max_diameter_cluster = list_index_clusters[index_max_diameter]
        return max_diameter_cluster

    def _split_clust(self, index_cluster_a=None, index_cluster_b=None):
        """Split cluster a in cluster a and cluster b (splitner group) based on average dissimilarity."""
        # Set index_cluster_a and X_a
        if index_cluster_a is None:
            X_a = self.X
            index_cluster_a = X_a.index.tolist()
        else:
            X_a = self.X.loc[index_cluster_a]
        # Get dict with average dissimilarity for each sample (given as index)
        dict_id_mean_dis = self._average_dissimilarity(X=X_a, list_index=index_cluster_a)
        # Recursive call if cluster a contains more than two objects
        if len(index_cluster_a) > 2:
            # Select index with maximum dissimilarity
            index_max_dis = max(dict_id_mean_dis.items(), key=operator.itemgetter(1))[0]
            index_selected_a = [i for i in index_cluster_a if i != index_max_dis]
            if index_cluster_b is None:
                index_selected_b = [index_max_dis]
            else:
                index_selected_b = index_cluster_b + [index_max_dis]
            # Calculate diameter (max dissimilarity between two objects in cluster)
            diameter_a = self._diameter(X=self.X.loc[index_selected_a])
            diameter_b = self._diameter(X=self.X.loc[index_selected_b])
            # Recursive call of split clust if diameter of cluster a is higher than for cluster b
            if diameter_a >= diameter_b and len(X_a) > 1:
                return self._split_clust(index_cluster_a=index_selected_a, index_cluster_b=index_selected_b)
            else:
                return index_cluster_a, index_cluster_b
        # Return both clusters if they exist
        elif len(index_cluster_a) == 2 and index_cluster_b is not None:
            return index_cluster_a, index_cluster_b
        # Split cluster a if no object in cluster b
        elif len(index_cluster_a) == 2:
            return [index_cluster_a[0]], [index_cluster_a[1]]
        else:
            return [index_cluster_a]

    def fit(self, list_index_clusters=None):
        """"""
        # Initiate clustering
        if list_index_clusters is None:
            index_cluster_a, index_cluster_b = self._split_clust()
            list_index_clusters = [index_cluster_a, index_cluster_b]
            self._update_dict_n_clusters(list_index_clusters)
        # Split biggest cluster from list of clusters
        max_diameter_cluster = self._max_diameter_cluster(list_index_clusters=list_index_clusters)
        list_index_clusters.remove(max_diameter_cluster)
        list_index_clusters.extend(self._split_clust(index_cluster_a=max_diameter_cluster))
        self._update_dict_n_clusters(list_index_clusters)
        # Recursive call of fit until biggest cluster consists of 2 scales
        max_n_custer = max([len(cluster) for cluster in list_index_clusters])
        if len(max_diameter_cluster) > 2 or max_n_custer > 1:
            return self.fit(list_index_clusters=list_index_clusters)
        else:
            return list_index_clusters

    def get_dict_n_clusters(self):
        return self.dict_n_clusters


# III Test/Caller Functions
def diana_caller():
    """"""
    folder_data = os.path.dirname(os.path.realpath(__file__)).replace("/hierarchical", "/data/")
    df = pd.read_excel(folder_data + "All_Scales_Norm_Modified.xlsx", index_col=0).transpose().sample(300,
                                                                                                      random_state=1)
    diana = Diana(X=df)
    list_index_clusters = diana.fit()
    for n in diana.get_dict_n_clusters():
        print(n, diana.get_dict_n_clusters()[n])

# IV Main
def main():
    t0 = time.time()
    diana_caller()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()

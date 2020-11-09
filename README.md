# ClusPy

The package offers an easy way to cluster data in python.
It can be combined with all algorithms from [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html).
Furthermore, it contains wrappers for [pyclustering](https://pyclustering.github.io/) implementations.

## Implemented algorithms

- Centroid-based clustering
    - X-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/d2/d8b/namespacepyclustering_1_1cluster_1_1xmeans.html))
    - G-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/dc/d86/namespacepyclustering_1_1cluster_1_1gmeans.html))
    - PG-Means
    - Fuzzy-C-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/de/df0/namespacepyclustering_1_1cluster_1_1fcm.html))
    - Dip-Means
    - Projected Dip-Means
- Density-based clustering
    - Multi Density DBSCAN
- Subspace clustering
    - SubKmeans
    - MDL-SubKmeans
- Alternative clustering (Non-redundant clustering)
    - NR-Kmeans
- Deep clustering
    - DEC
    - IDEC
    - DCN

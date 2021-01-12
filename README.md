# ClusPy

The package offers an easy way to cluster data in python.
It can be combined with all algorithms from [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html) 
and other packages.
Furthermore, it contains wrappers for [pyclustering](https://pyclustering.github.io/) implementations.

The main focus of this package is not efficiency but being able to try out a wide range of methods.
In particular, this should also make lesser-known methods accessible in a simple and convenient way.

## Components

### Algorithms

- Centroid-based clustering
    - X-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/d2/d8b/namespacepyclustering_1_1cluster_1_1xmeans.html))
    - G-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/dc/d86/namespacepyclustering_1_1cluster_1_1gmeans.html))
    - PG-Means
    - Fuzzy-C-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/de/df0/namespacepyclustering_1_1cluster_1_1fcm.html))
    - Dip-Means
    - Projected Dip-Means
    - DipExt & DipInit
    - Skinnydip
- Density-based clustering
    - Multi Density DBSCAN
- Subspace clustering
    - SubKmeans & MDL extension
- Hierarchical clustering
    - Diana
- Alternative clustering (Non-redundant clustering)
    - NR-Kmeans
- Deep clustering
    - DEC
    - IDEC
    - DCN
    - VaDE
    - DEDC
    
### Other implementations

- Metrics
    - Variation of information
    - Pair Counting Scores (f1, rand, jaccard, recall, precision)
    - Scores for multiple labelings
    - Confusion Matrix
- Utils
    - Hartigans Dip-test
    
## Compatible packages

- [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html) 
    - K-Means
    - Affinity propagation
    - Mean-shift
    - Spectral clustering
    - Ward hierarchical clustering
    - Agglomerative clustering
    - DBSCAN
    - OPTICS
    - Gaussian mixtures
	- BIRCH
- [kmodes](https://github.com/nicodv/kmodes)
    - k-modes
    - k-prototypes 
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
    - HDBSCAN
- [scikit-learn-extra](https://scikit-learn-extra.readthedocs.io/en/latest/index.html)
    - k-medoids
    - Density-Based common-nearest-neighbors clustering

# ClusPy

The package offers an easy way to cluster data in python.
It can be combined with all algorithms from [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html) 
and other packages.
Furthermore, it contains wrappers for [pyclustering](https://pyclustering.github.io/) implementations.

The main focus of this package is not efficiency but being able to try out a wide range of methods.
In particular, this should also make lesser-known methods accessible in a simple and convenient way.

## Installation

Just clone the repository, go to the directory and execute:

`pip install -r requirements.txt`

Afterwards, you will be good to go.

## Components

### Algorithms

- Partition-based clustering
    - X-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/d2/d8b/namespacepyclustering_1_1cluster_1_1xmeans.html))
    - G-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/dc/d86/namespacepyclustering_1_1cluster_1_1gmeans.html))
    - PG-Means
    - Fuzzy-C-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/de/df0/namespacepyclustering_1_1cluster_1_1fcm.html))
    - Dip-Means
    - Projected Dip-Means
    - DipExt & DipInit
    - UniDip & Skinnydip
- Density-based clustering
    - Multi Density DBSCAN
- Subspace clustering
    - SubKmeans
- Hierarchical clustering
    - Diana
- Alternative clustering (Non-redundant clustering)
    - NR-Kmeans
    - AutoNR
- Deep clustering
    - Simple Autoencoder
    - Stacked Autoencoder
    - DEC
    - IDEC
    - DCN
    - VaDE
    - DipDECK
    
### Other implementations

- Metrics
    - Variation of information
    - Unsupervised Clustering Accuracy
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

## Examples

### 1)

In this simple example, the XMeans algorithm is run on the optdigits dataset.
Afterwards, the normalized mutual information is calculated and printed.

```python
from cluspy.partition import XMeans
from cluspy.data import load_optdigits
from sklearn.metrics import normalized_mutual_info_score as nmi

data, labels = load_optdigits()
xm = XMeans()
xm.fit(data)
my_nmi = nmi(labels, xm.labels_)
print(my_nmi)
```

### 2)

In this simple example, the SubKmeans algorithm is run on a synthetic subspace dataset.
Then the normalized mutual information is calculated and printed again.

```python
from cluspy.subspace import SubKmeans
from cluspy.data import create_subspace_data
from sklearn.metrics import normalized_mutual_info_score as nmi

data, labels = create_subspace_data(1000, n_clusters=4, cluster_features=2, total_features = 5)
sk = SubKmeans(4)
sk.fit(data)
my_nmi = nmi(labels, sk.labels_)
print(my_nmi)
```

### 3)

In this more complex example, we use ClusPy's evaluation functions, 
which automatically run the specified algorithms multiple times on previously defined data sets.
All results of the given metrics are stored in a Pandas dataframe.

```python
from cluspy.evaluation import *
from cluspy.deep import DEC, IDEC, DCN, VaDE
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from cluspy.data import load_usps, load_mnist

def znorm(X):
    return (X - np.mean(X)) / np.std(X)

def identity_plus(X):
    return X + 1

to_ignore = ["VaDE", "IDEC"]
datasets = [
    EvaluationDataset("MNIST", data=load_mnist, preprocess_methods=znorm, ignore_algorithms=["DEC"]),
    EvaluationDataset("USPS", data=load_usps, preprocess_methods=znorm),
    EvaluationDataset("USPS+1", data=load_usps, preprocess_methods=[znorm, identity_plus],
                          ignore_algorithms=to_ignore)
    ]
algorithms = [
    EvaluationAlgorithm("DEC", DEC, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                   "dec_epochs": 150, "embedding_size": 10}),
    EvaluationAlgorithm("IDEC", IDEC, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                     "dec_epochs": 150, "embedding_size": 10}),
    EvaluationAlgorithm("DCN", DCN, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                   "dcn_epochs": 150, "embedding_size": 10}),
    EvaluationAlgorithm("VaDE", VaDE, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                     "vade_epochs": 150, "embedding_size": 10})]
metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ARI", ari)]
df = evaluate_multiple_datasets(datasets, algorithms, metrics, 10, True, True, False, True,
                                    save_path="valuation.csv", save_intermediate_results=True)
print(df)
```

test

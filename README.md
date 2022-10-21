# ClusPy

[![Build Status](https://app.travis-ci.com/collinleiber/ClusPy.svg?branch=master)](https://app.travis-ci.com/collinleiber/ClusPy)
[![codecov](https://codecov.io/github/collinleiber/ClusPy/branch/master/graphs/badge.svg)](https://codecov.io/github/collinleiber/ClusPy) 

The package provides a simple way to cluster data in Python.
For this purpose it provides a variety of algorithms. 
Additionally, methods that are often needed for research purposes, such as loading frequently used datasets 
(e.g. from the [UCI repository](https://archive.ics.uci.edu/ml/index.php)), are largely automated. 

The focus of the ClusPy package is not on efficiency (here we recommend e.g. [pyclustering](https://pyclustering.github.io/)), 
but on the possibility to try out a wide range of modern scientific methods.
In particular, this should also make lesser-known methods accessible in a simple and convenient way.

It can be combined with all algorithms from [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html) 
and other packages (see below).
Furthermore, it contains wrappers for [pyclustering](https://pyclustering.github.io/) implementations.

## Installation

### For Users

The package can be installed directly from git by executing:

`sudo pip3 install git+https://github.com/collinleiber/ClusPy.git`

Alternatively, clone the repository, go to the directory and execute:

`sudo python setup.py install`

If you have no sudo rights you can use:

`python setup.py install --prefix ~/.local`

### For Developers

Clone the repository, go to the directory and do the following.

Install package locally and compile C files:

`python setup.py install --prefix ~/.local`

Copy compiled C files to correct file location:

`python setup.py build_ext --inplace`

Remove cluspy via pip to avoid ambiguities during development, e.g., when changing files in the code:

`pip3 uninstall cluspy`

## Components

### Algorithms

- Partition-based clustering
    - LDA-K-Means
    - PG-Means
    - Dip-Means
    - Projected Dip-Means
    - DipExt & DipInit
    - SkinnyDip & UniDip
    - SubKmeans
    - Fuzzy-C-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/de/df0/namespacepyclustering_1_1cluster_1_1fcm.html))
    - X-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/d2/d8b/namespacepyclustering_1_1cluster_1_1xmeans.html))
    - G-Means (from [pyclustering](https://pyclustering.github.io/docs/0.10.0/html/dc/d86/namespacepyclustering_1_1cluster_1_1gmeans.html))
- Density-based clustering
    - Multi Density DBSCAN
- Hierarchical clustering
    - Diana
- Alternative clustering (Non-redundant clustering)
    - NR-Kmeans
    - AutoNR
- Deep clustering
    - Flexible Autoencoder
    - Stacked Autoencoder
    - Variational Autoencoder
    - DEC
    - IDEC
    - DCN
    - VaDE
    - DipDECK
    - DipEncoder
    - ENRC
    
### Other implementations

- Metrics
    - Unsupervised Clustering Accuracy
    - Variation of information
    - Confusion Matrix
    - Pair Counting Scores (f1, rand, jaccard, recall, precision)
    - Scores for multiple labelings (see alternative clustering algorithms)
- Utils
    - Hartigans Dip-test
    - Various plots
    - Evaluation methods
- Datasets
    - Synthetic dataset creator for subspace and alternative clustering 
    - Various real-world datasets (e.g. Iris, Wine, Mice protein, Optdigits, MNIST, ...)
    
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
- [Density Peak Clustering](https://github.com/colinwke/dpca)
    - DPC

## Examples

### 1)

In this first example, the subspace algorithm SubKmeans is run on a synthetic subspace dataset.
Afterwards, the clustering accuracy is calculated to evaluate the result.

```python
from cluspy.partition import SubKmeans
from cluspy.data import create_subspace_data
from cluspy.metrics import unsupervised_clustering_accuracy as acc

data, labels = create_subspace_data(1000, n_clusters=4, subspace_features=[2,5])
sk = SubKmeans(4)
sk.fit(data)
acc_res = acc(labels, sk.labels_)
print("Clustering accuracy:", acc_res)
```

### 2)

The second example covers the topic of non-redundant/alternative clustering.
Here, the NrKmeans algorithm is run on the CMUfaces dataset.
Beware that NrKmeans as a non-redundant clustering algorithm returns multiple labelings.
Therefore, we calculate the confusion matrix by comparing each combination of labels using the normalized mutual information (nmi).
The confusion matrix will be printed and finally the best matching nmi will be stated for each set of labels.

```python
from cluspy.alternative import NrKmeans
from cluspy.data import load_cmu_faces
from cluspy.metrics import MultipleLabelingsConfusionMatrix
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np

data, labels = load_cmu_faces()
nk = NrKmeans([20, 4, 4, 2])
nk.fit(data)
mlcm = MultipleLabelingsConfusionMatrix(labels, nk.labels_, nmi)
mlcm.rearrange()
print(mlcm.confusion_matrix)
print(np.max(mlcm.confusion_matrix, axis=1))
```

### 3)

One mentionable feature of the ClusPy package is the ability to run various modern deep clustering algorithms out of the box. 
For example, the following code runs the DEC algorithm on the Optdigits dataset. 
To evaluate the result, we compute the adjusted RAND index (ari).

```python
from cluspy.deep import DEC
from cluspy.data import load_optdigits
from sklearn.metrics import adjusted_rand_score as ari

data, labels = load_optdigits()
dec = DEC(10)
dec.fit(data)
my_ari = ari(labels, dec.labels_)
print(my_ari)
```

### 4)

In this more complex example, we use ClusPy's evaluation functions, 
which automatically run the specified algorithms multiple times on previously defined datasets.
All results of the given metrics are stored in a Pandas dataframe.

```python
from cluspy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, evaluate_multiple_datasets
from cluspy.partition import ProjectedDipMeans, SubKmeans
from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from cluspy.data import load_breast_cancer, load_iris, load_wine
from cluspy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.decomposition import PCA
import numpy as np

def reduce_dimensionality(X, dims):
    pca = PCA(dims)
    X_new = pca.fit_transform(X)
    return X_new

def znorm(X):
    return (X - np.mean(X)) / np.std(X)

def minmax(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

datasets = [
    EvaluationDataset("Breast_pca_znorm", data=load_breast_cancer, preprocess_methods=[reduce_dimensionality, znorm],
                      preprocess_params=[{"dims": 0.9}, {}], ignore_algorithms=["pdipmeans"]),
    EvaluationDataset("Iris_pca", data=load_iris, preprocess_methods=reduce_dimensionality,
                      preprocess_params={"dims": 0.9}),
    EvaluationDataset("Wine", data=load_wine),
    EvaluationDataset("Wine_znorm", data=load_wine, preprocess_methods=znorm)]

algorithms = [
    EvaluationAlgorithm("SubKmeans", SubKmeans, {"n_clusters": None}),
    EvaluationAlgorithm("pdipmeans", ProjectedDipMeans, {}),  # Determines n_clusters automatically
    EvaluationAlgorithm("dbscan", DBSCAN, {"eps": 0.01, "min_samples": 5}, preprocess_methods=minmax,
                        deterministic=True),
    EvaluationAlgorithm("kmeans", KMeans, {"n_clusters": None}),
    EvaluationAlgorithm("kmeans_minmax", KMeans, {"n_clusters": None}, preprocess_methods=minmax)]

metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ACC", acc),
           EvaluationMetric("Silhouette", silhouette_score, use_gt=False)]

df = evaluate_multiple_datasets(datasets, algorithms, metrics, repetitions=5,
                                aggregation_functions=[np.mean, np.std, np.max, np.min],
                                add_runtime=True, add_n_clusters=True, save_path=None,
                                save_intermediate_results=False)
print(df)
```

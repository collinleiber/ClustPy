![Logo](https://github.com/collinleiber/ClustPy/blob/main/docs/logo.png)

---
[![PyPI version](https://badge.fury.io/py/clustpy.svg)](https://pypi.org/project/clustpy/)
[![TestMain](https://github.com/collinleiber/clustpy/actions/workflows/test-main.yml/badge.svg)](https://github.com/collinleiber/ClustPy/actions/workflows/test-main.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/collinleiber/ClustPy/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/collinleiber/ClustPy/tree/main)
[![codecov](https://codecov.io/gh/collinleiber/ClustPy/branch/main/graph/badge.svg?token=5AJYQFIYFR)](https://codecov.io/gh/collinleiber/ClustPy)
[![Docs](https://readthedocs.org/projects/clustpy/badge/?version=latest)](https://clustpy.readthedocs.io/en/latest/)

The package provides a simple way to cluster data in Python.
For this purpose it provides a variety of algorithms from different domains. 
Additionally, ClustPy includes methods that are often needed for research purposes, such as plots, clustering metrics or evaluation methods.
Further, it integrates various frequently used datasets (e.g. from the [UCI repository](https://archive.ics.uci.edu/ml/index.php)) through largely automated loading options. 

The focus of the ClustPy package is not on efficiency (here we recommend e.g. [pyclustering](https://pyclustering.github.io/)), 
but on the possibility to try out a wide range of modern scientific methods.
In particular, this should also make lesser-known methods accessible in a simple and convenient way.

Since it largely follows the implementation conventions of [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html), 
it can be combined with many other packages (see below).

## Installation

### For Users

#### Stable Version

The current stable version can be installed by the following command:

`pip3 install clustpy`

#### Development Version

The current development version can be installed directly from git by executing:

`sudo pip3 install git+https://github.com/collinleiber/ClustPy.git`

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

Remove clustpy via pip to avoid ambiguities during development, e.g., when changing files in the code:

`pip3 uninstall clustpy`

## Components

### Algorithms

- Partition-based clustering
    - [DipExt + DipInit](https://link.springer.com/chapter/10.1007/978-3-030-67658-2_6)
    - [Dip-Means](https://proceedings.neurips.cc/paper/2012/hash/a8240cb8235e9c493a0c30607586166c-Abstract.html)
    - Dip'n'sub
    - [GapStatistic](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00293)
    - [G-Means](https://proceedings.neurips.cc/paper/2003/hash/234833147b97bb6aed53a8f4f1c7a7d8-Abstract.html)
    - [LDA-K-Means](https://dl.acm.org/doi/abs/10.1145/1273496.1273562)
    - [PG-Means](https://proceedings.neurips.cc/paper/2006/hash/a9986cb066812f440bc2bb6e3c13696c-Abstract.html)
    - [Projected Dip-Means](https://dl.acm.org/doi/abs/10.1145/3200947.3201008)
    - [SkinnyDip + UniDip](https://dl.acm.org/doi/abs/10.1145/2939672.2939740) & TailoredDip
    - [SubKmeans](https://dl.acm.org/doi/abs/10.1145/3097983.3097989)
    - [X-Means](https://web.cs.dal.ca/~shepherd/courses/csci6403/clustering/xmeans.pdf)
- Density-based clustering
    - [Multi Density DBSCAN](https://link.springer.com/chapter/10.1007/978-3-642-23878-9_53)
- Hierarchical clustering
    - [Diana](https://www.jstor.org/stable/2290430?origin=crossref)
- Alternative clustering / Non-redundant clustering
    - [AutoNR](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.26)
    - [NR-Kmeans](https://dl.acm.org/doi/abs/10.1145/3219819.3219945)
- Deep clustering
    - Autoencoder
        - [Flexible Autoencoder](https://www.aaai.org/Library/AAAI/1987/aaai87-050.php)
        - [Neighbor Encoder](https://arxiv.org/abs/1811.01557)
        - [Stacked Autoencoder](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf?ref=https://githubhelp.com)
        - [Variational Autoencoder](https://arxiv.org/abs/1312.6114)
    - [DCN](https://dl.acm.org/doi/abs/10.5555/3305890.3306080)
    - [DEC](https://dl.acm.org/doi/abs/10.5555/3045390.3045442)
    - [DipDECK](https://dl.acm.org/doi/10.1145/3447548.3467316)
    - [DipEncoder](https://dl.acm.org/doi/10.1145/3534678.3539407)
    - [ENRC](https://ojs.aaai.org/index.php/AAAI/article/view/5961)
    - [IDEC](https://dl.acm.org/doi/abs/10.5555/3172077.3172131)
    - [VaDE](https://dl.acm.org/doi/abs/10.5555/3172077.3172161)
    
### Other implementations

- Metrics
    - Confusion Matrix
    - [Pair Counting Scores (f1, rand, jaccard, recall, precision)](https://link.springer.com/article/10.1007/s10115-008-0150-6)
    - [Unsupervised Clustering Accuracy](https://ieeexplore.ieee.org/abstract/document/5454426)
    - [Variation of information](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_14)
    - Scores for multiple labelings (see alternative clustering algorithms)
        - Multiple Labelings Confusion Matrix
        - [Multiple Labelings Pair Counting Scores](https://ieeexplore.ieee.org/abstract/document/6228189)
- Utils
    - Automatic evaluation methods
    - [Hartigans Dip-test](https://www.jstor.org/stable/2241144)
    - Various plots
- Datasets
    - Synthetic dataset creator for subspace and alternative clustering 
    - Various real-world datasets (e.g. Iris, Wine, Mice protein, Optdigits, MNIST, ...)
    
## Compatible packages

We stick as close as possible to the implementation details of sklean clustering. Therefore, our methods are compatible with many other packages. Examples are:

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

## Coding Examples

### 1)

In this first example, the subspace algorithm SubKmeans is run on a synthetic subspace dataset.
Afterwards, the clustering accuracy is calculated to evaluate the result.

```python
from clustpy.partition import SubKmeans
from clustpy.data import create_subspace_data
from clustpy.metrics import unsupervised_clustering_accuracy as acc

data, labels = create_subspace_data(1000, n_clusters=4, subspace_features=[2,5])
sk = SubKmeans(4)
sk.fit(data)
acc_res = acc(labels, sk.labels_)
print("Clustering accuracy:", acc_res)
```

### 2)

The second example covers the topic of non-redundant/alternative clustering.
Here, the NrKmeans algorithm is run on the Fruit dataset.
Beware that NrKmeans as a non-redundant clustering algorithm returns multiple labelings.
Therefore, we calculate the confusion matrix by comparing each combination of labels using the normalized mutual information (nmi).
The confusion matrix will be printed and finally the best matching nmi will be stated for each set of labels.

```python
from clustpy.alternative import NrKmeans
from clustpy.data import load_fruit
from clustpy.metrics import MultipleLabelingsConfusionMatrix
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np

data, labels = load_fruit()
nk = NrKmeans([3, 3])
nk.fit(data)
mlcm = MultipleLabelingsConfusionMatrix(labels, nk.labels_, nmi)
mlcm.rearrange()
print(mlcm.confusion_matrix)
print(np.max(mlcm.confusion_matrix, axis=1))
```

### 3)

One mentionable feature of the ClustPy package is the ability to run various modern deep clustering algorithms out of the box. 
For example, the following code runs the DEC algorithm on the Newsgroups dataset. 
To evaluate the result, we compute the adjusted RAND index (ari).

```python
from clustpy.deep import DEC
from clustpy.data import load_newsgroups
from sklearn.metrics import adjusted_rand_score as ari

data, labels = load_newsgroups()
dec = DEC(20)
dec.fit(data)
my_ari = ari(labels, dec.labels_)
print(my_ari)
```

### 4)

In this more complex example, we use ClustPy's evaluation functions, 
which automatically run the specified algorithms multiple times on previously defined datasets.
All results of the given metrics are stored in a Pandas dataframe.

```python
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, evaluate_multiple_datasets
from clustpy.partition import ProjectedDipMeans, SubKmeans
from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from clustpy.data import load_breast_cancer, load_iris, load_wine
from clustpy.metrics import unsupervised_clustering_accuracy as acc
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

df = evaluate_multiple_datasets(datasets, algorithms, metrics, n_repetitions=5,
                                aggregation_functions=[np.mean, np.std, np.max, np.min],
                                add_runtime=True, add_n_clusters=True, save_path=None,
                                save_intermediate_results=False)
print(df)
```

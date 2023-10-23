<a href="https://github.com/collinleiber/ClustPy"><img src="https://raw.githubusercontent.com/collinleiber/ClustPy/main/docs/logo.png" height="175" ></a><br>
---
[![PyPI version](https://badge.fury.io/py/clustpy.svg)](https://pypi.org/project/clustpy/)
[![TestMain](https://github.com/collinleiber/clustpy/actions/workflows/test-main.yml/badge.svg)](https://github.com/collinleiber/ClustPy/actions/workflows/test-main.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/collinleiber/ClustPy/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/collinleiber/ClustPy/tree/main)
[![codecov](https://codecov.io/gh/collinleiber/ClustPy/branch/main/graph/badge.svg?token=5AJYQFIYFR)](https://codecov.io/gh/collinleiber/ClustPy)
[![Docs](https://readthedocs.org/projects/clustpy/badge/?version=latest)](https://clustpy.readthedocs.io/en/)

The package provides a simple way to perform clustering in Python.
For this purpose it provides a variety of algorithms from different domains. 
Additionally, ClustPy includes methods that are often needed for research purposes, such as plots, clustering metrics or evaluation methods.
Further, it integrates various frequently used datasets (e.g., from the [UCI repository](https://archive.ics.uci.edu/)) through largely automated loading options. 

The focus of the ClustPy package is not on efficiency (here we recommend e.g. [pyclustering](https://pyclustering.github.io/)), 
but on the possibility to try out a wide range of modern scientific methods.
In particular, this should also make lesser-known methods accessible in a simple and convenient way.

Since it largely follows the implementation conventions of [sklearn clustering](https://scikit-learn.org/stable/modules/clustering.html), 
it can be combined with many other packages (see below).

# Installation

## For Users

### Stable Version

The current stable version can be installed by the following command:

`pip install clustpy`

Note that a gcc compiler is required for installation.
Therefore, in case of an installation error, make sure that:
- Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/) is installed
- Linux/Mac: Python dev is installed (e.g., by running `apt-get install python-dev` - the exact command may differ depending on the linux distribution)

The error messages may look like this:
- 'error: command 'gcc' failed: No such file or directory'
- 'Could not build wheels for clustpy, which is required to install pyproject.toml-based projects'
- 'Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools'

### Development Version

The current development version can be installed directly from git by executing:

`sudo pip install git+https://github.com/collinleiber/ClustPy.git`

Alternatively, clone the repository, go to the directory and execute:

`sudo python setup.py install`

If you have no sudo rights you can use:

`python setup.py install --prefix ~/.local`

## For Developers

Clone the repository, go to the directory and do the following (NumPy must be installed beforehand).

Install package locally and compile C files:

`python setup.py install --prefix ~/.local`

Copy compiled C files to correct file location:

`python setup.py build_ext --inplace`

Remove clustpy via pip to avoid ambiguities during development, e.g., when changing files in the code:

`pip uninstall clustpy`

# Components

## Clustering Algorithms

- Partition-based clustering
    - DipExt + DipInit [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-67658-2_6)]
    - Dip-Means [[Paper](https://proceedings.neurips.cc/paper/2012/hash/a8240cb8235e9c493a0c30607586166c-Abstract.html)]
    - Dip'n'sub [[Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch13)]
    - GapStatistic [[Paper](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00293)]
    - G-Means [[Paper](https://proceedings.neurips.cc/paper/2003/hash/234833147b97bb6aed53a8f4f1c7a7d8-Abstract.html)]
    - LDA-K-Means [[Paper](https://dl.acm.org/doi/abs/10.1145/1273496.1273562)]
    - PG-Means [[Paper](https://proceedings.neurips.cc/paper/2006/hash/a9986cb066812f440bc2bb6e3c13696c-Abstract.html)]
    - Projected Dip-Means [[Paper](https://dl.acm.org/doi/abs/10.1145/3200947.3201008)]
    - SkinnyDip + UniDip [[Paper](https://dl.acm.org/doi/abs/10.1145/2939672.2939740)] & TailoredDip [[Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch13)]
    - SpecialK [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-46150-8_16)]
    - SubKmeans [[Paper](https://dl.acm.org/doi/abs/10.1145/3097983.3097989)]
    - X-Means [[Paper](https://web.cs.dal.ca/~shepherd/courses/csci6403/clustering/xmeans.pdf)]
- Density-based clustering
    - Multi Density DBSCAN [[Paper](https://link.springer.com/chapter/10.1007/978-3-642-23878-9_53)]
- Hierarchical clustering
    - Diana [[Paper](https://www.jstor.org/stable/2290430?origin=crossref)]
- Alternative clustering / Non-redundant clustering
    - AutoNR [[Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.26)]
    - NR-Kmeans [[Paper](https://dl.acm.org/doi/abs/10.1145/3219819.3219945)]
- Deep clustering
    - Autoencoder
        - Convolutional Autoencoder [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)]
        - Feedforward Autoencoder [[Paper](https://www.aaai.org/Library/AAAI/1987/aaai87-050.php)]
        - Neighbor Encoder [[Paper](https://arxiv.org/abs/1811.01557)]
        - Stacked Autoencoder [[Paper](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf?)]
        - Variational Autoencoder [[Paper](https://arxiv.org/abs/1312.6114)]
    - ACe/DeC [[Paper](https://www.ijcai.org/proceedings/2021/389)]
    - DCN [[Paper](https://dl.acm.org/doi/abs/10.5555/3305890.3306080)]
    - DEC [[Paper](https://dl.acm.org/doi/abs/10.5555/3045390.3045442)]
    - DipDECK [[Paper](https://dl.acm.org/doi/10.1145/3447548.3467316)]
    - DipEncoder [[Paper](https://dl.acm.org/doi/10.1145/3534678.3539407)]
    - DKM [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865520302749)]
    - ENRC [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5961)]
    - IDEC [[Paper](https://dl.acm.org/doi/abs/10.5555/3172077.3172131)]
    - VaDE [[Paper](https://dl.acm.org/doi/abs/10.5555/3172077.3172161)]

## Other implementations

- Metrics
    - Confusion Matrix
    - Fair Normalized Mutual Information (FNMI) [[Paper](https://dl.acm.org/doi/abs/10.1145/2808797.2809344)]
    - Information-Theoretic External Cluster-Validity Measure (DOM) [[Paper](https://dl.acm.org/doi/10.5555/2073876.2073893)]
    - Pair Counting Scores (f1, rand, jaccard, recall, precision) [[Paper](https://link.springer.com/article/10.1007/s10115-008-0150-6)]
    - Scores for multiple labelings (see alternative clustering algorithms)
        - Multiple Labelings Confusion Matrix
        - Multiple Labelings Pair Counting Scores [[Paper](https://ieeexplore.ieee.org/abstract/document/6228189)]
    - Unsupervised Clustering Accuracy [[Paper](https://ieeexplore.ieee.org/abstract/document/5454426)]
    - Variation of information [[Paper](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_14)]
- Utils
    - Automatic evaluation methods
    - Hartigans Dip-test [[Paper](https://www.jstor.org/stable/2241144)]
    - Various plots
- Datasets
    - Synthetic dataset creators for subspace and alternative clustering 
    - Real-world dataset loaders (e.g., Iris, Wine, Mice protein, Optdigits, MNIST, ...)
        - UCI Repository [[Website](https://archive.ics.uci.edu/)]
        - UEA & UCR Time Series Classification Repository [[Website](http://www.timeseriesclassification.com/)]
        - MedMNIST [[Website](https://medmnist.com/)]
        - Torchvision Datasets [[Website](https://pytorch.org/vision/stable/datasets.html)]
        - Sklearn Datasets [[Website](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)]
        - Others
    - Dataset loaders for datasets with multiple labelings
        - ALOI (subset) [[Website](https://aloi.science.uva.nl/)]
        - CMU Face [[Website](http://archive.ics.uci.edu/ml/datasets/cmu+face+images)]
        - Dancing Stickfigures [[Paper](https://dl.acm.org/doi/abs/10.1145/2623330.2623734)]
        - Fruit [[Paper](https://link.springer.com/article/10.1007/s10115-016-0998-9)]
        - NRLetters [[Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.26)]
        - WebKB [[Website](http://www.cs.cmu.edu/~webkb/)]

## Python environments

ClustPy utilizes global [Python environment variables](https://docs.python.org/3/library/os.html#os.environ) in some places. These can be defined using `os.environ['VARIABLE_NAME'] = VARIABLE_VALUE`.
The following variable names are used:

- 'CLUSTPY_DATA': Defines the path where downloaded datasets should be saved.
- 'CLUSTPY_DEVICE': Define the device to be used for Pytorch applications. Example: `os.environ['CLUSTPY_DEVICE'] = 'cuda:1'`

# Citation

If you use the ClustPy package in the context of a scientific publication, please cite it as follows:

*Leiber, C., Miklautz, L., Plant, C., Böhm, C. (2023, October).
Application of Deep Clustering Algorithms.
Proceedings of the 32nd ACM International Conference on Information and Knowledge Management.*

**BibTeX:**
```latex
@inproceedings{leiber2023application,
  title = {Application of Deep Clustering Algorithms},
  author = {Leiber, Collin and Miklautz, Lukas and Plant, Claudia and Böhm, Christian},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  year = {2023},
  pages = {5208–5211},
  organization = {Association for Computing Machinery},
  url = {https://doi.org/10.1145/3583780.3615290}
}
```

# Compatible packages

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

# Coding Examples

## 1)

In this first example, the subspace algorithm SubKmeans is run on a synthetic subspace dataset.
Afterward, the clustering accuracy is calculated to evaluate the result.

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

## 2)

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

## 3)

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

## 4)

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

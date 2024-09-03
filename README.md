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

If you want to install the complete package including all data loader functions, you should use:

`pip install clustpy[full]`

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

### Partition-based Clustering

| Algorithm                     | Publication                                                              | Published at       | Original Code                                                                                                                                                                                                                   | Docs                                                                                                                           |
|-------------------------------|--------------------------------------------------------------------------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| DipInit (incl. DipExt)        | [Utilizing Structure-Rich Features to Improve Clustering](https://link.springer.com/chapter/10.1007/978-3-030-67658-2_6) | ECML PKDD 2020     | [Link](https://figshare.com/articles/code/Utilizing_Structure-rich_Features_to_improve_Clustering/12063252/1) (R)                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.dipext.DipInit)                       |
| DipMeans                      | [Dip-means: an incremental clustering method for estimating the number of clusters](https://proceedings.neurips.cc/paper/2012/hash/a8240cb8235e9c493a0c30607586166c-Abstract.html) | NIPS 2012          | [Link](https://kalogeratos.com/psite/material/dip-means/) (Matlab)                                                                                                                                                              | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.dipmeans.DipMeans)                    |
| Dip'n'sub (incl. TailoredDip) | [Extension of the Dip-test Repertoire - Efficient and Differentiable p-value Calculation for Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch13) | SIAM SDM 2023      | [Link](https://figshare.com/articles/conference_contribution/Supplement_codes_and_data_for_the_paper_Extension_of_the_Dip-test_Repertoire_-_Efficient_and_Differentiable_p-value_Calculation_for_Clustering_/21916752) (Python) | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.dipnsub.DipNSub)                      |
| GapStatistic                  | [Estimating the number of clusters in a data set via the gap statistic](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00293) | RSS: Series B 2002 | -                                                                                                                                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.gapstatistic.GapStatistic)            | 
| G-Means                       | [Learning the k in k-means](https://proceedings.neurips.cc/paper/2003/hash/234833147b97bb6aed53a8f4f1c7a7d8-Abstract.html) | NIPS 2003          | -                                                                                                                                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.gmeans.GMeans)                        |
| LDA-K-Means                   | [Adaptive dimension reduction using discriminant analysis and K-means clustering](https://dl.acm.org/doi/abs/10.1145/1273496.1273562) | ICML 2007          | -                                                                                                                                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.ldakmeans.LDAKmeans)                  |
| PG-Means                      | [PG-means: learning the number of clusters in data](https://proceedings.neurips.cc/paper/2006/hash/a9986cb066812f440bc2bb6e3c13696c-Abstract.html) | NIPS 2006          | -                                                                                                                                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.pgmeans.PGMeans)                      |
| Projected Dip-Means           | [The Projected Dip-means Clustering Algorithm](https://dl.acm.org/doi/abs/10.1145/3200947.3201008) | SETN 2018          | -                                                                                                                                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.projected_dipmeans.ProjectedDipMeans) |                             
| SkinnyDip (incl. UniDip)      | [Skinny-dip: Clustering in a Sea of Noise](https://dl.acm.org/doi/abs/10.1145/2939672.2939740) | KDD 2016           | [Link](https://github.com/samhelmholtz/skinny-dip) (R)                                                                                                                                                                          | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.skinnydip.SkinnyDip)                  |
| SpecialK | [k Is the Magic Number—Inferring the Number of Clusters Through Nonparametric Concentration Inequalities](https://link.springer.com/chapter/10.1007/978-3-030-46150-8_16) | ECML PKDD 2019     | [Link](https://github.com/Sibylse/SpecialK) (Python)                                                                                                                                                                            | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.specialk.SpecialK)                    |
| SubKmeans | [Towards an Optimal Subspace for K-Means](https://dl.acm.org/doi/abs/10.1145/3097983.3097989) | KDD 2017           | [Link](http://dmm.dbs.ifi.lmu.de/downloads/) (Scala)                                                                                                                                                                            | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.subkmeans.SubKmeans) |
| X-Means | [X-means: Extending k-means with efficient estimation of the number of clusters](https://web.cs.dal.ca/~shepherd/courses/csci6403/clustering/xmeans.pdf) | ICML 2000          | -                                                                                                                                                                                                                               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.partition.html#clustpy.partition.xmeans.XMeans) |

### Density-based Clustering

| Algorithm                     | Publication                                                              | Published at                             | Original Code | Docs                                                                                                                           |
|-------------------------------|--------------------------------------------------------------------------|------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------|
| Multi Density DBSCAN | [Multi Density DBSCAN](https://link.springer.com/chapter/10.1007/978-3-642-23878-9_53) | IDEAL 2011 | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.density.html#clustpy.density.multi_density_dbscan.MultiDensityDBSCAN) |

### Hierarchical Clustering

| Algorithm | Publication                                                              | Published at | Original Code | Docs                                                                                                        |
|-----------|--------------------------------------------------------------------------|--------------|---------------|-------------------------------------------------------------------------------------------------------------|
| DIANA     | [Finding Groups in Data: An Introduction to Cluster Analysis](https://www.jstor.org/stable/2290430?origin=crossref) | JASA 1991    | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.hierarchical.html#clustpy.hierarchical.diana.Diana) |

### Alternative Clustering / Non-redundant Clustering

| Algorithm     | Publication                                                              | Published at  | Original Code                                         | Docs                                                                                                            |
|---------------|--------------------------------------------------------------------------|---------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| AutoNR        | [Automatic Parameter Selection for Non-Redundant Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.26) | SIAM SDM 2022 | [Link](https://dmm.dbs.ifi.lmu.de/downloads) (Python) | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.alternative.html#clustpy.alternative.autonr.AutoNR)     |
| NR-Kmeans     | [Discovering Non-Redundant K-means Clusterings in Optimal Subspaces](https://dl.acm.org/doi/abs/10.1145/3219819.3219945) | KDD 2018      | [Link](https://dmm.dbs.ifi.lmu.de/downloads) (Scala)  | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.alternative.html#clustpy.alternative.nrkmeans.NrKmeans) | 
| Orth1 + Orth2 | [Non-redundant multi-view clustering via orthogonalization](https://ieeexplore.ieee.org/document/4470237) | ICDM 2007     | -                                                     | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.alternative.html#clustpy.alternative.orth.OrthogonalClustering)     |

### Deep Clustering

| Algorithm  | Publication                                                                                                                   | Published at                     | Original Code                                                                 | Docs                                                                                                  |
|------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| ACe/DeC    | [Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces](https://www.ijcai.org/proceedings/2021/389)   | IJCAI 2021                       | [Link](https://gitlab.cs.univie.ac.at/lukas/acedec_public) (Python + PyTorch) | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.enrc.ACeDeC)           |
| AEC        | [Auto-encoder based data clustering](https://link.springer.com/chapter/10.1007/978-3-642-41822-8_15)    | CIARP 2013                       | [Link](https://github.com/developfeng/DeepClustering/) (Matlab)              | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.aec.AEC)               |
| DCN        | [Towards K-means-friendly spaces: simultaneous deep learning and clustering](https://dl.acm.org/doi/abs/10.5555/3305890.3306080) | ICML 2017                        | [Link](https://github.com/boyangumn/DCN) (Python + Theano)                    | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.dcn.DCN)               |
| DDC        | [Deep density-based image clustering](https://www.sciencedirect.com/science/article/pii/S0950705120302112) | Knowledge-Based Systems 2020     | [Link](https://github.com/Yazhou-Ren/DDC/tree/master) (Python + Keras)        | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.ddc_n2d.DDC)           | 
| DEC        | [Unsupervised deep embedding for clustering analysis](https://dl.acm.org/doi/abs/10.5555/3045390.3045442) | ICML 2016                        | [Link](https://github.com/piiswrong/dec) (Python + Caffe)                     | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.dec.DEC)               |
| DeepECT    | [Deep embedded cluster tree](https://ieeexplore.ieee.org/abstract/document/8970987) | ICDM 2019                        | [Link](https://dmm.dbs.ifi.lmu.de/downloads) (Python + PyTorch)               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.deepect.DeepECT)       |
| DipDECK    | [Dip-based Deep Embedded Clustering with k-Estimation](https://dl.acm.org/doi/10.1145/3447548.3467316) | KDD 2021                         | [Link](https://dmm.dbs.ifi.lmu.de/downloads) (Python + PyTorch)               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.dipdeck.DipDECK)       |
| DipEncoder | [The DipEncoder: Enforcing Multimodality in Autoencoders](https://dl.acm.org/doi/10.1145/3534678.3539407) | KDD 2022                         | [Link](https://dmm.dbs.ifi.lmu.de/downloads) (Python + PyTorch)               | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.dipencoder.DipEncoder) |
| DKM        | [Deep k-Means: Jointly clustering with k-Means and learning representations](https://www.sciencedirect.com/science/article/abs/pii/S0167865520302749) | Pattern Recognition Letters 2020 | [Link](https://github.com/MaziarMF/deep-k-means) (Python + Tensorflow)        | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.dkm.DKM)               |
| ENRC       | [Deep Embedded Non-Redundant Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/5961) | AAAI 2020                        | [Link](https://gitlab.cs.univie.ac.at/lukas/enrcpublic) (Python + PyTorch)    | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.enrc.ENRC)             |
| IDEC       | [Improved Deep Embedded Clustering with Local Structure Preservation](https://www.ijcai.org/proceedings/2017/243) | IJCAI 2017                       | [Link](https://github.com/XifengGuo/IDEC) (Python + Keras)                    | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.dec.IDEC)              |
| N2D        | [N2d:(not too) deep clustering via clustering the local manifold of an autoencoded embedding](https://ieeexplore.ieee.org/document/9413131) | ICPR 2021                        | [Link](https://github.com/XifengGuo/IDEC) (Python + Keras)                    | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.ddc_n2d.N2D)           |
| VaDE       | [Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering](https://www.ijcai.org/proceedings/2017/0273) | IJCAI 2017                       | [Link](https://github.com/slim1017/VaDE) (Python + Keras)                     | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.html#clustpy.deep.vade.VaDE)             |

#### Neural Networks

| Algorithm                          | Publication                                                                                                                    | Published at | Original Code | Docs                                                                                                                                                         |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Convolutional Autoencoder (ResNet) | [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | CVPR 2016    | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.neural_networks.html#clustpy.deep.neural_networks.convolutional_autoencoder.ConvolutionalAutoencoder) |
| Feedforward Autoencoder | [Modular Learning in Neural Networks](https://dl.acm.org/doi/abs/10.5555/1863696.1863746) | AAAI 1987    | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.neural_networks.html#clustpy.deep.neural_networks.feedforward_autoencoder.FeedforwardAutoencoder)     |
| Neighbor Encoder | [Representation Learning by Reconstructing Neighborhoods](https://arxiv.org/abs/1811.01557) | arXiv 2018   | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.neural_networks.html#clustpy.deep.neural_networks.neighbor_encoder.NeighborEncoder)                   |
| Stacked Autoencoder | [Greedy Layer-Wise Training of Deep Networks](https://proceedings.neurips.cc/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html) | NIPS 2006    | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.neural_networks.html#clustpy.deep.neural_networks.stacked_autoencoder.StackedAutoencoder)             |
| Variational Autoencoder | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | ICLR 2014    | -             | [Link](https://clustpy.readthedocs.io/en/latest/clustpy.deep.neural_networks.html#clustpy.deep.neural_networks.variational_autoencoder.VariationalAutoencoder)  |

## Other implementations

- Metrics
    - Confusion Matrix [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.confusion_matrix.ConfusionMatrix)]
    - Fair Normalized Mutual Information (FNMI) [[Publication](https://dl.acm.org/doi/abs/10.1145/2808797.2809344)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.clustering_metrics.fair_normalized_mutual_information)]
    - Hierarchical Metrics
        - Dendrogram Purity [[Publication](https://dl.acm.org/doi/abs/10.1145/1102351.1102389)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.hierarchical_metrics.dendrogram_purity)]
        - Leaf Purity [[Publication](https://link.springer.com/article/10.1007/s41019-020-00134-0)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.hierarchical_metrics.leaf_purity)]
    - Information-Theoretic External Cluster-Validity Measure (DOM) [[Publication](https://dl.acm.org/doi/10.5555/2073876.2073893)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.clustering_metrics.information_theoretic_external_cluster_validity_measure)]
    - Pair Counting Scores (f1, rand, jaccard, recall, precision) [[Publication](https://link.springer.com/article/10.1007/s10115-008-0150-6)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.pair_counting_scores.PairCountingScores)]
    - Purity [[Publication](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.clustering_metrics.purity)]
    - Scores for multiple labelings (see alternative clustering algorithms)
        - Multiple Labelings Confusion Matrix [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.multipe_labelings_scoring.MultipleLabelingsConfusionMatrix)]
        - Multiple Labelings Pair Counting Scores [[Publication](https://ieeexplore.ieee.org/abstract/document/6228189)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.multipe_labelings_scoring.MultipleLabelingsPairCountingScores)]
    - Unsupervised Clustering Accuracy [[Publication](https://ieeexplore.ieee.org/abstract/document/5454426)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.clustering_metrics.unsupervised_clustering_accuracy)]
    - Variation of information [[Publication](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_14)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.metrics.html#clustpy.metrics.clustering_metrics.variation_of_information)]
- Utils
    - Automatic evaluation methods [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.utils.html#module-clustpy.utils.evaluation)]
    - Hartigans Dip-test [[Publication](https://www.jstor.org/stable/2241144)] [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.utils.html#module-clustpy.utils.diptest)]
    - Various plots [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.utils.html#module-clustpy.utils.plots)]
- Datasets
    - Synthetic dataset creators
        - For common subspace clustering [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.data.html#clustpy.data.synthetic_data_creator.create_subspace_data)]
        - For alternative clustering [[Docs](https://clustpy.readthedocs.io/en/latest/clustpy.data.html#clustpy.data.synthetic_data_creator.create_nr_data)]
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
        - Dancing Stickfigures [[Publication](https://dl.acm.org/doi/abs/10.1145/2623330.2623734)]
        - Fruit [[Publication](https://link.springer.com/article/10.1007/s10115-016-0998-9)]
        - NRLetters [[Publication](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.26)]
        - WebKB [[Website](http://www.cs.cmu.edu/~webkb/)]

## Python environments

ClustPy utilizes global [Python environment variables](https://docs.python.org/3/library/os.html#os.environ) in some places. These can be defined using `os.environ['VARIABLE_NAME'] = VARIABLE_VALUE`.
The following variable names are used:

- 'CLUSTPY_DATA': Defines the path where downloaded datasets should be saved.
- 'CLUSTPY_DEVICE': Define the device to be used for Pytorch applications. Example: `os.environ['CLUSTPY_DEVICE'] = 'cuda:1'`

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

data, labels = load_fruit(return_X_y=True)
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
from clustpy.data import load_optdigits
from sklearn.metrics import adjusted_rand_score as ari

data, labels = load_optdigits(return_X_y=True)
dec = DEC(10)
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

# Citation

If you use the ClustPy package in the context of a scientific publication, please cite it as follows:

*Leiber, C., Miklautz, L., Plant, C., Böhm, C. (2023, December).
Benchmarking Deep Clustering Algorithms With ClustPy.
2023 IEEE International Conference on Data Mining Workshops (ICDMW).* [[DOI](https://doi.org/10.1109/ICDMW60847.2023.00087)]

**BibTeX:**
```latex
@inproceedings{leiber2023benchmarking,
  title = {Benchmarking Deep Clustering Algorithms With ClustPy},
  author = {Leiber, Collin and Miklautz, Lukas and Plant, Claudia and Böhm, Christian},
  booktitle = {2023 IEEE International Conference on Data Mining Workshops (ICDMW)}, 
  year = {2023},
  pages = {625-632},
  publisher = {IEEE},
  doi = {10.1109/ICDMW60847.2023.00087}
}
```

# Publications using ClustPy

- [Application of Deep Clustering Algorithms](https://dl.acm.org/doi/abs/10.1145/3583780.3615290) (10/2023)
- [Benchmarking Deep Clustering Algorithms With ClustPy](https://ieeexplore.ieee.org/document/10411702) (12/2023)

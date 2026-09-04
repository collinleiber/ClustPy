import numpy as np
from sklearn.preprocessing import LabelEncoder
from hdbscan.validity import validity_index
from clustpy.metrics._metrics_utils import _check_length_data_and_labels, handle_noise


def dbcv_score(
    X,
    labels,
    metric="euclidean",
    noise_strategy="keep",
    d=None,
    per_cluster_scores=False,
    mst_raw_dist=False,
    verbose=False,
    **kwd_args,
):
    """
    Compute the density based cluster validity index for the
    clustering specified by `labels` and for each cluster in `labels`.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    metric : optional, string (default 'euclidean')
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    noise_strategy : str
        Strategy for handling noise (see clustpy.metrics.handle_noise). Must be one of:
        - "keep"            : Keep all noise points as they are (default).
        - "one_cluster"     : Assign all noise points to a single new cluster.
        - "singletons"      : Assign each noise point to its own cluster.
        - "filter"          : Remove all noise points.
        - "nearest_cluster" : Assign each noise point to nearest cluster.

    d : optional, integer (or None) (default None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.

    per_cluster_scores : optional, boolean (default False)
        Whether to return the validity index for individual clusters.
        Defaults to False with the function returning a single float
        value for the whole clustering.

    mst_raw_dist : optional, boolean (default False)
        If True, the MST's are constructed solely via 'raw' distances (depending on the given metric, e.g. sqeuclidean distances)
        instead of using mutual reachability distances. Thus setting this parameter to True avoids using 'all-points-core-distances' at all.
        This is advantageous specifically in the case of elongated clusters that lie in close proximity to each other <citation needed>.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    validity_index : float
        The density based cluster validity index for the clustering. This
        is a numeric value between -1 and 1, with higher values indicating
        a 'better' clustering.

    per_cluster_validity_index : array (n_clusters,)
        The cluster validity index of each individual cluster as an array.
        The overall validity index is the weighted average of these values.
        Only returned if per_cluster_scores is set to True.


    Source
    ------
    Original Code from: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/validity.py
    Their License: BSD-3-Clause license (https://github.com/scikit-learn-contrib/hdbscan/blob/master/LICENSE)
    Our modifications:
        - Add fix to also handle labelings that are not continues and/or start at zero
        - Noise handling options

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    Link: https://epubs.siam.org/doi/abs/10.1137/1.9781611973440.96
    """

    labels, X, _ = handle_noise(labels, strategy=noise_strategy, X=X)
    labels[labels != -1] = LabelEncoder().fit_transform(labels[labels != -1])
    X, labels = _check_length_data_and_labels(X, labels)
    assert isinstance(labels, np.ndarray), "labels must be of type np.ndarray. Your input has type {0}".format(
        type(labels)
    )

    return validity_index(
        X,
        labels,
        metric=metric,
        d=d,
        per_cluster_scores=per_cluster_scores,
        mst_raw_dist=mst_raw_dist,
        verbose=verbose,
        **kwd_args,
    )

"""
@authors:
Pascal Weber
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from clustpy.utils.dctree import DCTree
from clustpy.metrics._metrics_utils import handle_noise


def disco_score(X: np.ndarray, labels: np.ndarray, min_points: int = 5, noise_strategy: str = "keep") -> float:
    """Compute the mean DISCO score of all samples.

    The DISCO score is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    DISCO score are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    Additionally, the DISCO score measures noise in relation to the given
    clustering. If noise lays within a sparse region and is far away from
    cluster regions it scores a high value, otherwise it will get a low value.

    The DISCO score is calculated using the mean intra-cluster on the
    dc-distance (``a``) and the mean nearest-cluster on the dc-distance (``b``)
    for each non noise sample.  The DISCO score for a non noise sample is
    ``(b - a) / max(a, b)``.
    To clarify, ``b`` is the dc-distance between a non noise sample and the nearest
    cluster that the sample is not a part of.
    ``-1`` in labels are considered as noise and their DISCO score is calculated
    by the minimum of the two different measures ``p_sparse`` and ``p_far``.
    ``p_sparse`` measures how well the noise sample is within a sparse region.
    ``p_far`` measure how well the noise is remote to non noise samples.
    Note that DISCO score is defined for all possible number of labels
    ``1 <= n_labels <= n_samples``.
    Except for ``-1`` every other value is considered as cluster label.

    This function returns the mean DISCO score over all samples.
    To obtain the values for each sample, use :func:`disco_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : {array-like, sparse matrix} of (n_samples_a, n_features)
        A feature array.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    min_points : int
        ``min_points`` value to use for the dc-distance.
    noise_strategy : str
        Strategy for handling noise. Must be one of:
        - "keep"               : Keep all noise points as they are (default).
        - "as_one_cluster"     : Assign all noise points to a single new cluster.
        - "as_singletons"      : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points.
        - "to_nearest_cluster" : Assign each noise point to nearest cluster.

    Returns
    -------
    disco_score : float
        Mean DISCO score for all samples.

    Example
    -------
    >>> from disco import disco_score
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import HDBSCAN
    >>> X, y = make_moons(random_state=42)
    >>> hdbscan = HDBSCAN()
    >>> labels = hdbscan.fit_predict(X).labels_
    >>> disco_score(X, labels)

    References
    ----------
    Internal Evaluation of Density-Based Clusterings with Noise
    Anna Beer, Lena Krieger, Pascal Weber, Martin Ritzert, Ira Assent, Claudia Plant
    International Conference on Learning Representations (ICLR), Rio de Janeiro, Brazil, 2026.
    """

    return np.mean(disco_samples(X, labels, min_points, noise_strategy=noise_strategy))


def disco_samples(X: np.ndarray, labels: np.ndarray, min_points: int = 5, noise_strategy: str = "keep") -> np.ndarray:
    """Compute the DISCO score for each sample.

    The DISCO score is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    DISCO score are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    Additionally, the DISCO score measures noise in relation to the given
    clustering. If noise lays within a sparse region and is far away from
    cluster regions it scores a high value, otherwise it will get a low value.

    The DISCO score is calculated using the mean intra-cluster on the
    dc-distance (``a``) and the mean nearest-cluster on the dc-distance (``b``)
    for each non noise sample.  The DISCO score for a non noise sample is
    ``(b - a) / max(a, b)``.
    To clarify, ``b`` is the dc-distance between a non noise sample and the nearest
    cluster that the sample is not a part of.
    ``-1`` in labels are considered as noise and their DISCO score is calculated
    by the minimum of the two different measures ``p_sparse`` and ``p_far``.
    ``p_sparse`` measures how well the noise sample is within a sparse region.
    ``p_far`` measure how well the noise is remote to non noise samples.
    Note that DISCO score is defined for all possible number of labels
    ``1 <= n_labels <= n_samples``.
    Except for ``-1`` every other value is considered as cluster label.

    This function returns the DISCO score for each sample.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : {array-like, sparse matrix} of (n_samples_a, n_features)
        A feature array.
    labels : array-like of shape (n_samples,)
        Label values for each sample.
    min_points : int
        ``min_points`` value to use for the dc-distance.
    noise_strategy : str
        Strategy for handling noise. Must be one of:
        - "keep"               : Keep all noise points as they are (default).
        - "as_one_cluster"     : Assign all noise points to a single new cluster.
        - "as_singletons"      : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points.
        - "to_nearest_cluster" : Assign each noise point to nearest cluster.

    Returns
    -------
    disco_score : array-like of shape (n_samples,)
        DISCO scores for each sample.

    Example
    -------
    >>> from disco import disco_score
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import HDBSCAN
    >>> X, y = make_moons(random_state=42)
    >>> hdbscan = HDBSCAN()
    >>> labels = hdbscan.fit_predict(X).labels_
    >>> disco_samples(X, labels)

    References
    ----------
    Internal Evaluation of Density-Based Clusterings with Noise
    Anna Beer, Lena Krieger, Pascal Weber, Martin Ritzert, Ira Assent, Claudia Plant
    International Conference on Learning Representations (ICLR), Rio de Janeiro, Brazil, 2026.
    """

    if len(X) == 0:
        raise ValueError("Can't calculate DISCO score for empty dataset.")
    if len(X) != len(labels):
        raise ValueError("Dataset size differs from label size.")

    labels, X = handle_noise(labels, strategy=noise_strategy, X=X)
    labels[labels != -1] = LabelEncoder().fit_transform(labels[labels != -1])

    # Labels needs to be a one dimensional vector
    labels = np.reshape(labels, -1)
    label_set = set(labels)

    # Only noise
    if label_set == {-1}:
        return np.full(len(X), -1)

    # One cluster without noise
    if len(label_set) == 1 and label_set != {-1}:
        return np.full(len(X), 0)

    dc_dists = dc_distances(X, min_points=min_points)
    # One cluster with noise
    if len(label_set) == 2 and -1 in label_set:
        l_ = labels.copy()
        l_[l_ == -1] = np.arange(-1, -len(l_[l_ == -1]) - 1, -1)
        disco_values = np.empty(len(X))
        disco_values[labels != -1] = p_cluster(dc_dists, l_, precomputed_dc_dists=True)[labels != -1]
        disco_values[labels == -1] = np.minimum(*p_noise(X, labels, min_points=min_points, dc_dists=dc_dists))
        return disco_values

    # More then one cluster with optional noise
    else:
        disco_values = np.empty(len(X))
        non_noise_dc_dists = dc_dists[np.ix_(labels != -1, labels != -1)]
        non_noise_labels = labels[labels != -1]
        disco_values[labels != -1] = p_cluster(non_noise_dc_dists, non_noise_labels, precomputed_dc_dists=True)
        disco_values[labels == -1] = np.minimum(*p_noise(X, labels, min_points=min_points, dc_dists=dc_dists))
        return disco_values


def p_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    min_points: int = 5,
    precomputed_dc_dists: bool = False,
) -> np.ndarray:
    """Compute p_cluster of all samples.

    p_cluster is the Silhouette Coefficient over the dc-distance metric.
    Contrary to the Silhouette Coefficient, it is definded for
    ``1 <= n_labels <= n_samples``.

    For ``n_labels == 1`` or `` ``n_labels == n_samples`` it will return
    ``np.zeros(n_labels)``.

    In this function, ``-1`` is NOT handled as noise, but as a valid cluster label!

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : {array-like, sparse matrix} of (n_samples_a, n_features)
        A feature array.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    min_points : int
        ``min_points`` value to use for the dc-distance.
    precomputed_dc_dists : bool
        Use X as dc-distance matrix if True, else calculate dc-distance for data ``X``.
        

    Returns
    -------
    p_cluster : array-like of shape (n_samples,)
        p_cluster scores for each sample.

    Example
    -------
    >>> from disco import p_cluster
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import HDBSCAN
    >>> X, y = make_moons(random_state=42)
    >>> hdbscan = HDBSCAN()
    >>> labels = hdbscan.fit_predict(X).labels_
    >>> p_cluster(X, labels)
    """
    if len(X) != len(labels):
        raise ValueError("Dataset size of `X` differs from label size of `lables`.")

    if len(X) == 0:
        return np.array([])

    if len(X) == 1:
        return np.array([0])

    if 1 == len(set(labels)) or len(set(labels)) == len(X):
        return np.zeros(len(X))

    if precomputed_dc_dists:
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("`X` needs to be a distance matrix if `precomputed_dc_dists` is `True`.")
        dc_dists = X
    else:
        dc_dists = dc_distances(X, min_points=min_points)

    return silhouette_samples(dc_dists, labels, metric="precomputed")


def p_noise(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    min_points: int = 5,
    dc_dists: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (p_sparse, p_far) of all samples.

    ``p_sparse`` calculates how well the noise sample lays within a sparse region.
    ``p_far`` calulcates how well the noise is remote remote to a non noise sample.
    To clarify, ``p_sparse`` and ``p_far`` are calculated depending on the existing
    clustering. Changing the clustering without changing the noise samples can change
    the values of ``p_sparse`` and ``p_far``.

    In this function, ``-1`` is handled as noise. The return value consists of a
    tuple with two arrays both of size ``n_noise``.

    The best value is 1 and the worst value is -1 for both ``p_sparse`` and ``p_far``.
    For ``p_sparse``, values near 0 indicate that the noise sample is within an area which
    is as sparse as the sparsest existing cluster. Note that this value is calculated
    depending on the sparsest cluster.  Negative values indicate that the noise sample has
    been labeled as noise, although it lays within a very dense region.
    For ``p_far``, values near 0 indicate that the noise sample lays at the border of
    an existing cluster. Negative values generally indicate that a noise sample lays within
    an existing cluster.

    Parameters
    ----------
    X : {array-like, sparse matrix} of (n_samples_a, n_features)
        A feature array.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    min_points : int
        ``min_points`` value to use for the dc-distance.
    dc_dists : array-like of shape (n_samples,)
        Precalculated dc-distances. If not provided, dc-distances will be calculated for data ``X``.

    Returns
    -------
    (p_sparse, p_far) : tuple of two array-like, both of shape (n_noise,)
        (p_sparse, p_far) for each sample, returned in two seperate arrays.

    Example
    -------
    >>> from disco import p_noise
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import HDBSCAN
    >>> X, y = make_moons(random_state=42)
    >>> hdbscan = HDBSCAN()
    >>> labels = hdbscan.fit_predict(X).labels_
    >>> p_noise(X, labels)
    """
    if len(X) == 0:
        raise ValueError("Can't calculate noise score for empty dataset.")
    if len(X) != len(labels):
        raise ValueError("Dataset size differs from label size.")

    label_set = set(labels)

    # Only noise
    if label_set == {-1}:
        return np.full(len(X), -1), np.full(len(X), -1)

    # No noise
    if -1 not in label_set:
        return np.array([]), np.array([])

    ## At least one cluster and noise ##
    if dc_dists is None:
        dc_dists = dc_distances(X, min_points=min_points)

    tree = KDTree(X)
    core_dists, _ = tree.query(X, k=min_points)
    core_dists = core_dists.max(axis=1)

    # Get maximum core distance per cluster
    cluster_ids = set(labels[labels != -1])
    max_core_dist = np.empty(len(cluster_ids))
    for i, cluster_id in enumerate(cluster_ids):
        max_core_dist[i] = core_dists[labels == cluster_id].max()

    # p_sparse calculation
    p_sparse = np.full(len(labels[labels == -1]), np.inf)
    for i in range(len(cluster_ids)):
        numerator = core_dists[labels == -1] - max_core_dist[i]
        denominator = np.maximum(core_dists[labels == -1], max_core_dist[i])
        p_sparse_i = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        p_sparse = np.minimum(p_sparse, p_sparse_i)

    # p_far calculation
    p_far = np.full(len(labels[labels == -1]), np.inf)
    for i, cluster_id in enumerate(cluster_ids):
        min_dist_to_cluster_i = np.min(dc_dists[np.ix_(labels == -1, labels == cluster_id)], axis=1)
        numerator = min_dist_to_cluster_i - max_core_dist[i]
        denominator = np.maximum(min_dist_to_cluster_i, max_core_dist[i])
        p_far_i = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        p_far = np.minimum(p_far, p_far_i)

    return p_sparse, p_far


def dc_distances(X, min_points=5):
    dc_dists = DCTree(X, min_points=min_points).dc_distances()
    return dc_dists

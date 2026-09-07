from sklearn.datasets import make_blobs
from scipy.stats import special_ortho_group
import numpy as np
from sklearn.utils import shuffle
from clustpy.utils.checks import check_random_state


def create_subspace_data(n_samples: int | list[int] | tuple[int, ...] = 1000, n_clusters: int = 3, subspace_features: int | tuple[int, int] | list[int] = (2, 2),
                         n_outliers: tuple[int, int] | list[int] = (0, 0), std: float | list[float] | tuple[float, float] = 1.,
                         box: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] | list[tuple[float, float]] = (-10., 10.), rotate_space: bool = True,
                         random_state: np.random.RandomState | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic subspace data set consisting of a subspace containing multiple Gaussian clusters (called
    clustered space) and a subspace containing a single Gaussian cluster (called noise space).
    This method is a special case of the create_nr_data method using only a single clustered space.
    See create_nr_data for more information.

    Parameters
    ----------
    n_samples : int | list[int] | tuple[int, ...]
        Number of samples in the clusters. If n_samples is int, the samples will be equally divided across all clusters.
        Otherwise, a tuple (e.g. (100, 200, 700)) can specify the size of each cluster individually (default: 1000)
    n_clusters : int
        Specifies the number of clusters in the clustered space (default: 3)
    subspace_features : int | tuple[int, int] | list[int]
        Number of features in each of the two subspaces (default: (2, 2))
    n_outliers : tuple[int, int] | list[int]
        Number of outliers for each subspace. Overall number of samples will be n_samples + n_outliers.
        Beware that n_samples + n_outliers must be equal for both subspaces (default: (0, 0))
    std : float | list[float] | tuple[float, float]
        Standard deviation of the Gaussian clusters. Can be a list specifying an individual value for each subspace (default: 1.)
    box : tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] | list[tuple[float, float]]
        The bounding box of the cluster centers. Can be a list specifying an individual value for each subspace (default: (-10., 10.))
    rotate_space : bool
        Specifies whether the feature space should be rotated by an orthonormal matrix (default: True)
    random_state: np.random.RandomState | int | None
        The random state (default: None)

    Returns
    -------
    data, labels : tuple[np.ndarray, np.ndarray]
        the data numpy array (n_samples x sum(subspace_features)), the labels numpy array (n_samples)
    """
    assert type(n_clusters) is int, "n_clusters must be of type int"
    if isinstance(n_outliers, (int, np.integer)):
        n_outliers = [n_outliers, 0]
    assert isinstance(n_outliers, (list, tuple)) and len(n_outliers) == 2, "n_outliers must be a list or tuple of length 2"
    if isinstance(n_samples, (list, tuple)):
        n_samples_adj = [n_samples, np.sum(n_samples) + n_outliers[0] - n_outliers[1]]
    else:
        n_samples_adj = [n_samples - n_outliers[0], n_samples - n_outliers[1]]
    assert isinstance(n_samples_adj, (list, tuple)) and len(n_samples_adj) == 2, "n_samples must be a list or tuple of length 2"
    X, L = create_nr_data(n_samples=n_samples_adj, n_clusters=(n_clusters, 1), subspace_features=subspace_features,
                          n_outliers=n_outliers, std=std, box=box, rotate_space=rotate_space,
                          random_state=random_state)
    return X, L[:, 0].astype(np.int32)


def create_nr_data(n_samples: int | list[int] | tuple[int, ...] | tuple[tuple[int, ...], ...] | list[list[int]] = 1000,
                   n_clusters: tuple[int, ...] | list[int] = (3, 3, 1), subspace_features: int | tuple[int, ...] | list[int] = (2, 2, 2),
                   n_outliers: tuple[int, ...] | list[int] = (0, 0, 0), std: float | list[float] | tuple[float, ...] = 1.,
                   box: tuple[float, float] | tuple[tuple[float, float], ...] | list[tuple[float, float]] = (-10., 10.), rotate_space: bool = True,
                   random_state: np.random.RandomState | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic non-redundant data set consisting of multiple subspaces containing Gaussian clusters (called
    clustered spaces). You can also create subspaces with a single Gaussian cluster (called noise space).
    The sklearn method make_blobs is used to create the clusters. The dimensionality of the subspaces is specified by
    the subspace_features parameter. It can be an integer, where the dimensionality is the same for all subspaces,
    or it can be a list.
    Additionally, one can specify the number of outliers for each subspace. Outliers will be created using a uniform
    distribution using the box parameter as limits. If outliers are used, the number of samples within the clusters is
    reduced accordingly.
    The standard deviation and the bounding box can be specified either for each subspace individually or a single value
    will be shared across all spaces.

    Parameters
    ----------
    n_samples : int | list[int] | tuple[int, ...] | tuple[tuple[int, ...], ...] | list[list[int]]
        Number of samples in the clusters. If n_samples is int, the samples will be equally divided across all clusters in each subspace.
        Otherwise, a tuple of tuples (e.g. ((100, 200, 700), (300,300,400), (300,300,400))) can specify the size of each cluster in each subspace individually.
        Beware that the overall number of samples (including outliers) must be equal for each subspace (default: 1000)
    n_clusters : tuple[int, ...] | list[int]
        Specifies the number of clusters for each subspace (default: (3, 3, 1))
    subspace_features : int | tuple[int, ...] | list[int]
        Number of features in each subspace (default: (2, 2, 2))
    n_outliers : tuple[int, ...] | list[int]
        Number of outliers for each subspace. Overall number of samples will be n_samples + n_outliers.
        Beware that n_samples + n_outliers must be equal for each subspace (default: (0, 0, 0))
    std : float | list[float] | tuple[float, ...]
        Standard deviation of the Gaussian clusters. Can be a list specifying an individual value for each subspace (default: 1.)
    box : tuple[float, float] | tuple[tuple[float, float], ...] | list[tuple[float, float]]
        The bounding box of the cluster centers. Can be a list specifying an individual value for each subspace (default: (-10., 10.))
    rotate_space : bool
        Specifies whether the feature space should be rotated by an orthonormal matrix (default: True)
    random_state: np.random.RandomState | int | None
        The random state (default: None)

    Returns
    -------
    data, labels : tuple[np.ndarray, np.ndarray]
        the data numpy array (n_samples x sum(subspace_features)), the labels numpy array (n_samples x len(subspace_features))
    """
    random_state = check_random_state(random_state)
    # Transform n_clusters to list
    if not isinstance(n_clusters, (list, tuple)):
        n_clusters = [n_clusters]
    # Transform n_outliers to list
    if not isinstance(n_outliers, (list, tuple)):
        n_outliers = [n_outliers] * len(n_clusters)
    assert len(n_clusters) == len(n_outliers), "inconsistent number of subspaces between n_clusters and n_outliers"
    # Transform n_samples to list
    if not isinstance(n_samples, (list, tuple)): # n_samples is int
        # Beware the outliers per subspace
        n_samples = [n_samples + n_outliers[0] - n_outliers[i] for i in range(len(n_clusters))]
    assert len(n_clusters) == len(
        n_samples), "inconsistent number of subspaces between n_clusters and n_samples"
    overall_samples = np.sum(n_samples[0]) + n_outliers[0]
    assert all([np.sum(n_samples[i]) + n_outliers[i] == overall_samples for i in
                range(len(n_clusters))]), "samples in each subspace must be equal (sum of cluster objects and outliers)"
    for i in range(len(n_clusters)):
        n_samples_sub = n_samples[i]
        if isinstance(n_samples_sub, (list, tuple)):
            assert len(n_samples_sub) == n_clusters[i], "number of clusters in n_samples does not match n_clusters"
        else:
            assert isinstance(n_samples_sub, (int, np.integer)), "n_samples must be int"
    # Transform cluster_features to list
    if not isinstance(subspace_features, (list, tuple)):
        subspace_features = [subspace_features] * len(n_clusters)
    assert len(n_clusters) == len(
        subspace_features), "inconsistent number of subspaces between n_clusters and subspace_features"
    # Transform std to list
    if not isinstance(std, (list, tuple)):
        std = [std] * len(n_clusters)
    assert len(n_clusters) == len(std), "inconsistent number of subspaces between n_clusters and std"
    # Transform box to list
    if not isinstance(box, (list, tuple)):
        raise Exception("Each entry of the tuple box must contain two values (upper and lower bound)")
    if len(box) == 2 and isinstance(box[0], (float, int)) and isinstance(box[1], (float, int)):
        box = [(float(box[0]), float(box[1]))] * len(n_clusters)
    assert len(n_clusters) == len(box), "inconsistent number of subspaces between n_clusters and box"
    # Create empty dataset
    X, L = np.empty((overall_samples, 0)), np.empty((overall_samples, 0), dtype=np.int32)
    for i in range(len(n_clusters)):
        # Create Clusters
        n_samples_sub = n_samples[i]
        if isinstance(n_samples_sub, tuple):
            n_samples_sub = list(n_samples_sub)
        centers_sub = None if isinstance(n_samples_sub, list) else n_clusters[i]
        X_tmp, L_tmp = make_blobs(n_samples_sub, subspace_features[i], centers=centers_sub,
                                  cluster_std=std[i], center_box=box[i], random_state=random_state)
        # Create outliers
        if n_outliers[i] != 0:
            X_out = random_state.random((n_outliers[i], subspace_features[i]))
            tmp_box = box[i]
            assert isinstance(tmp_box, (tuple, list)) and len(tmp_box) == 2, "Each entry of the tuple box must contain two values (upper and lower bound)"
            out_box = (tmp_box[0] - 4 * std[i], tmp_box[1] + 4 * std[i])
            X_out = X_out * (out_box[1] - out_box[0]) + out_box[0]
            L_out = np.array([-1] * n_outliers[i])
            X_tmp = np.r_[X_tmp, X_out]
            L_tmp = np.r_[L_tmp, L_out]
            # Shuffle data so outliers wont be always in the last positions
            X_tmp, L_tmp = shuffle(X_tmp, L_tmp, random_state=random_state)
        # Add subspace to dataset
        X = np.c_[X, X_tmp]
        L = np.c_[L, L_tmp]
    L = L.astype(np.int32)
    # Rotate space
    if rotate_space:
        V = special_ortho_group.rvs(dim=sum(subspace_features), random_state=random_state)
        X = np.matmul(X, V)
    return X, L

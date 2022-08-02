from sklearn.datasets import make_blobs
from scipy.stats import special_ortho_group
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import check_random_state


def create_subspace_data(n_samples: int = 1000, n_clusters: int = 3, subspace_features: tuple = (2, 2),
                         n_outliers: tuple = (0, 0), std: float = 1., box: tuple = (-10, 10), rotate_space: bool = True,
                         random_state: int = None) -> (np.ndarray, np.ndarray):
    """
    Create a synthetic subspace data set consisting of a subspace containing multiple Gaussian clusters (called
    clustered space) and a subspace containing a single Gaussian cluster (called noise space).
    This method is a special case of the create_nr_data method using only a single clustered space.
    See create_nr_data for more information.

    Parameters
    ----------
    n_samples : int
        Number of samples (default: 1000)
    n_clusters : int
        Specifies the number of clusters in the clustered space (default: 3)
    subspace_features : tuple
        Number of features in the two subspaces (default: (2, 2))
    n_outliers : tuple
        Number of outliers for each subspace. Number of samples within the clusters will be lowered by this value (default: (0, 0))
    std : float
        Standard deviation of the Gaussian clusters (default: 1.)
    box : tuple
        The bounding box of the cluster centers (default: (-10, 10))
    rotate_space : bool
        Specifies whether the feature space should be rotated by an orthonormal matrix (default: True)
    random_state: int / np.random.RandomState
        The random state (default: None)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (n_samples x sum(subspace_features)), the labels numpy array (n_samples)
    """
    assert type(n_clusters) is int, "n_clusters must be of type int"
    X, L = create_nr_data(n_samples=n_samples, n_clusters=(n_clusters, 1), subspace_features=subspace_features,
                          n_outliers=n_outliers, std=std, box=box, rotate_space=rotate_space,
                          random_state=random_state)
    return X, L[:, 0]


def create_nr_data(n_samples: int = 1000, n_clusters: tuple = (3, 3, 1), subspace_features: tuple = (2, 2, 2),
                   n_outliers: tuple = (0, 0, 0), std: float = 1., box: tuple = (-10, 10), rotate_space: bool = True,
                   random_state: int = None) -> (np.ndarray, np.ndarray):
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
    n_samples : int
        Number of samples (default: 1000)
    n_clusters : tuple
        Specifies the number of clusters for each subspace (default: (3, 3, 1))
    subspace_features : tuple
        Number of features in each subspace (default: (2, 2, 2))
    n_outliers : tuple
        Number of outliers for each subspace. Number of samples within the clusters will be lowered by this value (default: (0, 0, 0))
    std : float
        Standard deviation of the Gaussian clusters (default: 1.)
    box : tuple
        The bounding box of the cluster centers (default: (-10, 10))
    rotate_space : bool
        Specifies whether the feature space should be rotated by an orthonormal matrix (default: True)
    random_state: int / np.random.RandomState
        The random state (default: None)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (n_samples x sum(subspace_features)), the labels numpy array (n_samples x len(subspace_features))
    """
    random_state = check_random_state(random_state)
    # Transform n_clusters to list
    if type(n_clusters) is not list and type(n_clusters) is not tuple:
        n_clusters = [n_clusters]
    # Transform cluster_features to list
    if type(subspace_features) is not list and type(subspace_features) is not tuple:
        subspace_features = [subspace_features] * len(n_clusters)
    assert len(n_clusters) == len(
        subspace_features), "inconsistent number of subspaces between n_clusters and subspace_features"
    # Transform n_outliers to list
    if type(n_outliers) is not list and type(n_outliers) is not tuple:
        n_outliers = [n_outliers] * len(n_clusters)
    assert len(n_clusters) == len(n_outliers), "inconsistent number of subspaces between n_clusters and n_outliers"
    # Transform std to list
    if type(std) is not list and type(std) is not tuple:
        std = [std] * len(n_clusters)
    assert len(n_clusters) == len(std), "inconsistent number of subspaces between n_clusters and std"
    # Transform box to list
    if type(box) is not list and type(box) is not tuple:
        raise Exception("Each entry of the tuple box must contain two values (upper and lower bound)")
    if type(box[0]) is not list and type(box[0]) is not tuple:
        box = [box] * len(n_clusters)
    assert len(n_clusters) == len(box), "inconsistent number of subspaces between n_clusters and box"
    # Create empty dataset
    X, L = np.empty((n_samples, 0)), np.empty((n_samples, 0), dtype=np.int32)
    for i in range(len(n_clusters)):
        # Create Clusters
        X_tmp, L_tmp = make_blobs(n_samples - n_outliers[i], subspace_features[i], centers=n_clusters[i],
                                  cluster_std=std[i], center_box=box[i], random_state=random_state)
        # Create outliers
        if n_outliers[i] != 0:
            X_out = random_state.random((n_outliers[i], subspace_features[i]))
            out_box = (box[i][0] - 4 * std[i], box[i][1] + 4 * std[i])
            X_out = X_out * (out_box[1] - out_box[0]) + out_box[0]
            L_out = np.array([-1] * n_outliers[i])
            X_tmp = np.r_[X_tmp, X_out]
            L_tmp = np.r_[L_tmp, L_out]
            # Shuffle data so outliers wont be always in the last positions
            X_tmp, L_tmp = shuffle(X_tmp, L_tmp, random_state=random_state)
        # Add subspace to dataset
        X = np.c_[X, X_tmp]
        L = np.c_[L, L_tmp]
    # Rotate space
    if rotate_space:
        V = special_ortho_group.rvs(dim=sum(subspace_features), random_state=random_state)
        X = np.matmul(X, V)
    return X, L

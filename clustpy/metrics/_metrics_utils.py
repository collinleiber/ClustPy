import numpy as np
from sklearn.metrics.cluster._supervised import check_clusterings
from sklearn.utils import check_X_y
from scipy.spatial import cKDTree


def _check_labels_arrays(labels_true: np.ndarray, labels_pred: np.ndarray, allow_2d_labels: bool = False) -> (np.ndarray, np.ndarray):
    """
    Check that the ground truth labels and the prediction labels are compatible.
    If they do not match throw an exception.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm
    allow_2d_labels: bool
        Specifies whether 2d labels (multiple label sets) are allowed (default: False)

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The ground truth labels,
        The predicted labels
    """
    labels_true = np.asarray(labels_true).astype(int)
    labels_pred = np.asarray(labels_pred).astype(int)

    if labels_true.ndim == 1 and labels_pred.ndim == 1:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    elif allow_2d_labels:
        true_ref = labels_true[:, 0].copy() if labels_true.ndim > 1 else labels_true.copy()
        pred_ref = labels_pred[:, 0].copy() if labels_pred.ndim > 1 else labels_pred.copy()
        if labels_true.ndim > 1:
            labels_true = labels_true.copy()
            for i in range(labels_true.shape[1]):
                # Align each column of 'true' against the reference of 'pred'
                labels_true[:, i], _ = check_clusterings(labels_true[:, i], pred_ref)
        else:
            labels_true, _ = check_clusterings(labels_true, pred_ref)
        if labels_pred.ndim > 1:
            labels_pred = labels_pred.copy()
            for i in range(labels_pred.shape[1]):
                # Align each column of 'pred' against the (now potentially updated) 'true' reference
                _, labels_pred[:, i] = check_clusterings(true_ref, labels_pred[:, i])
        else:
            _, labels_pred = check_clusterings(true_ref, labels_pred)
    else:
        raise ValueError(f"Your labels are not 1d arrays. Shape of labels_true: {labels_true.shape}, shape of labels_pred: {labels_pred.shape}")
    return labels_true, labels_pred


def _check_length_data_and_labels(X: np.ndarray, labels: np.ndarray, allow_single_cluster: bool = False) -> (np.ndarray, np.ndarray):
    """
    Check that the data and the prediction labels are compatible.
    If they do not match throw an exception.

    Parameters
    ----------
    X : np.ndarray
        The data set
    labels : np.ndarray
        The labels as predicted by a clustering algorithm
    allow_single_cluster : bool
        Allow a single cluster within the labels (default: False)

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The data set,
        The predicted labels
    """
    X, labels = check_X_y(X, labels)
    labels = labels.astype(int)
    n_pred_clusters = len(np.unique(labels))
    if (not allow_single_cluster and n_pred_clusters == 1) or n_pred_clusters == X.shape[0]:
        raise ValueError("The number of different labels must be within [2, n_samples - 1]")
    return X, labels

# ============================================================
# Noise handling strategies
# ============================================================

def _assign_noise_points_to_one_cluster(labels: np.ndarray) -> np.ndarray:
    """
    Assign all noise points (label = -1) to a single new cluster.
    This function replaces all occurrences of -1 with a new cluster label,
    which is chosen as (max existing label + 1).

    Parameters
    ----------
    labels : np.ndarray
        1D array of cluster labels. Noise points must be labeled as -1.

    Returns
    -------
    new_labels : np.ndarray
        Copy of input labels where all noise points are assigned to one
        additional cluster. If no noise points exist, the input is returned unchanged.
    """
    new_labels = labels.copy()
    noise_mask = new_labels == -1
    if not np.any(noise_mask):
        return new_labels
    max_label = new_labels[new_labels >= 0].max(initial=-1)
    new_labels[noise_mask] = max_label + 1
    return new_labels


def _assign_noise_points_to_singletons(labels: np.ndarray) -> np.ndarray:
    """
    Assign each noise point (label = -1) to its own unique cluster.
    Each noise point becomes a singleton cluster with a unique label.
    New labels are assigned sequentially starting from (max existing label + 1).

    Parameters
    ----------
    labels : np.ndarray
        1D array of cluster labels. Noise points must be labeled as -1.

    Returns
    -------
    new_labels : np.ndarray
        Copy of input labels where each noise point is assigned a unique
        cluster label.
    """
    new_labels = labels.copy()
    noise_indices = np.where(new_labels == -1)[0]
    if len(noise_indices) == 0:
        return new_labels
    max_label = new_labels[new_labels >= 0].max(initial=-1)
    new_labels[noise_indices] = np.arange(len(noise_indices)) + max_label + 1
    return new_labels


def _remove_noise_points(labels: np.ndarray) -> np.ndarray:
    """
    Remove all noise points (label = -1) from the label array.
    This function filters out all entries labeled as -1. The resulting array
    is shorter than the input.

    Parameters
    ----------
    labels : np.ndarray
        1D array of cluster labels. Noise points must be labeled as -1.

    Returns
    -------
    new_labels : np.ndarray
        Array containing only non-noise labels (labels >= 0).
        Note: This changes the length of the array.

    non_noise_indices : np.ndarray
        The indices of the non-noise points.
    """
    non_noise_indices = np.where(labels >= 0)[0]
    if len(non_noise_indices) == 0:
        raise ValueError("Removing noise points would lead to an empty array: no non-noise points exist.")
    new_labels = labels[non_noise_indices]
    return new_labels, non_noise_indices


def _assign_noise_points_to_nearest_cluster(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Assign each noise point (label = -1) to the cluster of the nearest existing point.
    Each noise point is assigned to the same cluster as the closest non-noise point
    in Euclidean distance.

    Parameters
    ----------
    labels : np.ndarray
        1D array of shape (n_samples,) containing cluster labels.
        Noise points must be labeled as -1.
    X : np.ndarray
        Array of shape (n_samples, n_features) containing the data points.

    Returns
    -------
    new_labels : np.ndarray
        Copy of labels where each noise point is reassigned to the cluster of
        the nearest point.

    Raises
    ------
    ValueError
        If no non-noise points exist (all labels are -1) or X is None.
    """
    if X is None:
        raise ValueError("X must be provided for the 'nearest_cluster' strategy.")
    new_labels = labels.copy()
    # Indices of non-noise and noise points
    noise_indices = np.where(new_labels == -1)[0]
    if len(noise_indices) == 0:
        return new_labels
    non_noise_indices = np.where(labels >= 0)[0]
    if len(non_noise_indices) == 0:
        raise ValueError("Cannot assign to nearest cluster: no non-noise points exist.")
    outliers = X[noise_indices]
    non_outliers = X[non_noise_indices]
    # Build a KD-tree on non-noise points
    tree = cKDTree(non_outliers)
    # For each noise point, find nearest non-noise point
    _, nearest_idx_in_tree = tree.query(outliers)
    # Assign noise point to the cluster of its nearest non-noise point
    labels_non_outliers = new_labels[non_noise_indices]
    new_labels[noise_indices] = labels_non_outliers[nearest_idx_in_tree]
    return new_labels


# Unified Interface
def handle_noise(
    labels: np.ndarray,
    strategy: str,
    X: np.ndarray | None = None,
    labels_compare: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Handle noise points (label = -1) in clustering results using a specified strategy.
    If X is provided, also return the adapted X (e.g., rows corresponding to
    removed noise points are removed).

    Parameters
    ----------
    labels : np.ndarray
        1D array of cluster labels.

    strategy : str
        Strategy for handling noise. Must be one of:
        - "keep"               : Keep all noise points as they are.
        - "one_cluster"        : Assign all noise points to a single new cluster.
        - "singletons"         : Assign each noise point to its own cluster.
        - "filter"             : Remove all noise points.
        - "nearest_cluster" : Assign each noise point to nearest cluster (requires X).

    X : np.ndarray | None
        Data matrix of shape (n_samples, n_features).
        Required for "nearest_cluster".

    labels_compare : np.ndarray | None
        Second set of labels that will be reduced if strategy is "filter" (default: None)

    Returns
    -------
    new_labels : np.ndarray
        Labels after applying the chosen strategy.
        If X is provided, also returns new_X aligned with new_labels.

    new_X : np.ndarray, optional
        Adapted X after removing noise points (is None if X was not provided).

    new_labels_compare : np.ndarray, optional
        Adapted labels_compare after removing noise points (is None if labels_compare was not provided).

    Raises
    ------
    ValueError
        If an invalid strategy is provided or required inputs are missing.
    """
    if X is not None:
        X, labels = _check_length_data_and_labels(X, labels, True)
    if labels_compare is not None:
        labels, labels_compare = _check_labels_arrays(labels, labels_compare)
    strategy = strategy.lower()
    if strategy == "keep":
        new_labels = labels
    elif strategy == "one_cluster":
        new_labels = _assign_noise_points_to_one_cluster(labels)
    elif strategy == "singletons":
        new_labels = _assign_noise_points_to_singletons(labels)
    elif strategy == "filter":
        new_labels, non_noise_indices = _remove_noise_points(labels)
        if X is not None:
            X = X[non_noise_indices]
        if labels_compare is not None:
            labels_compare = labels_compare[non_noise_indices]
    elif strategy == "nearest_cluster":
        new_labels = _assign_noise_points_to_nearest_cluster(labels, X)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Valid options: "
            "'keep', 'one_cluster', 'singletons', 'filter', 'nearest_cluster'."
        )
    return new_labels, X, labels_compare

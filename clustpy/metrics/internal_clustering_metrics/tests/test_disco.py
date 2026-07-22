import numpy as np
from clustpy.metrics.internal_clustering_metrics.disco import disco_score, disco_samples, p_cluster, p_noise
import pytest
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_blobs():
    """Two tight, well-separated 2-D blobs with 4 points each."""
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [9.0, 9.0], [9.0, 10.0], [10.0, 9.0], [10.0, 10.0]],
        dtype=float,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, labels

def _make_symmetric_dist_matrix(n):
    """Return a valid symmetric distance matrix filled with predictable values."""
    rng = np.random.default_rng(0)
    raw = rng.random((n, n))
    D = (raw + raw.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# disco_score — edge cases
# ---------------------------------------------------------------------------

def test_disco_score_empty_dataset_raises():
    with pytest.raises(ValueError, match="empty"):
        disco_score(np.empty((0, 2)), np.array([]))

def test_disco_score_length_mismatch_raises():
    X = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        disco_score(X, labels)

def test_disco_score_only_noise_returns_minus_one():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    labels = np.array([-1, -1, -1])
    scores = disco_samples(X, labels)
    assert scores.shape == (3,)
    assert np.all(scores == -1.0)
    assert disco_score(X, labels) == -1.0

def test_disco_score_single_cluster_no_noise_returns_zero():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 0])
    scores = disco_samples(X, labels)
    assert scores.shape == (3,)
    assert np.all(scores == 0.0)
    assert disco_score(X, labels) == 0.0

def test_disco_score_equals_mean_of_disco_samples():
    X, labels = _make_two_blobs()
    assert disco_score(X, labels) == pytest.approx(np.mean(disco_samples(X, labels)))

def test_disco_score_scores_in_valid_range():
    X, labels = _make_two_blobs()
    scores = disco_samples(X, labels)
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)

def test_disco_score_output_shape_matches_input():
    X, labels = _make_two_blobs()
    scores = disco_samples(X, labels)
    assert scores.shape == (len(X),)

def test_disco_score_well_separated_blobs_positive_score():
    """Two tight, well-separated clusters should yield a clearly positive mean score."""
    X, labels = _make_two_blobs()
    assert disco_score(X, labels, min_points=1) > 0.5

def test_disco_score_overlapping_clusters_lower_score():
    """Scrambled labels should yield a lower mean score than correct labels."""
    X, _ = _make_two_blobs()
    correct_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bad_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assert disco_score(X, correct_labels, min_points=1) > disco_score(X, bad_labels, min_points=1)


# ---------------------------------------------------------------------------
# disco_samples — with noise
# ---------------------------------------------------------------------------

def test_disco_samples_noise_scores_in_valid_range():
    X, labels = _make_two_blobs()
    labels_with_noise = labels.copy()
    labels_with_noise[0] = -1  # demote first point to noise
    scores = disco_samples(X, labels_with_noise)
    assert scores.shape == (len(X),)
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)


def test_disco_samples_noise_far_from_clusters_has_positive_score():
    """A noise point far from any cluster should get a positive DISCO score."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],  # cluster 0
            [9.0, 9.0],
            [9.0, 10.0],
            [10.0, 9.0],
            [10.0, 10.0],  # cluster 1
            [50.0, 50.0],
        ],  # far noise
        dtype=float,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1])
    scores = disco_samples(X, labels)
    # The remote noise point should score positively
    assert scores[-1] > 0.0

def test_disco_samples_noise_inside_cluster_has_negative_score():
    """A noise point sitting inside a dense cluster should get a negative DISCO score."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],  # cluster 0
            [9.0, 9.0],
            [9.0, 10.0],
            [10.0, 9.0],
            [10.0, 10.0],  # cluster 1
            [0.5, 0.5],
        ],  # noise at cluster 0's centroid
        dtype=float,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1])
    scores = disco_samples(X, labels)
    # The noise point inside the cluster should score negatively
    assert scores[-1] < 0.0

def test_disco_samples_single_cluster_with_noise_scores_shape():
    """One cluster + noise: result must still cover all samples."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [100.0, 100.0]], dtype=float)
    labels = np.array([0, 0, 0, -1])
    scores = disco_samples(X, labels, min_points=1)
    assert scores.shape == (4,)
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)


# ---------------------------------------------------------------------------
# p_cluster
# ---------------------------------------------------------------------------:

def test_p_cluster_empty_returns_empty():
    result = p_cluster(np.empty((0, 2)), np.array([]))
    assert result.shape == (0,)

def test_p_cluster_single_sample_returns_zero():
    result = p_cluster(np.array([[1.0, 2.0]]), np.array([0]))
    assert result == pytest.approx(np.array([0.0]))

def test_p_cluster_all_same_label_returns_zeros():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 0])
    result = p_cluster(X, labels)
    assert np.all(result == 0.0)

def test_p_cluster_each_own_label_returns_zeros():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 2])
    result = p_cluster(X, labels)
    assert np.all(result == 0.0)

def test_p_cluster_length_mismatch_raises():
    with pytest.raises(ValueError):
        p_cluster(np.array([[0, 0], [1, 1]]), np.array([0, 0, 1]))

def test_p_cluster_precomputed_matches_sklearn_silhouette():
    """With a precomputed distance matrix p_cluster must equal sklearn's silhouette_samples."""
    n = 6
    D = _make_symmetric_dist_matrix(n)
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = p_cluster(D, labels, precomputed_dc_dists=True)
    expected = silhouette_samples(D, labels, metric="precomputed")
    np.testing.assert_allclose(result, expected)

def test_p_cluster_precomputed_invalid_matrix_raises():
    """Non-square matrix with precomputed=True must raise."""
    with pytest.raises(ValueError):
        p_cluster(np.zeros((3, 4)), np.array([0, 0, 1]), precomputed_dc_dists=True)

def test_p_cluster_output_range():
    X, labels = _make_two_blobs()
    result = p_cluster(X, labels)
    assert result.shape == (len(X),)
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)

def test_p_cluster_well_separated_blobs_high_score():
    X, labels = _make_two_blobs()
    result = p_cluster(X, labels, min_points=2)
    p_cluster_values = np.array(
        [0.91161165, 0.91161165, 0.91161165, 0.91161165, 0.91161165, 0.91161165, 0.91161165, 0.91161165]
    )
    np.testing.assert_allclose(result, p_cluster_values)


# ---------------------------------------------------------------------------
# p_noise
# ---------------------------------------------------------------------------

def test_p_noise_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        p_noise(np.empty((0, 2)), np.array([]))

def test_p_noise_length_mismatch_raises():
    with pytest.raises(ValueError):
        p_noise(np.array([[0, 0], [1, 1]]), np.array([0, 0, -1]))

def test_p_noise_only_noise_returns_minus_one():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    labels = np.array([-1, -1, -1])
    p_sparse, p_far = p_noise(X, labels)
    assert np.all(p_sparse == -1.0)
    assert np.all(p_far == -1.0)

def test_p_noise_no_noise_returns_empty_arrays():
    X, labels = _make_two_blobs()
    p_sparse, p_far = p_noise(X, labels)
    assert len(p_sparse) == 0
    assert len(p_far) == 0

def test_p_noise_output_count_matches_noise_count():
    """p_noise output size must equal the number of -1 labels."""
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [10.0, 9.0], [50.0, 50.0]],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, -1])
    p_sparse, p_far = p_noise(X, labels)
    n_noise = (labels == -1).sum()
    assert p_sparse.shape == (n_noise,)
    assert p_far.shape == (n_noise,)

def test_p_noise_output_range():
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [10.0, 9.0], [50.0, 50.0]],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, -1])
    p_sparse, p_far = p_noise(X, labels)
    assert np.all(p_sparse >= -1.0) and np.all(p_sparse <= 1.0)
    assert np.all(p_far >= -1.0) and np.all(p_far <= 1.0)

def test_p_noise_precomputed_dc_dists_gives_same_result():
    """Passing precomputed dc_dists must yield identical output to computing them internally."""
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [10.0, 9.0], [50.0, 50.0]],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, -1])
    dc_dists = np.array([
        [0.0, 70.71067812, 70.71067812, 70.71067812, 70.71067812],
        [70.71067812, 0.0, 70.00714249, 70.00714249, 70.71067812],
        [70.71067812, 70.00714249, 0.0, 57.98275606, 70.71067812],
        [70.71067812, 70.00714249, 57.98275606, 0.0, 70.71067812],
        [70.71067812, 70.71067812, 70.71067812, 70.71067812, 0.0],
    ])
    p_sparse_pre, p_far_pre = p_noise(X, labels, dc_dists=dc_dists)
    p_sparse_calc, p_far_calc = p_noise(X, labels)
    np.testing.assert_allclose(p_sparse_pre, p_sparse_calc, rtol=0, atol=1e-10)
    np.testing.assert_allclose(p_far_pre, p_far_calc, rtol=0, atol=1e-10)

def test_p_noise_far_noise_higher_p_far_than_nearby_noise():
    """A noise point far from all clusters must have higher p_far than one nearby."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],  # cluster 0
            [9.0, 9.0],
            [9.0, 10.0],
            [10.0, 9.0],
            [10.0, 10.0],  # cluster 1
            [0.6, 0.6],  # noise close to cluster 0
            [50.0, 50.0],
        ],  # noise far from everything
        dtype=float,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    _, p_far = p_noise(X, labels)
    # p_far index 0 = close noise, index 1 = far noise
    assert p_far[1] > p_far[0]


def test_p_noise_sparse_formula_with_precomputed():
    """
    Verify the p_sparse formula manually for a controlled case.

    Layout: cluster 0 at origin, cluster 1 far away, one noise point
    placed in a region sparser than both clusters so p_sparse > 0.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.0],
            [0.5, 0.5],  # cluster 0 (dense)
            [9.0, 9.0],
            [9.0, 9.5],
            [9.5, 9.0],
            [9.5, 9.5],  # cluster 1 (dense)
            [5.0, 5.0],
        ],  # noise (sparse midpoint)
        dtype=float,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1])
    min_points = 3

    # Manual core distance computation
    tree = KDTree(X)
    core_dists, _ = tree.query(X, k=min_points)
    core_dists = core_dists.max(axis=1)

    noise_core = core_dists[labels == -1]
    max_core_cluster0 = core_dists[labels == 0].max()
    max_core_cluster1 = core_dists[labels == 1].max()

    def p_sparse_formula(noise_cd, cluster_max_cd):
        num = noise_cd - cluster_max_cd
        den = np.maximum(noise_cd, cluster_max_cd)
        return num / den if den != 0 else 0.0

    expected = min(
        p_sparse_formula(noise_core[0], max_core_cluster0),
        p_sparse_formula(noise_core[0], max_core_cluster1),
    )

    p_sparse, _ = p_noise(X, labels, min_points=min_points)
    assert p_sparse[0] == pytest.approx(expected, abs=1e-9)

import numpy as np
from clustpy.centroid import ThreeCPO
from clustpy.centroid.threecpo import poisson_dist, poisson_seeding, _poisson_labels_through_centroids, initial_poisson_clustering_labels, get_log_probs_poisson
from sklearn.datasets import make_blobs
from clustpy.utils.checks import check_clustpy_estimator
from unittest.mock import patch
from scipy.special import xlogy


def test_threecpo_estimator():
    check_clustpy_estimator(ThreeCPO(3, allow_negative_values=True, init_strat_columns="none"), ("check_complex_data"))


def test_poisson_dist():
    x = np.array([0, 0, 0, 0], dtype=float)
    y = np.array([0, 0, 0, 0], dtype=float)
    assert poisson_dist(x, y) == 0.0
    x = np.array([1.0, 3.0, 2.0], dtype=float)
    assert poisson_dist(x, x) < 1e-10
    y = np.array([3.0, 9.0, 6.0], dtype=float)
    assert poisson_dist(x, x) < 1e-10
    # 2D inputs, compare each pair to the baseline
    X = np.array([[1, 2, 0],
                  [1, 3, 2],
                  [3, 1, 4]], dtype=float)
    Y = np.array([[2, 6, 4],
                  [1, 3, 2]], dtype=float)
    D = poisson_dist(X, Y)
    D_control = np.zeros((3, 2))
    assert D.shape == (X.shape[0], Y.shape[0])
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            expected = poisson_dist(X[i], Y[j])
            assert np.abs(D[i, j] - expected) < 1e-10
            expected_manual = 0.
            r_x = np.sum(X[i])
            r_y = np.sum(Y[j])
            N = r_x + r_y
            for l in range(X.shape[1]):
                c_ij = X[i, l] + Y[j, l]
                expected_manual += xlogy(X[i, l], X[i, l] * N / r_x / c_ij)
                expected_manual += xlogy(Y[j, l], Y[j, l] * N / r_y / c_ij)
            assert np.abs(D[i, j] - expected_manual) < 1e-10
            D_control[i, j] = expected_manual
    assert np.allclose(D, D_control)
    expected = np.array([[1.0978126912379067, 0.977274818487146],
                              [0. , 0. ],
                              [1.679183332640427 , 1.242947299087259]])
    assert np.allclose(D, expected)
    # 1d vs 2d vector
    Dx = poisson_dist(Y, x)
    assert Dx.shape == (Y.shape[0],)
    xD = poisson_dist(x, Y)
    assert xD.shape == (Y.shape[0],)
    assert np.allclose(Dx, xD)
    assert np.allclose(xD, [0, 0])


def test_poisson_seeding():
    # Helper: verify each row of selected_rows appears in X
    def rows_in_X(selected_rows: np.ndarray, X: np.ndarray) -> bool:
        for r in selected_rows:
            if not np.any(np.all(X == r, axis=1)):
                return False
        return True

    # Basic shape and membership
    rng = np.random.RandomState(42)
    X = rng.poisson(3.0, size=(50, 10)).astype(float)
    n_clusters = 7
    S = poisson_seeding(X, n_clusters=n_clusters, random_state=42)
    assert S.shape == (n_clusters, X.shape[1])
    assert rows_in_X(S, X)
    # Reproducibility with random_state
    S1 = poisson_seeding(X, n_clusters=6, random_state=7)
    S2 = poisson_seeding(X, n_clusters=6, random_state=7)
    S3 = poisson_seeding(X, n_clusters=6, random_state=8)
    assert np.array_equal(S1, S2)
    assert not np.array_equal(S1, S3)
    # subset_size
    subset_size = 25
    S_sub = poisson_seeding(X, n_clusters=5, subset_size=subset_size, random_state=42)
    assert S_sub.shape == (5, X.shape[1])
    assert rows_in_X(S_sub, X)
    S_sub = poisson_seeding(X, n_clusters=5, subset_size="auto", random_state=42)
    assert S_sub.shape == (5, X.shape[1])
    assert rows_in_X(S_sub, X)
    # n_local_trials
    S_trials = poisson_seeding(X, n_clusters=5, n_local_trials=100, random_state=0)
    assert S_trials.shape == (5, X.shape[1])
    assert rows_in_X(S_trials, X)
    S_trials = poisson_seeding(X, n_clusters=5, n_local_trials=1, random_state=0)
    assert S_trials.shape == (5, X.shape[1])
    assert rows_in_X(S_trials, X)
    # alpha parameter
    S = poisson_seeding(X, n_clusters=n_clusters, alpha=10, random_state=42)
    assert S.shape == (n_clusters, X.shape[1])
    assert rows_in_X(S, X)


def test__poisson_labels_through_centroids():
    centers = np.array([[4.0, 1.0],
                        [1.0, 5.0]], dtype=float)
    X = np.array([[8.0, 2.0],   # closer to center 0
                  [2.0, 9.0],   # closer to center 1
                  [6.0, 1.0],   # closer to center 0
                  [1.0, 7.0]],  # closer to center 1
                 dtype=float)
    labels = _poisson_labels_through_centroids(X, centers)
    assert np.array_equal(labels, np.array([0, 1, 0, 1]))


def test_initial_poisson_clustering_labels():
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.poisson(1.0, size=(10, 5)),
        rng.poisson(5.0, size=(10, 5)),
        rng.poisson(10.0, size=(10, 5)),
    ]).astype(float)
    X += 1e-3
    n_clusters = 3
    strategies = [
        "random",
        "random-centers",
        "random-centers-dist",
        "poisson++",
        "poisson++-dist",
        "kmeans",
        "kmeans++",
        "rf+kmeans",
        "rf+kmeans++"
    ]
    for strat in strategies:
        labels = initial_poisson_clustering_labels(X, n_clusters, strat, random_state=42)
        # Basic properties
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (X.shape[0],)
        assert np.issubdtype(labels.dtype, np.integer)
        # Labels must be valid cluster ids in [0, n_clusters-1]
        assert labels.min() == 0
        assert labels.max() == n_clusters - 1
        labels2 = initial_poisson_clustering_labels(X, n_clusters, strat, random_state=42)
        assert np.array_equal(labels, labels2)
        if strat in ["rf+kmeans", "rf+kmeans++"]:
            X_rf = X /  X.sum(1).reshape((-1, 1))
            labels_no_rf = initial_poisson_clustering_labels(X_rf, n_clusters, strat[3:], random_state=42)
            assert np.array_equal(labels, labels_no_rf)


def test_get_log_probs_poisson():
    # Tests for X_ij * log(lambda_j^C) - lambda_i^R * lambda_j^C.
    # Small, positive dataset
    X = np.array([[1.0, 2.0],
                  [0.0, 4.0],
                  [3.0, 3.0]])
    row_lambdas = np.array([3., 4., 6.])
    # Case A: reduction="columns" with 1D column_lambdas
    col_1d = np.array([0.4, 0.6])
    exp_cols_1d = np.array([
        (1*np.log(0.4) + 2*np.log(0.6)) - 3*(0.4 + 0.6),
        (0*np.log(0.4) + 4*np.log(0.6)) - 4*(0.4 + 0.6),
        (3*np.log(0.4) + 3*np.log(0.6)) - 6*(0.4 + 0.6)
    ])
    got_cols_1d = get_log_probs_poisson(X, row_lambdas, col_1d, reduction="columns")
    assert got_cols_1d.shape == (X.shape[0],)
    assert np.allclose(got_cols_1d, exp_cols_1d, rtol=1e-12, atol=1e-12)
    # Case B: reduction="columns" with 2D column_lambdas (two clusters)
    col_2d = np.array([[0.4, 0.6],
                       [0.8, 0.4]])
    exp_cols_2d = np.array([
        [(1*np.log(0.4) + 2*np.log(0.6)) - 3*(0.4 + 0.6),
         (1*np.log(0.8) + 2*np.log(0.4)) - 3*(0.8 + 0.4)],
         [(0*np.log(0.4) + 4*np.log(0.6)) - 4*(0.4 + 0.6),
          (0*np.log(0.8) + 4*np.log(0.4)) - 4*(0.8 + 0.4)],
         [(3*np.log(0.4) + 3*np.log(0.6)) - 6*(0.4 + 0.6),
          (3*np.log(0.8) + 3*np.log(0.4)) - 6*(0.8 + 0.4)]
    ])
    got_cols_2d = get_log_probs_poisson(X, row_lambdas, col_2d, reduction="columns")
    assert got_cols_2d.shape == (X.shape[0], col_2d.shape[0])
    assert np.allclose(got_cols_2d, exp_cols_2d, rtol=1e-12, atol=1e-12)
    # Case C: reduction="rows" with 1D column_lambdas
    exp_rows_1d = np.array([
        (1*np.log(0.4) + 0*np.log(0.4) + 3*np.log(0.4)) - 0.4*(3 + 4 + 6),
        (2*np.log(0.6) + 4*np.log(0.6) + 3*np.log(0.6)) - 0.6*(3 + 4 + 6)
    ])
    got_rows_1d = get_log_probs_poisson(X, row_lambdas, col_1d, reduction="rows")
    assert got_rows_1d.shape == (X.shape[1],)
    assert np.allclose(got_rows_1d, exp_rows_1d, rtol=1e-12, atol=1e-12)
    col_sums = X.sum(axis=0)  # [1, 5]
    # Case D: reduction="rows" with 2D column_lambdas
    exp_rows_2d = np.array([
        [(1*np.log(0.4) + 0*np.log(0.4) + 3*np.log(0.4)) - 0.4*(3 + 4 + 6),
         (2*np.log(0.6) + 4*np.log(0.6) + 3*np.log(0.6)) - 0.6*(3 + 4 + 6)],
        [(1*np.log(0.8) + 0*np.log(0.8) + 3*np.log(0.8)) - 0.8*(3 + 4 + 6),
         (2*np.log(0.4) + 4*np.log(0.4) + 3*np.log(0.4)) - 0.4*(3 + 4 + 6)]
    ])
    got_rows_2d = get_log_probs_poisson(X, row_lambdas, col_2d, reduction="rows")
    assert got_rows_2d.shape == col_2d.shape
    assert np.allclose(got_rows_2d, exp_rows_2d, rtol=1e-12, atol=1e-12)
    col_sums = X.sum(0)
    got_rows_2d_col_sum = get_log_probs_poisson(X, row_lambdas, col_2d, reduction="rows", column_sums=col_sums)
    assert np.allclose(got_rows_2d, got_rows_2d_col_sum, rtol=1e-12, atol=1e-12)
    # Case E: reduction="none" requires 1D column_lambdas; returns matrix like X
    exp_none = np.array([
        [1*np.log(0.4) - 3 * 0.4, 2*np.log(0.6) - 3 * 0.6],
        [0*np.log(0.4) - 4 * 0.4, 4*np.log(0.6) - 4 * 0.6],
        [3*np.log(0.4) - 6 * 0.4, 3*np.log(0.6) - 6 * 0.6]
    ])
    got_none = get_log_probs_poisson(X, row_lambdas, col_1d, reduction="none")
    assert got_none.shape == X.shape
    assert np.allclose(got_none, exp_none, rtol=1e-12, atol=1e-12)


"""
Tests regarding the PoissonL / PoissonC object
"""


def test_simple_threecpo():
    X, labels = make_blobs(200, 4, centers=3, random_state=1, center_box=(10, 20))
    threecpo = ThreeCPO(3, random_state=1)
    assert not hasattr(threecpo, "labels_")
    threecpo.fit(X)
    assert threecpo.labels_.dtype == np.int32
    assert threecpo.labels_.shape == labels.shape
    assert threecpo.column_lambdas_one_.shape == (threecpo.n_clusters, X.shape[1])
    assert threecpo.column_lambdas_zero_.shape == (X.shape[1],)
    assert threecpo.column_lambdas_minus_.shape == (X.shape[1],)
    assert len(np.unique(threecpo.labels_)) == threecpo.n_clusters
    assert np.array_equal(np.unique(threecpo.labels_), np.arange(threecpo.n_clusters))
    # Test if random state is working
    threecpo2 = ThreeCPO(3, random_state=1)
    threecpo2.fit(X)
    assert np.array_equal(threecpo.n_clusters, threecpo2.n_clusters)
    assert np.array_equal(threecpo.labels_, threecpo2.labels_)
    assert np.array_equal(threecpo.column_lambdas_one_, threecpo2.column_lambdas_one_)
    assert np.array_equal(threecpo.column_lambdas_zero_, threecpo2.column_lambdas_zero_)
    assert np.array_equal(threecpo.column_lambdas_minus_, threecpo2.column_lambdas_minus_)
    # Test with parameters
    threecpo = ThreeCPO(3, outliers=True, re_init_empty_clusters=True, init_strat_columns="random",
                        column_bias_type="bic", ignore_c_minus=False, ignore_c_zero=False,
                        allow_negative_values=True, random_state=1)
    threecpo.fit(X)
    assert threecpo.labels_.dtype == np.int32
    assert threecpo.labels_.shape == labels.shape
    assert threecpo.column_lambdas_one_.shape == (threecpo.n_clusters, X.shape[1])
    assert threecpo.column_lambdas_zero_.shape == (X.shape[1],)
    assert threecpo.column_lambdas_minus_.shape == (X.shape[1],)
    assert len(np.unique(threecpo.labels_)) == threecpo.n_clusters + 1
    assert np.array_equal(np.unique(threecpo.labels_), np.arange(-1, threecpo.n_clusters))
    # Test with different parameters
    threecpo = ThreeCPO(3, outliers=True, re_init_empty_clusters=True, init_strat_columns="none",
                        column_bias_type="none", ignore_c_minus=False, ignore_c_zero=False,
                        allow_negative_values=True, random_state=1)
    threecpo.fit(X)
    assert threecpo.labels_.dtype == np.int32
    assert threecpo.labels_.shape == labels.shape
    assert threecpo.column_lambdas_one_.shape == (threecpo.n_clusters, X.shape[1])
    assert threecpo.column_lambdas_zero_.shape == (X.shape[1],)
    assert threecpo.column_lambdas_minus_.shape == (X.shape[1],)
    assert len(np.unique(threecpo.labels_)) == threecpo.n_clusters + 1
    assert np.array_equal(np.unique(threecpo.labels_), np.arange(-1, threecpo.n_clusters))
    # Test with ignored column selection
    threecpo = ThreeCPO(3, outliers=True, ignore_c_minus=True, ignore_c_zero=True,
                        random_state=1)
    threecpo.fit(X)
    assert threecpo.labels_.dtype == np.int32
    assert threecpo.labels_.shape == labels.shape
    assert threecpo.column_lambdas_one_.shape == (threecpo.n_clusters, X.shape[1])
    assert threecpo.column_lambdas_zero_.shape == (X.shape[1],)
    assert np.sum(threecpo.column_lambdas_zero_) == 0
    assert threecpo.column_lambdas_minus_.shape == (X.shape[1],)
    assert np.sum(threecpo.column_lambdas_minus_) == 0
    assert len(np.unique(threecpo.labels_)) == threecpo.n_clusters + 1
    assert np.array_equal(np.unique(threecpo.labels_), np.arange(-1, threecpo.n_clusters))


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_subkmeans_result(mock_fig):
    X, labels = make_blobs(200, 4, centers=3, random_state=1, center_box=(10, 20))
    tcpo = ThreeCPO(3, max_iter=3, init_strat_columns="none", track_reward=True, n_init=1)
    tcpo.fit(X)
    assert None == tcpo.plot_reward()
    assert None == tcpo.plot_reward(True, False, False)

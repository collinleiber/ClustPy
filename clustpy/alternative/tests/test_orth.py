import numpy as np
from clustpy.alternative import OrthogonalClustering, ClusteringInOrthogonalSpaces
from clustpy.data import create_nr_data
from clustpy.utils.checks import check_clustpy_estimator


def test_orthogonal_clustering_estimator():
    # Ignore check_clustering as it does not accept multiple sets of labels
    check_clustpy_estimator(OrthogonalClustering([3, 3]), ("check_complex_data", "check_clustering"))


def test_clustering_in_orthogonal_spaces_estimator():
    # Ignore check_clustering as it does not accept multiple sets of labels
    check_clustpy_estimator(ClusteringInOrthogonalSpaces([3, 3]), ("check_complex_data", "check_clustering"))


def test_simple_orthogonal_clustering():
    X, labels = create_nr_data(200, n_clusters=[3, 4, 5], subspace_features=[2, 2, 2], n_outliers=[0, 0, 0],
                               random_state=1)
    orth = OrthogonalClustering([3, 4, 5], random_state=1)
    assert not hasattr(orth, "labels_")
    orth.fit(X)
    assert orth.labels_.dtype == np.int32
    assert orth.labels_.shape == labels.shape
    # Check if random state is working
    orth_2 = OrthogonalClustering([3, 4, 5], random_state=1)
    assert not hasattr(orth_2, "labels_")
    orth_2.fit(X)
    assert np.array_equal(orth_2.labels_, orth.labels_)
    assert all([np.array_equal(orth_2.cluster_centers_[i], orth.cluster_centers_[i]) for i in range(2)])
    # Check result with explained_variance_for_clustering=1
    orth_3 = OrthogonalClustering([3, 4, 5], explained_variance_for_clustering=1, random_state=1)
    assert not hasattr(orth_3, "labels_")
    orth_3.fit(X)
    assert orth_3.labels_.shape == labels.shape
    # Test predict
    assert np.array_equal(orth.labels_, orth.predict(X))
    assert np.array_equal(orth_2.labels_[:-1], orth_2.predict(X[:-1]))


def test_simple_clustering_in_orthogonal_spaces():
    X, labels = create_nr_data(200, n_clusters=[3, 4, 5], subspace_features=[2, 2, 2], n_outliers=[0, 0, 0],
                               random_state=1)
    orth = ClusteringInOrthogonalSpaces([3, 4, 5], random_state=1)
    assert not hasattr(orth, "labels_")
    orth.fit(X)
    assert orth.labels_.dtype == np.int32
    assert orth.labels_.shape == labels.shape
    # Check if random state is working
    orth_2 = ClusteringInOrthogonalSpaces([3, 4, 5], random_state=1)
    assert not hasattr(orth_2, "labels_")
    orth_2.fit(X)
    assert np.array_equal(orth_2.labels_, orth.labels_)
    assert all([np.array_equal(orth_2.cluster_centers_[i], orth.cluster_centers_[i]) for i in range(2)])
    # Check result with explained_variance_for_clustering=1
    orth_3 = ClusteringInOrthogonalSpaces([3, 4, 5], explained_variance_for_clustering=1, random_state=1)
    assert not hasattr(orth_3, "labels_")
    orth_3.fit(X)
    assert orth_3.labels_.shape == labels.shape
    # Test predict
    assert np.array_equal(orth.labels_, orth.predict(X))
    assert np.array_equal(orth_2.labels_[:-1], orth_2.predict(X[:-1]))

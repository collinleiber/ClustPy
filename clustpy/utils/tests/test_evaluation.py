from clustpy.utils import evaluate_multiple_datasets, evaluate_dataset, EvaluationAlgorithm, EvaluationDataset, \
    EvaluationMetric
from clustpy.utils.evaluation import _preprocess_dataset, _get_n_clusters_from_algo
import numpy as np


def _add_value(X: np.ndarray, value: int = 1):
    return X + value


def _add_value1_divide_by_value2(X: np.ndarray, value1: int, value2: float):
    return (X + value1) / value2


def test_preprocess_dataset():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    X_test = _preprocess_dataset(X, preprocess_methods=_add_value, preprocess_params={})
    assert np.array_equal(X_test - 1, X)
    X_test = _preprocess_dataset(X, preprocess_methods=_add_value1_divide_by_value2,
                                 preprocess_params={"value1": 1, "value2": 2})
    assert np.array_equal(X_test * 2 - 1, X)
    X_test = _preprocess_dataset(X, preprocess_methods=[_add_value1_divide_by_value2, _add_value, _add_value],
                                 preprocess_params=[{"value1": 1, "value2": 2}, {}, {"value": 2}])
    assert np.array_equal((X_test - 1 - 2) * 2 - 1, X)


def test_get_n_clusters_from_algo():
    class A1:
        def __init__(self):
            self.n_clusters = 3

    class A2:
        def __init__(self):
            self.n_clusters_ = 3

    class A3:
        def __init__(self):
            self.labels_ = np.array([0, 0, 1, 3])

    a1 = A1()
    a2 = A2()
    a3 = A3()
    assert _get_n_clusters_from_algo(a1) == 3
    assert _get_n_clusters_from_algo(a2) == 3
    assert _get_n_clusters_from_algo(a3) == 3


def test_evaluate_dataset():
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    X = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
    L = np.array([0] * 3 + [1] * 3)
    n_repetitions = 2
    aggregations = [np.mean, np.std, np.max]
    algorithms = [
        EvaluationAlgorithm(name="KMeans", algorithm=KMeans, params={"n_clusters": 2}),
        EvaluationAlgorithm(name="KMeans_with_preprocess", algorithm=KMeans, params={"n_clusters": 2},
                            preprocess_methods=[_add_value],
                            preprocess_params=[{"value": 1}]),
        EvaluationAlgorithm(name="DBSCAN", algorithm=DBSCAN, params={"eps": 0.5, "min_samples": 2}, deterministic=True)]
    metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
               EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]
    df = evaluate_dataset(X=X, evaluation_algorithms=algorithms, evaluation_metrics=metrics, labels_true=L,
                          n_repetitions=n_repetitions, aggregation_functions=aggregations, add_runtime=True,
                          add_n_clusters=True, save_path=None, ignore_algorithms=["KMeans_with_preprocess"],
                          random_state=1)
    assert df.shape == (n_repetitions + len(aggregations), len(algorithms) * (len(metrics) + 2))


def test_evaluate_multiple_datasets():
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    from clustpy.data import load_iris
    X = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
    L = np.array([0] * 3 + [1] * 3)
    X2 = np.c_[X, L]
    n_repetitions = 2
    aggregations = [np.mean, np.std, np.max]
    algorithms = [
        EvaluationAlgorithm(name="KMeans", algorithm=KMeans, params={"n_clusters": 2}),
        EvaluationAlgorithm(name="KMeans_with_preprocess", algorithm=KMeans, params={"n_clusters": 2},
                            preprocess_methods=[_add_value],
                            preprocess_params=[{"value": 1}]),
        EvaluationAlgorithm(name="DBSCAN", algorithm=DBSCAN, params={"eps": 0.5, "min_samples": 2}, deterministic=True)]
    metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
               EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]
    datasets = [EvaluationDataset(name="iris", data=load_iris, preprocess_methods=[_add_value],
                                  preprocess_params=[{"value": 2}]),
                EvaluationDataset(name="X", data=X, labels_true=L),
                EvaluationDataset(name="X2", data=X2, labels_true=-1, ignore_algorithms=["KMeans_with_preprocess"])
                ]
    df = evaluate_multiple_datasets(evaluation_datasets=datasets, evaluation_algorithms=algorithms,
                                    evaluation_metrics=metrics, n_repetitions=n_repetitions,
                                    aggregation_functions=aggregations, add_runtime=True, add_n_clusters=True,
                                    save_path=None, save_intermediate_results=False, random_state=1)
    assert df.shape == (len(datasets) * (n_repetitions + len(aggregations)), len(algorithms) * (len(metrics) + 2))

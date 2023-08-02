import torch
from clustpy.utils import evaluate_multiple_datasets, evaluate_dataset, EvaluationAlgorithm, EvaluationDataset, \
    EvaluationMetric, EvaluationAutoencoder, load_saved_autoencoder, evaluation_df_to_latex_table
from clustpy.utils.evaluation import _preprocess_dataset, _get_n_clusters_from_algo
import numpy as np
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.data import create_subspace_data
import os
import pytest


def _add_value(X: np.ndarray, value: int = 1):
    return X + value


def _add_value1_divide_by_value2(X: np.ndarray, value1: int, value2: float):
    return (X + value1) / value2


@pytest.fixture
def cleanup_autoencoders():
    yield
    filename1 = "autoencoder1.ae"
    if os.path.isfile(filename1):
        os.remove(filename1)
    filename2 = "autoencoder2.ae"
    if os.path.isfile(filename2):
        os.remove(filename2)


@pytest.mark.usefixtures("cleanup_autoencoders")
def test_load_saved_autoencoder():
    path = "autoencoder1.ae"
    layers = [4, 2]
    X, _ = create_subspace_data(500, subspace_features=(2, 2), random_state=1)
    ae = FeedforwardAutoencoder(layers=layers)
    assert ae.fitted is False
    ae.fit(2, optimizer_params={"lr": 1e-3}, data=X, model_path=path)
    assert ae.fitted is True
    ae2 = load_saved_autoencoder(path, FeedforwardAutoencoder, {"layers": layers})
    assert ae2.fitted is True
    # Check if all parameters are equal
    for p1, p2 in zip(ae.parameters(), ae2.parameters()):
        assert torch.equal(p1.data, p2.data)


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


@pytest.mark.usefixtures("cleanup_autoencoders")
def test_evaluate_dataset_with_autoencoders():
    from sklearn.cluster import KMeans
    from clustpy.deep import DEC
    from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    np.random.seed(10)
    torch.use_deterministic_algorithms(True)
    n_repetitions = 2
    path1 = "autoencoder1.ae"
    path2 = "autoencoder2.ae"
    layers = [12, 5]
    X, L = create_subspace_data(500, subspace_features=(2, 10), random_state=1)
    aes = []
    for path in [path1, path2]:
        ae = FeedforwardAutoencoder(layers=layers)
        ae.fit(2, optimizer_params={"lr": 1e-3}, data=X, model_path=path)
        aes.append(ae)
    autoencoders = [EvaluationAutoencoder(path1, FeedforwardAutoencoder, {"layers": layers}),
                    EvaluationAutoencoder(path2, FeedforwardAutoencoder, {"layers": layers})]
    algorithms = [
        EvaluationAlgorithm(name="KMeans", algorithm=KMeans, params={"n_clusters": None, "random_state": 10}),
        EvaluationAlgorithm(name="DEC1", algorithm=DEC,
                            params={"n_clusters": None, "embedding_size": 5, "clustering_epochs": 0}),
        EvaluationAlgorithm(name="DEC2", algorithm=DEC,
                            params={"n_clusters": None, "embedding_size": 5, "clustering_epochs": 0}),
    ]
    metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
               EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]
    df = evaluate_dataset(X=X, evaluation_algorithms=algorithms, evaluation_metrics=metrics,
                          iteration_specific_autoencoders=autoencoders,
                          labels_true=L, n_repetitions=n_repetitions, add_runtime=False, add_n_clusters=False,
                          save_path=None, random_state=1)
    # Check if scores are equal
    assert abs(df.at[0, ("DEC1", "nmi")] - df.at[0, ("DEC2", "nmi")]) < 1e-8  # is equal
    assert abs(df.at[0, ("DEC1", "silhouette")] - df.at[0, ("DEC2", "silhouette")]) < 1e-8  # is equal
    assert abs(df.at[1, ("DEC1", "nmi")] - df.at[1, ("DEC2", "nmi")]) < 1e-8  # is equal
    assert abs(df.at[1, ("DEC1", "silhouette")] - df.at[1, ("DEC2", "silhouette")]) < 1e-8  # is equal
    assert abs(df.at[0, ("DEC1", "nmi")] - df.at[1, ("DEC1", "nmi")]) > 1e-2  # is not equal
    assert abs(df.at[0, ("DEC1", "silhouette")] - df.at[1, ("DEC1", "silhouette")]) > 1e-2  # is not equal


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
        EvaluationAlgorithm(name="DBSCAN", algorithm=DBSCAN, params={"eps": 0.5, "min_samples": 2},
                            deterministic=True)]
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


@pytest.fixture
def cleanup_latex_table():
    inputfile = "df.csv"
    with open(inputfile, "w") as f:
        f.write("algorithm,,Algo1,Algo1,Algo1,Algo2,Algo2,Algo2\n")
        f.write("metric,,NMI,AMI,runtime,NMI,AMI,runtime\n")
        f.write("Data1,0,0.11111,0.22222,12.34500,1.00000,1.00000,12.34500\n")
        f.write("Data1,1,0.33333,0.22222,56.78900,0.80000,0.60000,43.21000\n")
        f.write("Data1,mean,0.22222,0.22222,13.00000,0.90000,0.80000,33.12300\n")
        f.write("Data1,std,0.11111,0.00000,1.00000,0.10000,0.20000,1.50000\n")
        f.write("Data2,0,0.111,0.222,12.345,1.000,1.000,12.345\n")
        f.write("Data2,1,0.333,0.222,56.789,0.800,0.600,43.210\n")
        f.write("Data2,mean,0.222,0.222,13.000,0.900,0.800,33.123\n")
        f.write("Data2,std,0.111,0.000,1.000,0.100,0.200,1.500\n")
    yield
    if os.path.isfile(inputfile):
        os.remove(inputfile)
    outputfile1 = "latex1.txt"
    if os.path.isfile(outputfile1):
        os.remove(outputfile1)
    outputfile2 = "latex2.txt"
    if os.path.isfile(outputfile2):
        os.remove(outputfile2)


@pytest.mark.usefixtures("cleanup_latex_table")
def test_evaluation_df_to_latex_table():
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    # Test with input file
    assert None == evaluation_df_to_latex_table("df.csv", "latex1.txt", True, True, True, "red", True, 2)
    assert os.path.isfile("latex1.txt")
    # Test with df
    X = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
    L = np.array([0] * 3 + [1] * 3)
    n_repetitions = 2
    datasets = [EvaluationDataset(name="Data1", data=X, labels_true=L),
                EvaluationDataset(name="Data2", data=X * 2, labels_true=L)]
    algorithms = [
        EvaluationAlgorithm(name="KMeans1", algorithm=KMeans, params={"n_clusters": 2}),
        EvaluationAlgorithm(name="KMeans2", algorithm=KMeans, params={"n_clusters": 3})]
    metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
               EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]
    df = evaluate_multiple_datasets(evaluation_datasets=datasets, evaluation_algorithms=algorithms,
                                    evaluation_metrics=metrics, n_repetitions=n_repetitions, add_runtime=True,
                                    add_n_clusters=True,
                                    save_path=None, save_intermediate_results=False, random_state=1)
    assert None == evaluation_df_to_latex_table(df, "latex2.txt", False, False, False, None, False, 0)
    assert os.path.isfile("latex2.txt")

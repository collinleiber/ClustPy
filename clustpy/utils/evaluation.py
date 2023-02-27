import pandas as pd
import numpy as np
import time
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin
from collections.abc import Callable


def _preprocess_dataset(X: np.ndarray, preprocess_methods: list, preprocess_params: list) -> np.ndarray:
    """
    Preprocess the data set before a clustering algorithm is executed.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    preprocess_methods : list
        Can be either a list of callable functions or a single callable function
    preprocess_params : list
        List of dictionaries containing the parameters for the preprocessing methods.
        Needs one entry for each method in preprocess_methods.
        If only a single preprocessing method is given (instead of a list) a single dictionary is expected.

    Returns
    -------
    X_processed : np.ndarray
        The data set after all specified preprocessing methods have been applied
    """
    # Do preprocessing
    if type(preprocess_methods) is list:
        # If no parameters for preprocessing are specified all should be None
        if type(preprocess_params) is dict and not preprocess_params:
            preprocess_params = [{}] * len(preprocess_methods)
        # Execute multiple preprocessing steps
        assert type(preprocess_params) is list and len(preprocess_params) == len(
            preprocess_methods), \
            "preprocess_params must be a list of equal length if preprocess_methods is a list"
        X_processed = X
        for method_index, method in enumerate(preprocess_methods):
            local_params = preprocess_params[method_index]
            assert type(local_params) is dict, "All entries of preprocess_params must be of type dict"
            assert callable(method), "All entries of preprocess_methods must be callable"
            X_processed = method(X_processed, **local_params)
    else:
        # Execute single preprocessing step
        X_processed = preprocess_methods(X, **preprocess_params)
    return X_processed


def _get_n_clusters_from_algo(algo_obj: ClusterMixin) -> int:
    """
    Get n_clusters from a clustering algorithm object.
    Some algorithm need the number of clusters as input parameter and its name is 'n_clusters'.
    Other algorithms provide the number of clusters as output and its name is 'n_clusters_'.
    In rare cases the objects do not contain any information about the number of clusters.
    In those cases the the number of clusters will be obtained by analyzing the labels.

    Parameters
    ----------
    algo_obj : ClusterMixin
        The input clustering algorithm object

    Returns
    -------
    n_clusters : int
        The number of clusters
    """
    if hasattr(algo_obj, "n_clusters"):
        n_clusters = algo_obj.n_clusters
    elif hasattr(algo_obj, "n_clusters_"):
        n_clusters = algo_obj.n_clusters_
    else:
        n_clusters = np.unique(algo_obj.labels_).shape[0]
    return n_clusters


def evaluate_dataset(X: np.ndarray, evaluation_algorithms: list, evaluation_metrics: list = None,
                     labels_true: np.ndarray = None, n_repetitions: int = 10,
                     aggregation_functions: list = [np.mean, np.std], add_runtime: bool = True,
                     add_n_clusters: bool = False, save_path: str = None, ignore_algorithms: list = [],
                     random_state: np.random.RandomState = None) -> pd.DataFrame:
    """
    Evaluate the clustering result of different clustering algorithms (as specified by evaluation_algorithms) on a given data set using different metrics (as specified by evaluation_metrics).
    Each algorithm will be executed n_repetitions times and all specified metrics will be used to evaluate the clustering result.
    The final result is a pandas DataFrame containing all the information.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    evaluation_algorithms : list
        Contains objects of type EvaluationAlgorithm which are wrappers for the clustering algorithms
    evaluation_metrics : list
        Contains objects of type EvaluationMetric which are wrappers for the metrics (default: None)
    labels_true : np.ndarray
        The ground truth labels of the data set (default: None)
    n_repetitions : int
        Number of times that the clustering procedure should be executed on the same data set (default: 10)
    aggregation_functions : list
        List of aggregation functions that should be applied to the n_repetitions different results of a single clustering algorithm (default: [np.mean, np.std])
    add_runtime : bool
        Add runtime of each execution to the final table (default: True)
    add_n_clusters : bool
        Add the resulting number of clusters to the final table (default: False)
    save_path : str
        The path where the final DataFrame should be saved as csv. If None, the DataFrame will not be saved (default: None)
    ignore_algorithms : list
        List of algorithm names (as specified in the EvaluationAlgorithm object) that should be ignored for this specific data set (default: [])
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Returns
    -------
    df : pd.DataFrame
        The final DataFrame

    Examples
    ----------
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
    """
    assert evaluation_metrics is not None or add_runtime or add_n_clusters, \
        "Either evaluation metrics must be defined or add_runtime/add_n_clusters must be True"
    assert type(aggregation_functions) is list, "aggregation_functions must be list"
    if type(evaluation_algorithms) is not list:
        evaluation_algorithms = [evaluation_algorithms]
    if type(evaluation_metrics) is not list and evaluation_metrics is not None:
        evaluation_metrics = [evaluation_metrics]
    # Use same seed for each algorithm
    random_state = check_random_state(random_state)
    seeds = random_state.choice(10000, n_repetitions, replace=False)
    algo_names = [a.name for a in evaluation_algorithms]
    metric_names = [] if evaluation_metrics is None else [m.name for m in evaluation_metrics]
    # Add additional columns
    if add_runtime:
        metric_names += ["runtime"]
    if add_n_clusters:
        metric_names += ["n_clusters"]
    header = pd.MultiIndex.from_product([algo_names, metric_names], names=["algorithm", "metric"])
    data = np.zeros((n_repetitions, len(algo_names) * len(metric_names)))
    df = pd.DataFrame(data, columns=header, index=range(n_repetitions))
    for eval_algo in evaluation_algorithms:
        automatically_set_n_clusters = False
        try:
            assert type(eval_algo) is EvaluationAlgorithm, "All algortihms must be of type EvaluationAlgortihm"
            if eval_algo.name in ignore_algorithms:
                print("Ignoring algorithm {0}".format(eval_algo.name))
                continue
            print("Use algorithm {0}".format(eval_algo.name))
            # Add n_clusters automatically to algorithm parameters if it is None
            if "n_clusters" in eval_algo.params and eval_algo.params["n_clusters"] is None and labels_true is not None:
                automatically_set_n_clusters = True
            if automatically_set_n_clusters:
                if labels_true.ndim == 1:
                    # In case of normal ground truth
                    eval_algo.params["n_clusters"] = len(np.unique(labels_true[labels_true >= 0]))
                else:
                    # In case of hierarchical or nr ground truth
                    eval_algo.params["n_clusters"] = [len(np.unique(labels_true[labels_true[:, i] >= 0, i])) for i in
                                                      range(labels_true.shape[1])]
            # Algorithms can preprocess datasets (e.g. PCA + K-means)
            if eval_algo.preprocess_methods is not None:
                X_processed = _preprocess_dataset(X, eval_algo.preprocess_methods, eval_algo.preprocess_params)
            else:
                X_processed = X
            # Execute the algorithm multiple times
            for rep in range(n_repetitions):
                print("- Iteration {0}".format(rep))
                # set seed
                np.random.seed(seeds[rep])
                # Execute algorithm
                start_time = time.time()
                algo_obj = eval_algo.obj(**eval_algo.params)
                try:
                    algo_obj.fit(X_processed)
                except Exception as e:
                    print("Execution of {0} raised an exception in iteration {1}".format(eval_algo.name, rep))
                    print(e)
                    continue
                runtime = time.time() - start_time
                if add_runtime:
                    df.at[rep, (eval_algo.name, "runtime")] = runtime
                    print("-- runtime: {0}".format(runtime))
                if add_n_clusters:
                    n_clusters = _get_n_clusters_from_algo(algo_obj)
                    df.at[rep, (eval_algo.name, "n_clusters")] = n_clusters
                    print("-- n_clusters: {0}".format(n_clusters))
                # Get result of all metrics
                if evaluation_metrics is not None:
                    for eval_metric in evaluation_metrics:
                        try:
                            assert type(eval_metric) is EvaluationMetric, "All metrics must be of type EvaluationMetric"
                            # Check if metric uses ground truth (e.g. NMI, ACC, ...)
                            if eval_metric.use_gt:
                                assert labels_true is not None, "Ground truth can not be None if it is used for the chosen metric"
                                result = eval_metric.method(labels_true, algo_obj.labels_, **eval_metric.params)
                            else:
                                # Metric does not use ground truth (e.g. Silhouette, ...)
                                result = eval_metric.method(X, algo_obj.labels_, **eval_metric.params)
                            df.at[rep, (eval_algo.name, eval_metric.name)] = result
                            print("-- {0}: {1}".format(eval_metric.name, result))
                        except Exception as e:
                            print("Metric {0} raised an exception and will be skipped".format(eval_metric.name))
                            print(e)
                if eval_algo.deterministic:
                    if add_runtime:
                        df.at[np.arange(1, n_repetitions), (eval_algo.name, "runtime")] = df.at[
                            0, (eval_algo.name, "runtime")]
                    if add_n_clusters:
                        df.at[np.arange(1, n_repetitions), (eval_algo.name, "n_clusters")] = df.at[
                            0, (eval_algo.name, "n_clusters")]
                    for eval_metric in evaluation_metrics:
                        df.at[np.arange(1, n_repetitions), (eval_algo.name, eval_metric.name)] = df.at[
                            0, (eval_algo.name, eval_metric.name)]
                    break
        except Exception as e:
            print("Algorithm {0} raised an exception and will be skipped".format(eval_algo.name))
            print(e)
        # Prepare eval_algo params for next dataset
        if automatically_set_n_clusters:
            eval_algo.params["n_clusters"] = None
    for agg in aggregation_functions:
        df.loc[agg.__name__] = agg(df.values, axis=0)
    if save_path is not None:
        df.to_csv(save_path)
    return df


def evaluate_multiple_datasets(evaluation_datasets: list, evaluation_algorithms: list, evaluation_metrics: list = None,
                               n_repetitions: int = 10, aggregation_functions: list = [np.mean, np.std],
                               add_runtime: bool = True, add_n_clusters: bool = False, save_path: str = None,
                               save_intermediate_results: bool = False,
                               random_state: np.random.RandomState = None) -> pd.DataFrame:
    """
    Evaluate the clustering result of different clustering algorithms (as specified by evaluation_algorithms) on a set of data sets (as specified by evaluation_datasets) using different metrics (as specified by evaluation_metrics).
    Each algorithm will be executed n_repetitions times and all specified metrics will be used to evaluate the clustering result.
    The final result is a pandas DataFrame containing all the information.

    Parameters
    ----------
    evaluation_datasets : list
        Contains objects of type EvaluationDataset which are wrappers for the data sets
    evaluation_algorithms : list
        Contains objects of type EvaluationAlgorithm which are wrappers for the clustering algorithms
    evaluation_metrics : list
        Contains objects of type EvaluationMetric which are wrappers for the metrics (default: None)
    n_repetitions : int
        Number of times that the clustering procedure should be executed on the same data set (default: 10)
    aggregation_functions : list
        List of aggregation functions that should be applied to the n_repetitions different results of a single clustering algorithm (default: [np.mean, np.std])
    add_runtime : bool
        Add runtime of each execution to the final table (default: True)
    add_n_clusters : bool
        Add the resulting number of clusters to the final table (default: False)
    save_path : str
        The path where the final DataFrame should be saved as csv. If None, the DataFrame will not be saved (default: None)
    save_intermediate_results : bool
        Defines whether the result of each data set should be separately saved. Useful if the evaluation takes a lot of time (default: False)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Returns
    -------
    df : pd.DataFrame
        The final DataFrame

    Examples
    ----------
    See the readme.md

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
    """
    assert not save_intermediate_results or save_path is not None, "save_path can not be None if " \
                                                                   "save_intermediate_results is True"
    if type(evaluation_datasets) is not list:
        evaluation_datasets = [evaluation_datasets]
    data_names = [d.name for d in evaluation_datasets]
    df_list = []
    for eval_data in evaluation_datasets:
        try:
            assert type(eval_data) is EvaluationDataset, "All datasets must be of type EvaluationDataset"
            print("=== Start evaluation of {0} ===".format(eval_data.name))
            # If data is a path read file. If it is a callable load data
            labels_true = None
            if type(eval_data.data) is str:
                X = np.genfromtxt(eval_data.data, **eval_data.file_reader_params)
            elif type(eval_data.data) is np.ndarray:
                X = eval_data.data
            else:
                X, labels_true = eval_data.data(**eval_data.file_reader_params)
            # Check if ground truth columns are defined
            if type(eval_data.labels_true) is int or type(eval_data.labels_true) is list:
                labels_true = X[:, eval_data.labels_true]
                X = np.delete(X, eval_data.labels_true, axis=1)
            elif type(eval_data.labels_true) is np.ndarray:
                labels_true = eval_data.labels_true
            print("=== (Data shape: {0} / Ground truth shape: {1}) ===".format(X.shape,
                                                                               labels_true if labels_true is None else labels_true.shape))
            if eval_data.preprocess_methods is not None:
                X = _preprocess_dataset(X, eval_data.preprocess_methods, eval_data.preprocess_params)
            inner_save_path = None if not save_intermediate_results else "{0}_{1}.{2}".format(save_path.split(".")[0],
                                                                                              eval_data.name,
                                                                                              save_path.split(".")[1])
            df = evaluate_dataset(X, evaluation_algorithms, evaluation_metrics=evaluation_metrics,
                                  labels_true=labels_true,
                                  n_repetitions=n_repetitions, aggregation_functions=aggregation_functions,
                                  add_runtime=add_runtime, add_n_clusters=add_n_clusters, save_path=inner_save_path,
                                  ignore_algorithms=eval_data.ignore_algorithms, random_state=random_state)
            df_list.append(df)
        except Exception as e:
            print("Dataset {0} raised an exception and will be skipped".format(eval_data.name))
            print(e)
    all_dfs = pd.concat(df_list, keys=data_names)
    if save_path is not None:
        all_dfs.to_csv(save_path)
    return all_dfs


class EvaluationDataset():
    """
    The EvaluationDataset object is a wrapper for actual data sets.
    It contains all the information necessary to evaluate a data set using the evaluate_multiple_datasets method.

    Parameters
    ----------
    name : str
        Name of the data set. Can be chosen freely
    data : np.ndarray
        The actual data set. Can be a np.ndarray, a path to a data file (of type str) or a callable (e.g. a method from clustpy.data)
    labels_true : np.ndarray
        The ground truth labels. Can be a np.ndarray, an int or list specifying which columns of the data contain the labels or None if no ground truth labels are present (default: None)
    file_reader_params : dict
        Dictionary containing the information necessary to load a data file. Only relevant if data is of type str (default: {})
    preprocess_methods : list
        Specify preprocessing steps before evaluating the data set.
        Can be either a list of callable functions or a single callable function (default: None)
    preprocess_params : list
        List of dictionaries containing the parameters for the preprocessing methods.
        Needs one entry for each method in preprocess_methods.
        If only a single preprocessing method is given (instead of a list) a single dictionary is expected (default: {})
    ignore_algorithms : list
        List of algorithm names (as specified in the EvaluationAlgorithm object) that should be ignored for this specific data set (default: [])

    Examples
    ----------
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
    """

    def __init__(self, name: str, data: np.ndarray, labels_true: np.ndarray = None, file_reader_params: dict = {},
                 preprocess_methods: list = None, preprocess_params: list = {}, ignore_algorithms: list = []):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(data) is np.ndarray or type(data) is str or callable(data), "data must be a numpy array, a string " \
                                                                                "containing the path to a data file or a " \
                                                                                "function returning a data and a labels array"
        self.data = data
        assert labels_true is None or type(labels_true) is int or type(labels_true) is list or type(labels_true) is \
               np.ndarray, "gt_columns must be an int, a list, a numpy array or None"
        self.labels_true = labels_true
        assert type(file_reader_params) is dict, "file_reader_params must be a dict"
        self.file_reader_params = file_reader_params
        assert callable(preprocess_methods) or type(
            preprocess_methods) is list or preprocess_methods is None, "preprocess_methods must be a method, a list of methods or None"
        self.preprocess_methods = preprocess_methods
        assert type(preprocess_params) is dict or type(
            preprocess_methods) is list, "preprocess_params must be a dict or a list of dicts"
        self.preprocess_params = preprocess_params
        assert type(ignore_algorithms) is list, "ignore_algorithms must be a list"
        self.ignore_algorithms = ignore_algorithms


class EvaluationMetric():
    """
    The EvaluationMetric object is a wrapper for evaluation metrics.
    It contains all the information necessary to evaluate a data set using the evaluate_dataset or evaluate_multiple_datasets method.

    Parameters
    ----------
    name : str
        Name of the metric. Can be chosen freely
    metric : Callable
        The actual metric function
    params : dict
        Parameters given to the metric function (default: {})
    use_gt : bool
        If true, the input to the metric will be the ground truth labels and the predicted labels (e.g. normalized mutual information).
        If false, the input will be the data and the predicted labels (e.g. silhouette score) (default: True)

    Examples
    ----------
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
    """

    def __init__(self, name: str, metric: Callable, params: dict = {}, use_gt: bool = True):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert callable(metric), "method must be a method"
        self.method = metric
        assert type(params) is dict, "params must be a dict"
        self.params = params
        assert type(use_gt) is bool, "use_gt must be bool"
        self.use_gt = use_gt


class EvaluationAlgorithm():
    """
    The EvaluationAlgorithm object is a wrapper for clustering algorithms.
    It contains all the information necessary to evaluate a data set using the evaluate_dataset or evaluate_multiple_datasets method.

    Parameters
    ----------
    name : str
        Name of the metric. Can be chosen freely
    algorithm : ClusterMixin
        The actual object of the clustering algorithm
    params : dict
        Parameters given to the clustering algorithm (default: {})
    deterministic : bool
        Defines if the algorithm produces a deterministic clustering result (e.g. like DBSCAN).
        In this case the algorithm will only be executed once even though a higher number of repetitions is specified when evaluating a data set (default: False)
    preprocess_methods : list
        Specify preprocessing steps performed on each data set before executing the clustering algorithm.
        Can be either a list of callable functions or a single callable function (default: None)
    preprocess_params : list
        List of dictionaries containing the parameters for the preprocessing methods.
        Needs one entry for each method in preprocess_methods.
        If only a single preprocessing method is given (instead of a list) a single dictionary is expected (default: {})

    Examples
    ----------
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
    """

    def __init__(self, name: str, algorithm: ClusterMixin, params: dict = {}, deterministic: bool = False,
                 preprocess_methods: list = None, preprocess_params: list = {}):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(algorithm) is type, "name must be Algorithm class"
        self.obj = algorithm
        assert type(params) is dict, "params must be a dict"
        self.params = params
        assert type(deterministic) is bool, "deterministic must be bool"
        self.deterministic = deterministic
        assert callable(preprocess_methods) or type(
            preprocess_methods) is list or preprocess_methods is None, "preprocess_methods must be a method, a list of methods or None"
        self.preprocess_methods = preprocess_methods
        assert type(preprocess_params) is dict or type(
            preprocess_methods) is list, "preprocess_params must be a dict or a list of dicts"
        self.preprocess_params = preprocess_params

import pandas as pd
import numpy as np
import time
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin
from collections.abc import Callable
import os
import inspect
from sklearn.datasets._base import Bunch
import sys


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


def _get_fixed_seed_for_each_run(n_repetitions: int, 
                                 random_state: np.random.RandomState | int | list) -> list:
    """
    Get the same seed for each run of an algorithm and data set.

    Parameters
    ----------
    n_repetitions : int
        Number of times that the clustering procedure should be executed on the same data set
    random_state : np.random.RandomState | int | list
        use a fixed random state to get a repeatable solution. Can also be of type int. 
        Furthermore, if can be a list containing an int for each repetition

    Returns
    -------
    seeds : list
        List of seeds (integers), one for earch repetition
    """
    if random_state is None or isinstance(random_state, (int, np.integer)) or isinstance(random_state, np.random.RandomState):
        random_state = check_random_state(random_state)
        seeds = random_state.choice(10000, n_repetitions, replace=False)
    elif type(random_state) is list or type(random_state) is tuple or type(random_state) is np.ndarray:
        seeds = random_state
    else:
        raise Exception("random_state must be of type int, np.random.RandomState or list")
    assert len(seeds) == n_repetitions, "If random_state is a list, its length must be equal to the number of repetitions"
    assert all([isinstance(entry, (int, np.integer)) for entry in seeds]), "If random_state is a list, all entries must be integers"
    assert np.unique(seeds).shape[0] == n_repetitions, "Each seed must be unique, however duplicates were found in the seeds/random_state"
    return seeds


def evaluate_dataset(X: np.ndarray, evaluation_algorithms: list, evaluation_metrics: list = None,
                     labels_true: np.ndarray = None, n_repetitions: int = 10,
                     X_test: np.ndarray = None, labels_true_test: np.ndarray = None,
                     aggregation_functions: tuple = (np.mean, np.std), add_runtime: bool = True,
                     add_n_clusters: bool = False, save_path: str = None, save_labels_path: str = None,
                     ignore_algorithms: tuple = (), dataset_name: str = None,
                     random_state: np.random.RandomState | int | list = None, quiet: bool = False) -> pd.DataFrame:
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
    X_test : np.ndarray
        An optional test data set that will be evaluated using the predict method of the clustering algorithms (default: None)
    labels_true_test : np.ndarray
        The ground truth labels of the test data set (default: None)
    aggregation_functions : tuple
        List of aggregation functions that should be applied to the n_repetitions different results of a single clustering algorithm (default: [np.mean, np.std])
    add_runtime : bool
        Add runtime of each execution to the final table (default: True)
    add_n_clusters : bool
        Add the resulting number of clusters to the final table (default: False)
    save_path : str
        The path where the final DataFrame should be saved as csv. If None, the DataFrame will not be saved (default: None)
    save_labels_path : str
        The path where the clustering labels should be saved as csv. If None, the labels will not be saved (default: None)
    ignore_algorithms : tuple
        List of algorithm names (as specified in the EvaluationAlgorithm object) that should be ignored for this specific data set (default: [])
    dataset_name : str
        The name of the dataset; only relevant if iteration_specific_params are defined for an EvaluationAlgorithm (default: None)
    random_state : np.random.RandomState | int | list
        use a fixed random state to get a repeatable solution. Can also be of type int. 
        Furthermore, if can be a list containing an int for each repetition (default: None)
    quiet : bool
        Do not print any output

    Returns
    -------
    df : pd.DataFrame
        The final DataFrame

    Examples
    ----------
    >>> from sklearn.cluster import KMeans, DBSCAN
    >>> from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    >>>
    >>> def _add_value(x, value):
    >>>     return x + value
    >>>
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
    >>> L = np.array([0] * 3 + [1] * 3)
    >>> n_repetitions = 2
    >>> aggregations = [np.mean, np.std, np.max]
    >>> algorithms = [
    >>>     EvaluationAlgorithm(name="KMeans", algorithm=KMeans, params={"n_clusters": 2}),
    >>>     EvaluationAlgorithm(name="KMeans_with_preprocess", algorithm=KMeans, params={"n_clusters": 2},
    >>>                         preprocess_methods=[_add_value],
    >>>                         preprocess_params=[{"value": 1}]),
    >>>     EvaluationAlgorithm(name="DBSCAN", algorithm=DBSCAN, params={"eps": 0.5, "min_samples": 2}, deterministic=True)]
    >>> metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
    >>>            EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]
    >>> df = evaluate_dataset(X=X, evaluation_algorithms=algorithms, evaluation_metrics=metrics, labels_true=L,
    >>>                       n_repetitions=n_repetitions, aggregation_functions=aggregations, add_runtime=True,
    >>>                       add_n_clusters=True, save_path=None, ignore_algorithms=["KMeans_with_preprocess"],
    >>>                       random_state=1)
    """
    assert evaluation_metrics is not None or add_runtime or add_n_clusters, \
        "Either evaluation metrics must be defined or add_runtime/add_n_clusters must be True"
    assert type(aggregation_functions) is list or type(
        aggregation_functions) is tuple, "aggregation_functions must be list or tuple. Yout input is of type {0}".format(type(aggregation_functions))
    if type(evaluation_algorithms) is not list:
        evaluation_algorithms = [evaluation_algorithms]
    if type(evaluation_metrics) is not list and evaluation_metrics is not None:
        evaluation_metrics = [evaluation_metrics]
    if save_labels_path is not None and "." not in save_labels_path:
        save_labels_path = save_labels_path + ".csv"
    assert save_labels_path is None or len(
        save_labels_path.split(".")) == 2, "save_labels_path must only contain a single dot. E.g., NAME.csv"
    if save_path is not None and "." not in save_path:
        save_path = save_path + ".csv"
    assert save_path is None or len(
        save_path.split(".")) == 2, "save_path must only contain a single dot. E.g., NAME.csv"
    seeds = _get_fixed_seed_for_each_run(n_repetitions, random_state)
    algo_names = [a.name for a in evaluation_algorithms]
    assert max(
        np.unique(algo_names, return_counts=True)[1]) == 1, "Some names of your algorithms do not seem to be unique!"
    metric_names = [] if evaluation_metrics is None else [m.name for m in evaluation_metrics]
    if X_test is not None:
        metric_names += [mn + "_TEST" for mn in metric_names]
    # Add additional columns
    if add_runtime:
        metric_names += ["runtime"]
    if add_n_clusters:
        metric_names += ["n_clusters"]
    assert len(metric_names) == 0 or max(np.unique(metric_names, return_counts=True)[
                                             1]) == 1, "Some names of your metrics do not seem to be unique! Note that metrics must not be named 'runtime' or 'n_clusters'"
    header = pd.MultiIndex.from_product([algo_names, metric_names], names=["algorithm", "metric"])
    value_placeholder = np.zeros((n_repetitions, len(algo_names) * len(metric_names)))
    df = pd.DataFrame(value_placeholder, columns=header, index=range(n_repetitions))
    for eval_algo in evaluation_algorithms:
        automatically_set_n_clusters = False
        try:
            assert type(eval_algo) is EvaluationAlgorithm, "All algortihms must be of type EvaluationAlgortihm"
            if eval_algo.name in ignore_algorithms:
                if not quiet:
                    print("Ignoring algorithm {0}".format(eval_algo.name))
                continue
            if not quiet:
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
                if X_test is not None:
                    X_test_processed = _preprocess_dataset(X_test, eval_algo.preprocess_methods,
                                                           eval_algo.preprocess_params)
            else:
                X_processed = X
                if X_test is not None:
                    X_test_processed = X_test
            # Execute the algorithm multiple times
            for rep in range(n_repetitions):
                if not quiet:
                    print("- Iteration {0}".format(rep))
                # set seed
                np.random.seed(seeds[rep])
                tmp_params = eval_algo.params.copy()
                # Check if algorithm uses iteration_specific_params and if length of values is correct
                if eval_algo.iteration_specific_params is not None:
                    for iteration_params_key in eval_algo.iteration_specific_params.keys():
                        assert type(iteration_params_key) is str or (type(iteration_params_key) is tuple and len(
                            iteration_params_key) == 2), "All keys within iteration_specific_params must be str or a tuple of length 2, i.e., (dataset name, parameter name). Your key: {0}".format(
                            iteration_params_key)
                        assert len(eval_algo.iteration_specific_params[
                                       iteration_params_key]) == n_repetitions, "All values within iteration_specific_params must be lists with length equal to n_repetitions. Should be {0}, but is {1}".format(
                            n_repetitions, len(eval_algo.iteration_specific_params[iteration_params_key]))
                        if type(iteration_params_key) is str:
                            tmp_params[iteration_params_key] = \
                                eval_algo.iteration_specific_params[iteration_params_key][rep]
                        elif iteration_params_key[0] == dataset_name:
                            tmp_params[iteration_params_key[1]] = \
                                eval_algo.iteration_specific_params[iteration_params_key][rep]
                # Add random state to algorithm
                algo_input_params = inspect.getfullargspec(eval_algo.algorithm).args + inspect.getfullargspec(eval_algo.algorithm).kwonlyargs
                if "random_state" in algo_input_params and "random_state" not in tmp_params.keys():
                    tmp_params["random_state"] = seeds[rep]
                # Execute algorithm
                start_time = time.time()
                algo_obj = eval_algo.algorithm(**tmp_params)
                try:
                    algo_obj.fit(X_processed)
                except Exception as e:
                    if not quiet:
                        print("Execution of {0} raised an exception in iteration {1}".format(eval_algo.name, rep))
                        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                    continue
                # Optional: Obtain labels from the predict method
                if X_test is not None:
                    try:
                        labels_predicted_test = algo_obj.predict(X_test_processed)
                    except Exception as e:
                        if not quiet:
                            print("Problem when running the predict method of {0} in iteration {1}".format(eval_algo.name,
                                                                                                       rep))
                            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                        labels_predicted_test = None
                runtime = time.time() - start_time
                if add_runtime:
                    df.at[rep, (eval_algo.name, "runtime")] = runtime
                    if not quiet:
                        print("-- runtime: {0}".format(runtime))
                if add_n_clusters:
                    n_clusters = _get_n_clusters_from_algo(algo_obj)
                    df.at[rep, (eval_algo.name, "n_clusters")] = n_clusters
                    if not quiet:
                        print("-- n_clusters: {0}".format(n_clusters))
                # Optional: Save labels
                if save_labels_path is not None:
                    save_labels_path_algo = None if save_labels_path is None else "{0}_{1}_{2}.{3}".format(
                        save_labels_path.split(".")[0], eval_algo.name, rep, save_labels_path.split(".")[1])
                    # Check if directory exists
                    parent_directory = os.path.dirname(save_labels_path_algo)
                    if parent_directory != "" and not os.path.isdir(parent_directory):
                        os.makedirs(parent_directory)
                    np.savetxt(save_labels_path_algo, algo_obj.labels_)
                    # Also save predict labels
                    if X_test is not None and labels_predicted_test is not None:
                        save_labels_path_algo_test = "{0}_TEST.{1}".format(save_labels_path_algo.split(".")[0],
                                                                           save_labels_path_algo.split(".")[1])
                        np.savetxt(save_labels_path_algo_test, labels_predicted_test)
                # Get result of all metrics
                if evaluation_metrics is not None:
                    for eval_metric in evaluation_metrics:
                        try:
                            assert type(eval_metric) is EvaluationMetric, "All metrics must be of type EvaluationMetric"
                            # Check if metric uses ground truth (e.g. NMI, ACC, ...)
                            if eval_metric.metric_type == "external":
                                assert labels_true is not None, "Ground truth can not be None if an external metric is used"
                                result = eval_metric.method(labels_true, algo_obj.labels_, **eval_metric.params)
                                if X_test is not None and labels_predicted_test is not None:
                                    result_test = eval_metric.method(labels_true_test, labels_predicted_test,
                                                                     **eval_metric.params)
                            elif eval_metric.metric_type == "internal":
                                # Metric does not use ground truth (e.g. Silhouette, ...)
                                result = eval_metric.method(X, algo_obj.labels_, **eval_metric.params)
                                if X_test is not None and labels_predicted_test is not None:
                                    result_test = eval_metric.method(X_test, labels_predicted_test,
                                                                     **eval_metric.params)
                            else:
                                # A custom metric is used
                                result = eval_metric.method(X, labels_true, algo_obj.labels_, algo_obj, **eval_metric.params)
                                if X_test is not None and labels_predicted_test is not None:
                                    result_test = eval_metric.method(X_test, labels_true_test, labels_predicted_test, algo_obj,
                                                                     **eval_metric.params)
                            df.at[rep, (eval_algo.name, eval_metric.name)] = result
                            if not quiet:
                                print("-- {0}: {1}".format(eval_metric.name, result))
                            if X_test is not None and labels_predicted_test is not None:
                                df.at[rep, (eval_algo.name, eval_metric.name + "_TEST")] = result_test
                                if not quiet:
                                    print("-- {0} (TEST): {1}".format(eval_metric.name, result_test))
                        except Exception as e:
                            if not quiet:
                                print("Metric {0} raised an exception and will be skipped".format(eval_metric.name))
                                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                if eval_algo.deterministic:
                    for element in range(1, n_repetitions):
                        if add_runtime:
                            df.at[element, (eval_algo.name, "runtime")] = df.at[
                                0, (eval_algo.name, "runtime")]
                        if add_n_clusters:
                            df.at[element, (eval_algo.name, "n_clusters")] = df.at[
                                0, (eval_algo.name, "n_clusters")]
                        for eval_metric in evaluation_metrics:
                            df.at[element, (eval_algo.name, eval_metric.name)] = df.at[
                                0, (eval_algo.name, eval_metric.name)]
                            if X_test is not None:
                                df.at[element, (eval_algo.name, eval_metric.name + "_TEST")] = df.at[
                                    0, (eval_algo.name, eval_metric.name + "_TEST")]
                    break
        except Exception as e:
            if not quiet:
                print("Algorithm {0} raised an exception and will be skipped".format(eval_algo.name))
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        # Prepare eval_algo params for next dataset
        if automatically_set_n_clusters:
            eval_algo.params["n_clusters"] = None
    for agg in aggregation_functions:
        df.loc[agg.__name__] = agg(df.values, axis=0)
    if save_path is not None:
        # Check if directory exists
        parent_directory = os.path.dirname(save_path)
        if parent_directory != "" and not os.path.isdir(parent_directory):
            os.makedirs(parent_directory)
        df.to_csv(save_path)
    return df


def evaluate_multiple_datasets(evaluation_datasets: list, evaluation_algorithms: list, evaluation_metrics: list = None,
                               n_repetitions: int = 10, aggregation_functions: tuple = (np.mean, np.std),
                               add_runtime: bool = True, add_n_clusters: bool = False, save_path: str = None,
                               save_intermediate_results: bool = False, save_labels_path: str = None,
                               random_state: np.random.RandomState | int | list = None, quiet: bool = False) -> pd.DataFrame:
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
    aggregation_functions : tuple
        List of aggregation functions that should be applied to the n_repetitions different results of a single clustering algorithm (default: [np.mean, np.std])
    add_runtime : bool
        Add runtime of each execution to the final table (default: True)
    add_n_clusters : bool
        Add the resulting number of clusters to the final table (default: False)
    save_path : str
        The path where the final DataFrame should be saved as csv. If None, the DataFrame will not be saved (default: None)
    save_intermediate_results : bool
        Defines whether the result of each data set should be separately saved. 
        Useful if the evaluation takes a lot of time.
        The files will be saved as [save_path]_[DATASET_NAME]. 
        This implies that save_path has to be defined if save_intermediate_results is set to True (default: False)
    save_labels_path : str
        The path where the clustering labels should be saved as csv. If None, the labels will not be saved (default: None)
    random_state : np.random.RandomState | int | list
        use a fixed random state to get a repeatable solution. Can also be of type int.
        Furthermore, if can be a list containing an int for each repetition (default: None)
    quiet : bool
        Do not print any output

    Returns
    -------
    df : pd.DataFrame
        The final DataFrame

    Examples
    ----------
    See the readme.md

    >>> from sklearn.cluster import KMeans, DBSCAN
    >>> from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    >>> from clustpy.data import load_iris
    >>>
    >>> def _add_value(x, value):
    >>>     return x + value
    >>>
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
    >>> L = np.array([0] * 3 + [1] * 3)
    >>> X2 = np.c_[X, L]
    >>> n_repetitions = 2
    >>> aggregations = [np.mean, np.std, np.max]
    >>> algorithms = [
    >>>     EvaluationAlgorithm(name="KMeans", algorithm=KMeans, params={"n_clusters": 2}),
    >>>     EvaluationAlgorithm(name="KMeans_with_preprocess", algorithm=KMeans, params={"n_clusters": 2},
    >>>                         preprocess_methods=[_add_value],
    >>>                         preprocess_params=[{"value": 1}]),
    >>>     EvaluationAlgorithm(name="DBSCAN", algorithm=DBSCAN, params={"eps": 0.5, "min_samples": 2}, deterministic=True)]
    >>> metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
    >>>            EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]
    >>> datasets = [EvaluationDataset(name="iris", data=load_iris, preprocess_methods=[_add_value],
    >>>                               preprocess_params=[{"value": 2}]),
    >>>             EvaluationDataset(name="X", data=X, labels_true=L),
    >>>             EvaluationDataset(name="X2", data=X2, labels_true=-1, ignore_algorithms=["KMeans_with_preprocess"])
    >>>             ]
    >>> df = evaluate_multiple_datasets(evaluation_datasets=datasets, evaluation_algorithms=algorithms,
    >>>                                 evaluation_metrics=metrics, n_repetitions=n_repetitions,
    >>>                                 aggregation_functions=aggregations, add_runtime=True, add_n_clusters=True,
    >>>                                 save_path=None, save_intermediate_results=False, random_state=1)
    """
    assert not save_intermediate_results or save_path is not None, "save_path can not be None if " \
                                                                   "save_intermediate_results is True"
    if type(evaluation_datasets) is not list:
        evaluation_datasets = [evaluation_datasets]
    if save_labels_path is not None and "." not in save_labels_path:
        save_labels_path = save_labels_path + ".csv"
    assert save_labels_path is None or len(
        save_labels_path.split(".")) == 2, "save_labels_path must only contain a single dot. E.g., NAME.csv"
    if save_path is not None and "." not in save_path:
        save_path = save_path + ".csv"
    assert save_path is None or len(
        save_path.split(".")) == 2, "save_path must only contain a single dot. E.g., NAME.csv"
    data_names = [d.name for d in evaluation_datasets]
    assert max(
        np.unique(data_names, return_counts=True)[1]) == 1, "Some names of your datasets do not seem to be unique!"
    seeds = _get_fixed_seed_for_each_run(n_repetitions, random_state)
    df_list = []
    for eval_data in evaluation_datasets:
        try:
            assert type(eval_data) is EvaluationDataset, "All datasets must be of type EvaluationDataset"
            if not quiet:
                print("=== Start evaluation of {0} ===".format(eval_data.name))
            X, labels_true, X_test, labels_true_test = _get_data_and_labels_from_evaluation_dataset(eval_data.data,
                                                                                                    eval_data.data_loader_params,
                                                                                                    eval_data.labels_true,
                                                                                                    eval_data.train_test_split)
            if not quiet:
                print("=== (Data shape: {0} / Ground truth shape: {1}) ===".format(X.shape,
                                                                               labels_true if labels_true is None else labels_true.shape))
            if eval_data.preprocess_methods is not None:
                X = _preprocess_dataset(X, eval_data.preprocess_methods, eval_data.preprocess_params)
                if X_test is not None:
                    X_test = _preprocess_dataset(X_test, eval_data.preprocess_methods, eval_data.preprocess_params)
            inner_save_path = None if not save_intermediate_results else "{0}_{1}.{2}".format(save_path.split(".")[0],
                                                                                              eval_data.name,
                                                                                              save_path.split(".")[1])
            inner_save_labels_path = None if save_labels_path is None else "{0}_{1}.{2}".format(
                save_labels_path.split(".")[0], eval_data.name, save_labels_path.split(".")[1])
            df = evaluate_dataset(X, evaluation_algorithms, evaluation_metrics=evaluation_metrics,
                                  labels_true=labels_true,
                                  n_repetitions=n_repetitions, X_test=X_test, labels_true_test=labels_true_test,
                                  aggregation_functions=aggregation_functions,
                                  add_runtime=add_runtime, add_n_clusters=add_n_clusters, save_path=inner_save_path,
                                  save_labels_path=inner_save_labels_path,
                                  ignore_algorithms=eval_data.ignore_algorithms, dataset_name=eval_data.name,
                                  random_state=seeds, quiet=quiet)
            df_list.append(df)
        except Exception as e:
            if not quiet:
                print("Dataset {0} raised an exception and will be skipped".format(eval_data.name))
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    all_dfs = pd.concat(df_list, keys=data_names)
    if save_path is not None:
        # Check if directory exists
        parent_directory = os.path.dirname(save_path)
        if parent_directory != "" and not os.path.isdir(parent_directory):
            os.makedirs(parent_directory)
        all_dfs.to_csv(save_path)
    return all_dfs


def _get_data_and_labels_from_evaluation_dataset(data_input: np.ndarray, data_loader_params_input: dict,
                                                 labels_input: np.ndarray, train_test_split: np.ndarray) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Use the parameters stored in the EvaluationDataset to load the data and the labels.
    If specifies it will also load a distinct test dataset.

    Parameters
    ----------
    data_input : np.ndarray
        The actual data set. Can be a np.ndarray, a path to a data file (of type str) or a callable (e.g. a method from clustpy.data)
    data_loader_params_input : dict
        Dictionary containing the information necessary to load data from a function or file. Only relevant if data is of type callable or str
    labels_input : np.ndarray
        The ground truth labels. Can be a np.ndarray, an int or list specifying which columns of the data contain the labels or None if no ground truth labels are present.
        If data is a callable, the ground truth labels can also be obtained by that function and labels_true can be None
    train_test_split : bool
        Specifies if the loaded dataset should be split into a train and test set. Can be of type bool, list or np.ndarray.
        If train_test_split is a boolean and true, the data loader will use the parameter "subset" to load a train and test set. In that case data must be a callable.
        If train_test_split is a list/np.ndarray, the entries specify the indices of the data array that should be used for the test set

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The dataset,
        The labels (can be None),
        The test dataset (can be None),
        The test labels (can be None)
    """
    # If data is a path read file. If it is a callable load data
    labels_true = None
    X_test = None
    labels_true_test = None
    if type(data_input) is str:
        X = np.genfromtxt(data_input, **data_loader_params_input)
    elif type(data_input) is np.ndarray:
        X = data_input
    else:
        data_loader_params = inspect.getfullargspec(data_input).args
        # Check if dataset should be split in train and test set
        if type(train_test_split) is bool and train_test_split and "subset" in data_loader_params:
            dataset = data_input(subset="train", **data_loader_params_input)
            testset = data_input(subset="test", **data_loader_params_input)
            if type(testset) is Bunch:
                X_test = testset.data
                labels_true_test = testset.target
            else:
                X_test, labels_true_test = testset
        else:
            dataset = data_input(**data_loader_params_input)
        if type(dataset) is Bunch:
            X = dataset.data
            labels_true = dataset.target
        else:
            X, labels_true = dataset
    # Check if ground truth columns are defined
    if type(labels_input) is int or type(labels_input) is list:
        labels_true = X[:, labels_input]
        X = np.delete(X, labels_input, axis=1)
    elif type(labels_input) is np.ndarray:
        labels_true = labels_input
    # Check if dataset should be split in train and test set
    if type(train_test_split) is list or type(train_test_split) is np.ndarray:
        test_subset = np.zeros(X.shape[0], dtype=bool)
        test_subset[train_test_split] = True
        X_test = X[test_subset]
        X = X[~test_subset]
        if labels_true is not None:
            labels_true_test = labels_true[test_subset]
            labels_true = labels_true[~test_subset]
    return X, labels_true, X_test, labels_true_test


def evaluation_df_to_latex_table(df: pd.DataFrame | str, relevant_row : str | int = "mean", output_path: str = None, pm_row: str | int | None = "std", 
                                 bracket_row: str | int | None = None, best_in_bold: bool = True, second_best_underlined: bool = True, 
                                 color_by_value: str = None, higher_is_better: list = None, multiplier: int | float | list | None = 100,
                                 decimal_places: int = 1, color_min_max: tuple = (5, 70)) -> str:
    """
    Convert the resulting dataframe of an evaluation into a latex table.
    Note that the latex package booktabs is required, so usepackage{booktabs} must be included in the latex file.
    This method will only consider the values contained in the row with the name relevant_row.
    The default relevant_row is "mean", which implies that the mean was used as an aggregation function when creating the dataframe.
    Other values can be added to the latex table either after plus-minus by specifying pm_row or in brackets by specifying bracket_row.

    Parameters
    ----------
    df : pd.DataFrame | str
        The pandas dataframe. Can also be a string that contains the path to the saved dataframe
    relevant_row : str | int
        The name of the row in the df that is used to create the latex table (default: "mean")
    output_path : str
        The path were the resulting latex table text file will be stored (default: None)
    pm_row : str | int
        The name of the row in the df that should be added to the latex table after the value from relevant_row separated by plus-minus (default: "std")
    bracket_row : str | int
        The name of the row in the df that should be added to the latex table in brackets after the value from relevant_row and, if stated, the value from pm_row (default: None)
    best_in_bold : bool
        Print best value for each combination of dataset and metric in bold.
        Note, that the latex package bm is used, so usepackage{bm} must be included in the latex file (default: True)
    second_best_underlined : bool
        Print second-best value for each combination of dataset and metric underlined (default: True)
    color_by_value : str
        Define the color that should be used to indicate the difference between the values of the metrics.
        Uses colorcell, so usepackage{colortbl} or usepackage[table]{xcolor} must be included in the latex file.
        Can be 'blue' for example (default: None)
    higher_is_better : list
        List with booleans. Each value indicates if a high value for a certain metric is better than a low value.
        The length of the list must be equal to the number of different metrics.
        Entries can also be None if neither higher nor lower is better.
        If None, it is always assumed that a higher value is better, except for the runtime and for n_clusters (default: None)
    multiplier : int | float | list | None
        If defined, all values, except n_clusters and runtime, will be multiplied by this value, e.g. to receive values in percent they will be multiplied by 100.
        Can also be a list containing a different value for each metric. 
        If it is None, the original values will be used (default: 100)
    decimal_places : int
        Number of decimal places that should be used in the latex table (default: 1)
    color_min_max : tuple
        Range of the color saturation used if color_by_value is defined. First entry is min (>= 0) and second entry is max (<= 100) (default: (5, 70))

    Returns
    -------
    output : str
        The created latex string
    """
    # Load dataframe
    assert type(df) == pd.DataFrame or type(df) == str, "Type of df must be pandas DataFrame or string (path to file)"
    if type(df) == str:
        df_file = open(df, "r").readlines()
        multiple_datasets = df_file[2].split(",")[0] != "0"
        df = pd.read_csv(df, index_col=[0, 1] if multiple_datasets else [0], header=[0, 1])
    else:
        multiple_datasets = isinstance(df.index, pd.MultiIndex)
    # Get main information from dataframe
    if multiple_datasets:
        datasets = list(dict.fromkeys([s[0] for s in df.index]))
        pm_row_contained = pm_row in [s[1] for s in df.index]
        bracket_row_contained = bracket_row in [s[1] for s in df.index]
    else:
        datasets = [None]
        pm_row_contained = pm_row in [s for s in df.index]
        bracket_row_contained = bracket_row in [s for s in df.index]
    algorithms = list(dict.fromkeys([s[0] for s in df.keys()]))
    metrics = list(dict.fromkeys([s[1] for s in df.keys()]))
    if multiplier is None or type(multiplier) is int or type(multiplier) is float:
        multiplier = [multiplier] * len(metrics)
    assert len(multiplier) == len(
        metrics), "multiplier must be float/int or the length of multiplier must match the number of metrics. multiplier = {0} (length {1}), metrics = {2} (length {3})".format(
        multiplier, len(multiplier), metrics, len(metrics))
    assert higher_is_better is None or len(higher_is_better) == len(
        metrics), "Length of higher_is_better and the number of metrics does not match. higher_is_better = {0} (length {1}), metrics = {2} (length {3})".format(
        higher_is_better, len(higher_is_better), metrics, len(metrics))
    # Check color range
    assert (type(color_min_max) is tuple or type(color_min_max) is list) and len(color_min_max) == 2 and color_min_max[0] <= color_min_max[
        1] and color_min_max[0] >= 0 and color_min_max[1] <= 100, "color_min_max must be a tuple containing two values within [0, 100], where the first value is smaller than the second"
    c_min, c_max = color_min_max
    c_max_adj = c_max - c_min
    # Create string
    output = ""
    # Create string for standard table
    output += "\\begin{table}\n\\centering\n\\caption{TODO}\n\\resizebox{1\\textwidth}{!}{\n\\begin{tabular}{l|"
    if multiple_datasets:
        output += "l|" + "c" * len(algorithms) + "}\n\\toprule\n\\textbf{Dataset} & "
    else:
        output += "c" * len(algorithms) + "}\n\\toprule\n"
    output += "\\textbf{Metric} & " + " & ".join([a.replace("_", "\\_") for a in algorithms]) + "\\\\\n\\midrule\n"
    # Write values into table
    for j, d in enumerate(datasets):
        for i, m in enumerate(metrics):
            # Check if a higher value is better for this metric
            if higher_is_better is None:
                if m == "n_clusters":
                    metric_is_higher_better = None
                else:
                    metric_is_higher_better = (m != "runtime")
            else:
                metric_is_higher_better = higher_is_better[i]
            # Escape underscore that could be contained in metric name
            m_write = m.replace("_", "\\_")
            # Write name of dataset and metric
            if multiple_datasets:
                if i == 0:
                    # Escape underscore that could be contained in dataset name
                    to_write = d.replace("_", "\\_") + " & " + m_write
                else:
                    to_write = "& " + m_write
            else:
                to_write = m_write
            # Get all values from the experiments (are stored separately to calculated min values)
            all_values = []
            for a in algorithms:
                if multiple_datasets:
                    relevant_value = df[a, m][d, relevant_row]
                else:
                    relevant_value = df[a, m][relevant_row]
                if relevant_value is not None and not np.isnan(relevant_value):
                    if multiplier[i] is not None and m not in ["n_clusters", "runtime"]:
                        relevant_value *= multiplier[i]
                    relevant_value = round(relevant_value, decimal_places)
                all_values.append(relevant_value)
            all_values_sorted = np.unique(all_values)  # automatically sorted
            all_values_sorted = all_values_sorted[~np.isnan(all_values_sorted)]
            for k, a in enumerate(algorithms):
                relevant_value = all_values[k]
                value_write = "$" + str(relevant_value)
                # If pm_row is specified and contained in the dataframe, information will be added
                if pm_row is not None and pm_row_contained:
                    if multiple_datasets:
                        pm_value = df[a, m][d, pm_row]
                    else:
                        pm_value = df[a, m][pm_row]
                    if pm_value is not None and not np.isnan(pm_value):
                        if multiplier[i] is not None and m not in ["n_clusters", "runtime"]:
                            pm_value *= multiplier[i]
                        pm_value = round(pm_value, decimal_places)
                    value_write = value_write + " \\pm " + str(pm_value)
                # If bracket_row is specified and contained in the dataframe, information will be added
                if bracket_row is not None and bracket_row_contained:
                    if multiple_datasets:
                        bracket_value = df[a, m][d, bracket_row]
                    else:
                        bracket_value = df[a, m][bracket_row]
                    if bracket_value is not None and not np.isnan(bracket_value):
                        if multiplier[i] is not None and m not in ["n_clusters", "runtime"]:
                            bracket_value *= multiplier[i]
                        bracket_value = round(bracket_value, decimal_places)
                    value_write = value_write + "~(" + str(bracket_value) +")"
                value_write = value_write + "$"
                if relevant_value is not None and not np.isnan(relevant_value):
                    # Optional: Write best value in bold and second best underlined
                    if best_in_bold and metric_is_higher_better is not None and (
                            (relevant_value == all_values_sorted[-1] and metric_is_higher_better) or (
                            relevant_value == all_values_sorted[0] and not metric_is_higher_better)):
                        value_write = "\\bm{" + value_write + "}"
                    elif second_best_underlined and metric_is_higher_better is not None and (
                            (relevant_value == all_values_sorted[-2] and metric_is_higher_better) or (
                            relevant_value == all_values_sorted[1] and not metric_is_higher_better)):
                        value_write = "\\underline{" + value_write + "}"
                    # Optional: Color cells by value difference
                    if color_by_value is not None and metric_is_higher_better is not None:
                        if all_values_sorted[-1] != all_values_sorted[0]:
                            if metric_is_higher_better:
                                color_saturation = round((relevant_value - all_values_sorted[0]) / (
                                        all_values_sorted[-1] - all_values_sorted[0]) * c_max_adj) + c_min  # value between c_min and c_max
                            else:
                                color_saturation = round((all_values_sorted[-1] - relevant_value) / (
                                        all_values_sorted[-1] - all_values_sorted[0]) * c_max_adj) + c_min  # value between c_min and c_max
                        else:
                            color_saturation = 0
                        assert type(color_saturation) is int, "color_saturation must be an int but is {0}".format(
                            type(color_saturation))
                        value_write = "\\cellcolor{" + color_by_value + "!" + str(color_saturation) + "}" + value_write
                to_write += " & " + value_write
            to_write += "\\\\\n"
            output += to_write
        if j != len(datasets) - 1:
            output += "\\midrule\n"
        else:
            output += "\\bottomrule\n\\end{tabular}}\n\\end{table}"
    if output_path != None:
        with open(output_path, "w") as f:
            f.write(output)
    return output


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
        The ground truth labels. Can be a np.ndarray, an int or list specifying which columns of the data contain the labels or None if no ground truth labels are present.
        If data is a callable, the ground truth labels can also be obtained by that function and labels_true can be None (default: None)
    data_loader_params : dict
        Dictionary containing the information necessary to load data from a function or file. Only relevant if data is of type callable or str (default: {})
    train_test_split : bool
        Specifies if the laoded dataset should be split into a train and test set. Can be of type bool, list or np.ndarray.
        If train_test_split is a boolean and true, the data loader will use the parameter "subset" to load a train and test set. In that case data must be a callable.
        If train_test_split is a list/np.ndarray, the entries specify the indices of the data array that should be used for the test set (default: None)
    preprocess_methods : list
        Specify preprocessing steps before evaluating the data set.
        Can be either a list of callable functions or a single callable function.
        Will also be applied to an optional test data set (default: None)
    preprocess_params : list
        List of dictionaries containing the parameters for the preprocessing methods.
        Needs one entry for each method in preprocess_methods.
        If only a single preprocessing method is given (instead of a list) a single dictionary is expected (default: {})
    ignore_algorithms : tuple
        List of algorithm names (as specified in the EvaluationAlgorithm object) that should be ignored for this specific data set (default: [])

    Examples
    ----------
    See evaluate_multiple_datasets()

    >>> from clustpy.data import load_iris, load_wine
    >>> ed1 = EvaluationDataset(name="iris", data=load_iris)
    >>> X, L = load_wine()
    >>> ed2 = EvaluationDataset(name="wine", data=X, labels_true=L)
    """

    def __init__(self, name: str, data: np.ndarray, labels_true: np.ndarray = None, data_loader_params: dict = None,
                 train_test_split: bool = None, preprocess_methods: list = None, preprocess_params: list = None,
                 ignore_algorithms: tuple = ()):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert "." not in name, "name must not contain a dot"
        assert type(data) is np.ndarray or type(data) is str or callable(data), "data must be a numpy array, a string " \
                                                                                "containing the path to a data file or a " \
                                                                                "function returning a data and a labels array"
        self.data = data
        assert labels_true is None or type(labels_true) is int or type(labels_true) is list or type(labels_true) is \
               np.ndarray, "gt_columns must be an int, a list, a numpy array or None"
        self.labels_true = labels_true
        assert data_loader_params is None or type(data_loader_params) is dict, "data_loader_params must be a dict"
        self.data_loader_params = {} if data_loader_params is None else data_loader_params
        assert train_test_split is None or type(train_test_split) is bool or type(train_test_split) is list or type(
            train_test_split) is np.ndarray, "train_test_split must be None, a bool, list or numpy array"
        assert type(train_test_split) is not bool or callable(
            data), "If train_test_split is a bool, data must be callable"
        self.train_test_split = train_test_split
        assert callable(preprocess_methods) or type(
            preprocess_methods) is list or preprocess_methods is None, "preprocess_methods must be a method, a list of methods or None"
        self.preprocess_methods = preprocess_methods
        assert preprocess_params is None or type(preprocess_params) is dict or type(
            preprocess_params) is list, "preprocess_params must be a dict or a list of dicts"
        self.preprocess_params = {} if preprocess_params is None else preprocess_params
        assert type(ignore_algorithms) is list or type(
            ignore_algorithms) is tuple, "ignore_algorithms must be a tuple or a list"
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
    metric_type : str
        The type of an EvaluationMetric can be either 'external', 'internal' or 'custom'.
        If 'external', the metric (e.g. normalized mutual information) compares the predicted labels with ground truth labels, i.e., it is defined as metric(labels_true, labels_pred, **params).
        If 'internal', the metric (e.g. silhouette score) compares the predicted labels with patterns in the data, i.e., it is defined as metric(X, labels_pred, **params).
        If 'custom', a custom metric is used that can use the data, ground truth labels, predicted labels and other attributes from the algorithm, i.e., it is defined as metric(X, labels_true, labels_pred, algorithm_obj, **params) (default: "external")

    Examples
    ----------
    See evaluate_multiple_datasets()

    >>> from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
    >>> em1 = EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, metric_type="external"),
    >>> em2 = EvaluationMetric(name="silhouette", metric=silhouette, metric_type="internal")
    """

    def __init__(self, name: str, metric: Callable, params: dict = None, metric_type: str = "external"):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert callable(metric), "method must be a method"
        self.method = metric
        assert params is None or type(params) is dict, "params must be a dict"
        self.params = {} if params is None else params
        assert type(metric_type) is str and metric_type in ["external", "internal", "custom"], "metric_type must be str. Options are 'external', 'internal' and 'custom'"
        self.metric_type = metric_type


class EvaluationAlgorithm():
    """
    The EvaluationAlgorithm object is a wrapper for clustering algorithms.
    It contains all the information necessary to evaluate a data set using the evaluate_dataset or evaluate_multiple_datasets method.
    If the algorithm requires the number of clusters as input parameter, params should contain {"n_clusters": None}.

    Parameters
    ----------
    name : str
        Name of the metric. Can be chosen freely
    algorithm : ClusterMixin
        The actual object of the clustering algorithm
    params : dict
        Parameters given to the clustering algorithm.
        If the algorithm uses a n_clusters parameter, it can be set to None, e.g., params={"n_clusters": None}.
        In this case the evaluation methods will automatically use the correct number of clusters for the specific data set (default: {})
    deterministic : bool
        Defines if the algorithm produces a deterministic clustering result (e.g. like DBSCAN).
        In this case the algorithm will only be executed once even though a higher number of repetitions is specified when evaluating a data set (default: False)
    iteration_specific_params : dict
        Dictionary containing parameters that are specefic for a certain iteration.
        The keys of the dict can be either of type str which referes to the name of the parameter or of type tuple.
        If a key is a tuple, the parameters are only valid for a specific dataset.
        Here, the name of the dataset (see EvaluationDataset) is defined in the first entry of the tuple and the name of the parameter in the second, e.g. {("Iris", "eps"): [0.1,0.2,...]}.
        All values within the dict must be of type list, where the length must be equal to 'n_repetitions' in 'evaluate_multiple_datasets()' and 'evaluate_dataset()'.
        Can be None if no iteration-specific parameters are used (default: None)
    preprocess_methods : list
        Specify preprocessing steps performed on each data set before executing the clustering algorithm.
        Can be either a list of callable functions or a single callable function.
        Will also be applied to an optional test data set (default: None)
    preprocess_params : dict
        List of dictionaries containing the parameters for the preprocessing methods.
        Needs one entry for each method in preprocess_methods.
        If only a single preprocessing method is given (instead of a list) a single dictionary is expected (default: {})


    Examples
    ----------
    See evaluate_multiple_datasets()

    >>> from sklearn.cluster import DBSCAN
    >>> from clustpy.partition import SubKmeans
    >>> ea1 = EvaluationAlgorithm(name="DBSCAN", algorithm=DBSCAN, params={"eps": 0.5, "min_samples": 2}, deterministic=True)
    >>> ea2 = EvaluationAlgorithm(name="SubKMeans", algorithm=SubKmeans, params={"n_clusters": None})
    """

    def __init__(self, name: str, algorithm: ClusterMixin, params: dict = None, deterministic: bool = False,
                 iteration_specific_params: dict = None, preprocess_methods: list = None,
                 preprocess_params: dict = None):
        assert type(name) is str, "name must be a string"
        assert "." not in name, "name must not contain a dot"
        self.name = name
        self.algorithm = algorithm
        assert params is None or type(params) is dict, "params must be a dict"
        self.params = {} if params is None else params
        assert type(deterministic) is bool, "deterministic must be bool"
        self.deterministic = deterministic
        assert type(
            iteration_specific_params) is dict or iteration_specific_params is None, "iteration_specific_params must be a dict or None"
        self.iteration_specific_params = iteration_specific_params
        assert callable(preprocess_methods) or type(
            preprocess_methods) is list or preprocess_methods is None, "preprocess_methods must be a method, a list of methods or None"
        self.preprocess_methods = preprocess_methods
        assert preprocess_params is None or type(preprocess_params) is dict or type(
            preprocess_params) is list, "preprocess_params must be a dict or a list of dicts"
        self.preprocess_params = {} if preprocess_params is None else preprocess_params

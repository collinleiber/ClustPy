import pandas as pd
import numpy as np
import time
from cluspy.utils._wrapper_methods import _get_n_clusters_from_algo
import random


def _preprocess_dataset(X, preprocess_methods, preprocess_params):
    # Do preprocessing
    if type(preprocess_methods) is list:
        # If no parameters for preprocessing are specified all should be None
        if type(preprocess_params) is dict and not preprocess_params:
            preprocess_params = [{}] * len(preprocess_methods)
        # Execute multiple preprocessing steps
        assert type(preprocess_params) is list and len(preprocess_params) == len(
            preprocess_methods), \
            "preprocess_params must be a list of equal length if preprocess_methods is a list"
        for method_index, method in enumerate(preprocess_methods):
            local_params = preprocess_params[method_index]
            assert type(local_params) is dict, "All entries of preprocess_params must be of type dict"
            assert callable(method), "All entries of preprocess_methods must be methods"
            X_processed = method(X, **local_params)
    else:
        # Execute single preprocessing step
        X_processed = preprocess_methods(X, **preprocess_params)
    return X_processed


def evaluate_dataset(X, evaluation_algorithms, evaluation_metrics=None, gt=None, repetitions=10, add_average=True,
                     add_std=True, add_runtime=True, add_n_clusters=False, save_path=None, ignore_algorithms=[]):
    assert evaluation_metrics is not None or add_runtime or add_n_clusters, \
        "Either evaluation metrics must be defined or add_runtime/add_n_clusters must be True"
    if type(evaluation_algorithms) is not list:
        evaluation_algorithms = [evaluation_algorithms]
    if type(evaluation_metrics) is not list and evaluation_metrics is not None:
        evaluation_metrics = [evaluation_metrics]
    # Use same seed for each algorithm
    seeds = random.sample(range(1000), repetitions)
    algo_names = [a.name for a in evaluation_algorithms]
    metric_names = [] if evaluation_metrics is None else [m.name for m in evaluation_metrics]
    if add_runtime:
        metric_names += ["runtime"]
    if add_n_clusters:
        metric_names += ["n_clusters"]
    header = pd.MultiIndex.from_product([algo_names, metric_names], names=["algorithm", "metric"])
    data = np.zeros((repetitions, len(algo_names) * len(metric_names)))
    df = pd.DataFrame(data, columns=header, index=range(repetitions))
    for eval_algo in evaluation_algorithms:
        automatically_set_n_clusters = False
        try:
            assert type(eval_algo) is EvaluationAlgorithm, "All algortihms must be of type EvaluationAlgortihm"
            if eval_algo.name in ignore_algorithms:
                print("Ignoring algorithm {0}".format(eval_algo.name))
                continue
            print("Use algorithm {0}".format(eval_algo.name))
            # Add n_clusters automatically to algorithm parameters if it is None
            if "n_clusters" in eval_algo.params and eval_algo.params["n_clusters"] is None and gt is not None:
                automatically_set_n_clusters = True
            if automatically_set_n_clusters:
                if gt.ndim == 1:
                    # In case of normal ground truth
                    eval_algo.params["n_clusters"] = len(np.unique(gt[gt >= 0]))
                else:
                    # In case of hierarchical or nr ground truth
                    eval_algo.params["n_clusters"] = [len(np.unique(gt[gt[:, i] >= 0, i])) for i in range(gt.shape[1])]
            # Algorithms can preprocess datasets (e.g. PCA + K-means)
            if eval_algo.preprocess_methods is not None:
                X_processed = _preprocess_dataset(X, eval_algo.preprocess_methods, eval_algo.preprocess_params)
            else:
                X_processed = X
            # Execute the algorithm multiple times
            for rep in range(repetitions):
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
                    all_n_clusters = _get_n_clusters_from_algo(algo_obj)
                    n_clusters = all_n_clusters if eval_algo.label_column is None else all_n_clusters[
                        eval_algo.label_column]
                    df.at[rep, (eval_algo.name, "n_clusters")] = n_clusters
                    print("-- n_clusters: {0}".format(n_clusters))
                labels = algo_obj.labels_ if eval_algo.label_column is None else algo_obj.labels_[:,
                                                                                 eval_algo.label_column]
                # Get result of all metrics
                if evaluation_metrics is not None:
                    for eval_metric in evaluation_metrics:
                        try:
                            assert type(eval_metric) is EvaluationMetric, "All metrics must be of type EvaluationMetric"
                            # Check if metric uses ground truth (e.g. NMI, ACC, ...)
                            if eval_metric.use_gt:
                                assert gt is not None, "Ground truth can not be None if it is used for the chosen metric"
                                result = eval_metric.method(gt, labels, **eval_metric.params)
                            else:
                                # Metric does not use ground truth (e.g. Silhouette, ...)
                                result = eval_metric.method(X, labels, **eval_metric.params)
                            df.at[rep, (eval_algo.name, eval_metric.name)] = result
                            print("-- {0}: {1}".format(eval_metric.name, result))
                        except Exception as e:
                            print("Metric {0} raised an exception and will be skipped".format(eval_metric.name))
                            print(e)
                if eval_algo.deterministic:
                    if add_runtime:
                        df.at[np.arange(1, repetitions), (eval_algo.name, "runtime")] = df.at[
                            0, (eval_algo.name, "runtime")]
                    if add_n_clusters:
                        df.at[np.arange(1, repetitions), (eval_algo.name, "n_clusters")] = df.at[
                            0, (eval_algo.name, "n_clusters")]
                    for eval_metric in evaluation_metrics:
                        df.at[np.arange(1,repetitions), (eval_algo.name, eval_metric.name)] = df.at[0, (eval_algo.name, eval_metric.name)]
                    break
        except Exception as e:
            print("Algorithm {0} raised an exception and will be skipped".format(eval_algo.name))
            print(e)
        if automatically_set_n_clusters:
            eval_algo.params["n_clusters"] = None
    if add_average:
        df.loc["avg"] = np.mean(df.values, axis=0)
    if add_std:
        df.loc["std"] = np.std(df.values, axis=0)
    if save_path is not None:
        df.to_csv(save_path)
    return df


def evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, evaluation_metrics=None, repetitions=10,
                               add_average=True, add_std=True, add_runtime=True, add_n_clusters=False, save_path=None,
                               save_intermediate_results=False):
    """
    Example:
    from cluspy.evaluation import *
    from cluspy.deep import DEC, IDEC, DCN, VaDE
    from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
    from cluspy.data import load_usps, load_mnist

    def znorm(X):
        return (X - np.mean(X)) / np.std(X)

    def identity(X):
        return X

    to_ignore = ["VaDE", "IDEC"]
    datasets = [
        EvaluationDataset("MNIST", data=load_mnist, preprocess_methods=znorm, ignore_algorithms=["DEC"]),
        EvaluationDataset("USPS", data=load_usps, preprocess_methods=znorm),
        EvaluationDataset("USPS+1", data=load_usps, preprocess_methods=[znorm, identity],
                          ignore_algorithms=to_ignore)
    ]
    algorithms = [EvaluationAlgorithm("DEC", DEC, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                   "dec_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("IDEC", IDEC, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                     "dec_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("DCN", DCN, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                   "dcn_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("VaDE", VaDE, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                     "vade_epochs": 150, "embedding_size": 10})]
    metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ARI", ari)]
    df = evaluate_multiple_datasets(datasets, algorithms, metrics, 10, True, True, False, True,
                                    save_path="valuation.csv", save_intermediate_results=True)
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
            # If data is a path, load file
            gt = None
            if type(eval_data.data) is str:
                X = np.genfromtxt(eval_data.data, **eval_data.file_reader_params)
            elif type(eval_data.data) is np.ndarray:
                X = eval_data.data
            else:
                X, gt = eval_data.data(**eval_data.file_reader_params)
            # Check if ground truth columns are defined
            if type(eval_data.gt_columns) is int or type(eval_data.gt_columns) is list:
                gt = X[:, eval_data.gt_columns]
                X = np.delete(X, eval_data.gt_columns, axis=1)
            elif type(eval_data.gt_columns) is np.ndarray:
                gt = eval_data.gt_columns
            print("=== (Data shape: {0} / Ground truth shape: {1}) ===".format(X.shape, gt if gt is None else gt.shape))
            if eval_data.preprocess_methods is not None:
                X = _preprocess_dataset(X, eval_data.preprocess_methods, eval_data.preprocess_params)
            inner_save_path = None if not save_intermediate_results else "{0}_{1}.{2}".format(save_path.split(".")[0],
                                                                                              eval_data.name,
                                                                                              save_path.split(".")[1])
            df = evaluate_dataset(X, evaluation_algorithms, evaluation_metrics=evaluation_metrics, gt=gt,
                                  repetitions=repetitions, add_average=add_average, add_std=add_std,
                                  add_runtime=add_runtime, add_n_clusters=add_n_clusters, save_path=inner_save_path,
                                  ignore_algorithms=eval_data.ignore_algorithms)
            df_list.append(df)
        except Exception as e:
            print("Dataset {0} raised an exception and will be skipped".format(eval_data.name))
            print(e)
    all_dfs = pd.concat(df_list, keys=data_names)
    if save_path is not None:
        all_dfs.to_csv(save_path)
    return all_dfs


class EvaluationDataset():

    def __init__(self, name, data, gt_columns=None, file_reader_params={}, preprocess_methods=None, preprocess_params={},
                 ignore_algorithms=[]):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(data) is np.ndarray or type(data) is str or callable(data), "data must be a numpy array, a string " \
                                                                                "containing the path to a data file or a " \
                                                                                "function returning a data and a labels array"
        self.data = data
        assert gt_columns is None or type(gt_columns) is int or type(gt_columns) is list or type(gt_columns) is\
               np.ndarray, "gt_columns must be an int, a list, a numpy array or None"
        self.gt_columns = gt_columns
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

    def __init__(self, name, method, params={}, use_gt=True):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert callable(method), "method must be a method"
        self.method = method
        assert type(params) is dict, "params must be a dict"
        self.params = params
        assert type(use_gt) is bool, "use_gt must be bool"
        self.use_gt = use_gt


class EvaluationAlgorithm():

    def __init__(self, name, obj, params={}, deterministic=False, preprocess_methods=None, preprocess_params={}, label_column=None):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(obj) is type, "name must be Algorithm class"
        self.obj = obj
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
        assert label_column is None or type(label_column) is int, "label_column must be None or int"
        self.label_column = label_column

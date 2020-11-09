import pandas as pd
import numpy as np
import time
from cluspy.utils._wrapper_methods import _get_n_clusters_from_algo

def evaluate_dataset(X, evaluation_algorithms, evaluation_metrics=None, gt=None, repetitions=10, add_average=True,
                     add_runtime=True, add_n_clusters=False, save_path=None):
    """
    Example:
    from cluspy.data.synthetic_data_creator import create_subspace_data
    from cluspy.density import MultiDensityDBSCAN
    from cluspy.subspace import SubKmeans
    from cluspy.centroid import XMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, adjusted_rand_score as ars, silhouette_score as sc
    X, L = create_subspace_data(1500, total_features=2)
    algorithms = [EvaluationAlgorithm("MDDBSCAN_k=15", MultiDensityDBSCAN, {"k":15}), EvaluationAlgorithm("MDDBSCAN_k=25", MultiDensityDBSCAN, {"k":25}),
                      EvaluationAlgorithm("Xmeans", XMeans), EvaluationAlgorithm("SubKmeans", SubKmeans, {"n_clusters":3}, 0)]
    metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("AMI", ami), EvaluationMetric("Adjusted rand", ars), EvaluationMetric("Silhouette", sc, use_gt=False)]
    df = evaluate_dataset(X, algorithms, metrics, L, 10, save_path="test_results.csv", add_n_clusters=True)

    :param X: dataset
    :param evaluation_algorithms: input algorithms - list of EvaluationAlgorithm
    :param evaluation_metrics: input metrics - list of EvaluationMetric (default: None)
    :param gt: ground truth (Default: None)
    :param repetitions: number of repetitions to execute (default: 10)
    :param add_average: add average vor each column
    :param add_runtime: add runtime into the results table (default: True)
    :param add_n_clusters: add n_clusters into the results table (default: False)
    :param save_path: optional - path where the results should be saved (default: None)
    :return: dataframe with evaluation results
    """
    assert evaluation_metrics is not None or add_runtime or add_n_clusters, \
        "Either evaluation metrics must be defined or add_runtime/add_n_clusters must be True"
    if type(evaluation_algorithms) is not list:
        evaluation_algorithms = [evaluation_algorithms]
    if type(evaluation_metrics) is not list and evaluation_metrics is not None:
        evaluation_metrics = [evaluation_metrics]
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
        print("Use algorithm {0}".format(eval_algo.name))
        assert type(eval_algo) is EvaluationAlgorithm, "All algortihms must be of type EvaluationAlgortihm"
        # Execute the algorithm multiple times
        for rep in range(repetitions):
            print("- Iteration {0}".format(rep + 1))
            start_time = time.time()
            algo_obj = eval_algo.obj(**eval_algo.params)
            algo_obj.fit(X)
            runtime = time.time() - start_time
            if add_runtime:
                df.at[rep, (eval_algo.name, "runtime")] = runtime
            if add_n_clusters:
                all_n_clusters = _get_n_clusters_from_algo(algo_obj)
                n_clusters = all_n_clusters if eval_algo.label_column is None else all_n_clusters[
                    eval_algo.label_column]
                df.at[rep, (eval_algo.name, "n_clusters")] = n_clusters
            labels = algo_obj.labels if eval_algo.label_column is None else algo_obj.labels[:, eval_algo.label_column]
            # Get result of all metrics
            if evaluation_metrics is not None:
                for eval_metric in evaluation_metrics:
                    assert type(eval_metric) is EvaluationMetric, "All metrics must be of type EvaluationMetric"
                    # Check if metric uses ground truth (e.g. NMI, ACC, ...)
                    if eval_metric.use_gt:
                        assert gt is not None, "Ground truth can not be None if it is used for the chosen metric"
                        result = eval_metric.method(gt, labels, **eval_metric.params)
                    else:
                        # Metric does not use ground truth (e.g. Silhouette, ...)
                        result = eval_metric.method(X, labels, **eval_metric.params)
                    df.at[rep, (eval_algo.name, eval_metric.name)] = result
    if add_average:
        df.loc["avg"] = np.mean(df.values, axis=0)
    if save_path is not None:
        df.to_csv(save_path)
    return df


def evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, evaluation_metrics=None, repetitions=10,
                               add_average=True, add_runtime=True, add_n_clusters=False, save_path=None):
    if type(evaluation_datasets) is not list:
        evaluation_datasets = [evaluation_datasets]
    data_names = [d.name for d in evaluation_datasets]
    df_list = []
    for eval_data in evaluation_datasets:
        print("=== Start evaluation of {0} ===". format(eval_data.name))
        assert type(eval_data) is EvaluationDataset, "All datasets must be of type EvaluationDataset"
        data_file = np.genfromtxt(eval_data.path, delimiter=eval_data.delimiter)
        X = np.delete(data_file, eval_data.gt_columns, axis=1)
        gt = data_file[:, eval_data.gt_columns]
        if eval_data.preprocess_method is not None:
            X = eval_data.preprocess_method(X, **eval_data.preprocess_params)
        inner_save_path = None if save_path is None else "{0}_{1}.{2}".format(save_path.split(".")[0], eval_data.name,
                                                                            save_path.split(".")[1])
        df = evaluate_dataset(X, evaluation_algorithms, evaluation_metrics=evaluation_metrics, gt=gt,
                                   repetitions=repetitions, add_average=add_average, add_runtime=add_runtime,
                                   add_n_clusters=add_n_clusters, save_path=inner_save_path)
        df_list.append(df)
    all_dfs = pd.concat(df_list, keys=data_names)
    if save_path is not None:
        all_dfs.to_csv(save_path)
    return all_dfs

class EvaluationDataset():

    def __init__(self, name, path, gt_columns=-1, delimiter=",", preprocess_method = None, preprocess_params = {}):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(path) is str, "path must be a string"
        self.path = path
        assert type(gt_columns) is int or type(gt_columns) is list, "gt_columns must be an int or a list"
        self.gt_columns = gt_columns
        assert type(delimiter) is str, "delimiter must be a string"
        self.delimiter = delimiter
        assert callable(preprocess_method) or preprocess_method is None, "preprocess_method must be a method or None"
        self.preprocess_method = preprocess_method
        assert type(preprocess_params) is dict, "preprocess_params must be a dict"
        self.preprocess_params = preprocess_params

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

    def __init__(self, name, obj, params={}, label_column=None):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(obj) is type, "name must be Algorithm class"
        self.obj = obj
        assert type(params) is dict, "params must be a dict"
        self.params = params
        assert label_column is None or type(label_column) is int, "label_column must be None or int"
        self.label_column = label_column

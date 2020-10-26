import pandas as pd
import numpy as np
import time


def evaluate(X, evaluation_algorithms, evaluation_metrics=None, GT=None, repetitions=10, add_runtime=True,
                     save_path=None):
    """
    Example:
    from cluspy.data.synthetic_data_creator import create_subspace_data
    from cluspy.density.MultiDensityDBSCAN import MultiDensityDBSCAN
    from cluspy.subspace.SubKmeans import SubKmeans
    from cluspy.estimatek.XMeans import XMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, adjusted_rand_score as ars, silhouette_score as sc
    X, L = create_subspace_data(1500, total_features=2)
    algorithms = [EvaluationAlgorithm("MDDBSCAN_k=15", MultiDensityDBSCAN, {"k":15}), EvaluationAlgorithm("MDDBSCAN_k=25", MultiDensityDBSCAN, {"k":25}),
                  EvaluationAlgorithm("Xmeans", XMeans), EvaluationAlgorithm("SubKmeans", SubKmeans, {"n_clusters":3}, 0)]
    metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("AMI", ami), EvaluationMetric("Adjusted rand", ars), EvaluationMetric("Silhouette", sc, use_gt=False)]
    df = evaluate(X, algorithms, metrics, L, 10, True, "test_results.csv")

    :param X: dataset
    :param evaluation_algorithms: input algorithms - list of EvaluationAlgorithm
    :param evaluation_metrics: input metrics - list of EvaluationMetric (can be None)
    :param GT: ground truth (Default: None)
    :param repetitions: number of repetitions to execute
    :param save_path: path where the results should be saved (optional)
    :return: dataframe with evaluation results
    """
    assert evaluation_metrics is not None or add_runtime, "Either evaluation metrics must be defined or add_runtime must be True"
    if type(evaluation_algorithms) is not list:
        evaluation_algorithms = [evaluation_algorithms]
    if type(evaluation_metrics) is not list and evaluation_metrics is not None:
        evaluation_metrics = [evaluation_metrics]
    algo_names = [a.name for a in evaluation_algorithms]
    if add_runtime:
        metric_names = [] if evaluation_metrics is None else [m.name for m in evaluation_metrics]
        metric_names += ["runtime"]
    else:
        metric_names = [m.name for m in evaluation_metrics]
    header = pd.MultiIndex.from_product([algo_names, metric_names], names=["algorithm", "metric"])
    data = np.zeros((repetitions, len(algo_names) * len(metric_names)))
    df = pd.DataFrame(data, columns=header, index=range(repetitions))
    for eval_algo in evaluation_algorithms:
        assert type(eval_algo) is EvaluationAlgorithm, "The algortihms must be of type EvaluationAlgortihm"
        for rep in range(repetitions):
            start_time = time.time()
            algo_obj = eval_algo.obj(**eval_algo.params)
            algo_obj.fit(X)
            runtime = time.time() - start_time
            if add_runtime:
                df.at[rep, (eval_algo.name, "runtime")] = runtime
            labels = algo_obj.labels if eval_algo.label_column is None else algo_obj.labels[:, eval_algo.label_column]
            if evaluation_metrics is not None:
                for eval_metric in evaluation_metrics:
                    assert type(eval_metric) is EvaluationMetric, "The metrics must be of type EvaluationMetric"
                    if eval_metric.use_gt:
                        assert GT is not None, "Ground truth can not be None if it is used for the chosen metric"
                        result = eval_metric.method(GT, labels, **eval_metric.params)
                    else:
                        result = eval_metric.method(X, labels, **eval_metric.params)
                    df.at[rep, (eval_algo.name, eval_metric.name)] = result
    if save_path is not None:
        df.to_csv(save_path)
    return df


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

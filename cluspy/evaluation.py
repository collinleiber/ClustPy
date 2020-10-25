import pandas as pd
import numpy as np

def get_scores(X, GT, evaluation_algorithms, evaluation_metrics, repetitions=10, save_path=None):
    """
    Example:
    from cluspy.data.synthetic_data_creator import create_subspace_data
    from cluspy.density.MultiDensityDBSCAN import MultiDensityDBSCAN
    from cluspy.subspace.SubKmeans import SubKmeans
    from cluspy.estimatek.XMeans import XMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, adjusted_rand_score as ars
    X, L = create_subspace_data(1500)
    algorithms = [EvaluationAlgorithm("MDDBSCAN_k=15", MultiDensityDBSCAN, {"k":15}), EvaluationAlgorithm("MDDBSCAN_k=25", MultiDensityDBSCAN, {"k":25}), EvaluationAlgorithm("Xmeans", XMeans, {}),
                  EvaluationAlgorithm("SubKmeans", SubKmeans, {"n_clusters":3}, 0)]
    metrics = [EvaluationMetric("NMI", nmi, {}), EvaluationMetric("AMI", ami, {}), EvaluationMetric("Adjusted rand", ars, {})]
    df = get_scores(X, L, algorithms, metrics, 10, "test_results.csv")

    :param X: dataset
    :param GT: ground truth
    :param evaluation_algorithms: input algorithms - list of EvaluationAlgorithm
    :param evaluation_metrics: input metrics - list of EvaluationMetric
    :param repetitions: number of repetitions to execute
    :param save_path: path where the results should be saved (optional)
    :return: dataframe with evaluation results
    """
    algo_names = [a.name for a in evaluation_algorithms]
    metric_names = [m.name for m in evaluation_metrics]
    header = pd.MultiIndex.from_product([algo_names, metric_names], names=["algorithm", "metric"])
    data = np.zeros((repetitions, len(algo_names) * len(metric_names)))
    df = pd.DataFrame(data, columns=header, index=range(repetitions))
    for eval_algo in evaluation_algorithms:
        assert type(eval_algo) is EvaluationAlgorithm, "The algortihms must be of type EvaluationAlgortihm"
        for rep in range(repetitions):
            algo_obj = eval_algo.obj(**eval_algo.params)
            algo_obj.fit(X)
            labels = algo_obj.labels if eval_algo.label_column is None else algo_obj.labels[:, eval_algo.label_column]
            for eval_metric in evaluation_metrics:
                assert type(eval_metric) is EvaluationMetric, "The metrics must be of type EvaluationMetric"
                result = eval_metric.method(GT, labels, **eval_metric.params)
                df.at[rep, (eval_algo.name, eval_metric.name)] = result
    if save_path is not None:
        df.to_csv(save_path)
    return df

class EvaluationMetric():
    def __init__(self, name, method, params = {}):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert callable(method), "method must be a method"
        self.method = method
        assert type(params) is dict, "params must be a dict"
        self.params = params

class EvaluationAlgorithm():
    def __init__(self, name, obj, params = {}, label_column = None):
        assert type(name) is str, "name must be a string"
        self.name = name
        assert type(obj) is type, "name must be Algorithm class"
        self.obj = obj
        assert type(params) is dict, "params must be a dict"
        self.params = params
        assert label_column is None or type(label_column) is int, "label_column must be None or int"
        self.label_column = label_column

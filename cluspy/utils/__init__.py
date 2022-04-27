from .evaluation import evaluate_dataset, evaluate_multiple_datasets, EvaluationDataset, EvaluationAlgorithm, EvaluationMetric
from .diptest import dip_test, dip_pval, dip_boot_samples, dip_gradient, dip_pval_gradient
from .plots import plot_with_transformation, plot_image, plot_scatter_matrix, plot_histogram, plot_1d_data

__all__ = ['evaluate_dataset',
           'evaluate_multiple_datasets',
           'EvaluationMetric',
           'EvaluationAlgorithm',
           'EvaluationDataset',
           'dip_test',
           'dip_pval',
           'dip_boot_samples',
           'plot_with_transformation',
           'plot_image',
           'plot_scatter_matrix',
           'plot_histogram',
           'plot_1d_data',
           'dip_gradient',
           'dip_pval_gradient']
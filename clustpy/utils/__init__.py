from .evaluation import evaluate_dataset, evaluate_multiple_datasets, EvaluationDataset, \
    EvaluationAlgorithm, EvaluationMetric, evaluation_df_to_latex_table
from .diptest import dip_test, dip_pval, dip_boot_samples, dip_gradient, dip_pval_gradient, plot_dip
from .plots import plot_with_transformation, plot_image, plot_scatter_matrix, plot_histogram, plot_1d_data, \
    plot_2d_data, plot_3d_data

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
           'plot_2d_data',
           'plot_3d_data',
           'dip_gradient',
           'dip_pval_gradient',
           'plot_dip',
           'evaluation_df_to_latex_table']

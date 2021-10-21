from .diptest import dip, dip_test, dip_pval, dip_boot_samples, PVAL_BY_BOOT, PVAL_BY_TABLE, PVAL_BY_FUNCTION
from .plots import plot_with_transformation, plot_image, plot_scatter_matrix, plot_histogram, plot_1d_data

__all__ = ['dip',
           'dip_test',
           'dip_pval',
           'dip_boot_samples',
           'PVAL_BY_TABLE',
           'PVAL_BY_FUNCTION',
           'PVAL_BY_BOOT',
           'plot_with_transformation',
           'plot_image',
           'plot_scatter_matrix',
           'plot_histogram',
           'plot_1d_data']
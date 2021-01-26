from .diptest import dip, dip_test, dip_pval, dip_boot_samples, PVAL_BY_BOOT, PVAL_BY_TABLE, PVAL_BY_FUNCTION
from .plots import transformation_plot

__all__ = ['dip',
           'dip_test',
           'dip_pval',
           'dip_boot_samples',
           'PVAL_BY_TABLE',
           'PVAL_BY_FUNCTION',
           'PVAL_BY_BOOT',
           'transformation_plot']
from clustpy.deep import SHADE
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_shade_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(SHADE(min_points=2, pretrain_epochs=0, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_shade():
    shade = SHADE()
    _test_dc_algorithm_simple(shade, check_predict=False)


def test_shade_wo_matrix_distance():
    shade = SHADE(use_matrix_dc_distance=False)
    _test_dc_algorithm_simple(shade, check_predict=False)


def test_shade_wo_complete_dc_tree():
    shade = SHADE(use_complete_dc_tree=False)
    _test_dc_algorithm_simple(shade, check_predict=False)

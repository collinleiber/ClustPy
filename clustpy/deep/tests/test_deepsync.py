from clustpy.deep import DeepSynC
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_deepsync_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DeepSynC(pretrain_epochs=10, clustering_max_epochs=20, k_nearest_neighbors=10),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_deepsync():
    deepsync = DeepSynC()
    _test_dc_algorithm_simple(deepsync, check_predict=False)

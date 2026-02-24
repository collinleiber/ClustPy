from clustpy.data.tests._helpers_for_tests import _helper_test_data_loader
from clustpy.data import load_banknotes, load_spambase, load_seeds, load_skin, load_soybean_small, load_soybean_large, \
    load_pendigits, load_ecoli, load_htru2, load_letterrecognition, load_har, load_statlog_shuttle, load_mice_protein, \
    load_user_knowledge, load_breast_tissue, load_forest_types, load_dermatology, load_multiple_features, \
    load_statlog_australian_credit_approval, load_breast_cancer_wisconsin_original, load_optdigits, load_semeion, \
    load_cmu_faces, load_gene_expression_cancer_rna_seq, load_sport_articles, load_wholesale_customers, load_reuters21578
import pytest
import shutil


@pytest.fixture(autouse=True, scope='function')
def my_tmp_dir(tmp_path):
    # Code that will run before the tests
    tmp_dir = str(tmp_path)
    # Test functions will be run at this point
    yield tmp_dir
    # Code that will run after the tests
    shutil.rmtree(tmp_dir)


@pytest.mark.data
def test_load_banknotes(my_tmp_dir):
    _helper_test_data_loader(load_banknotes, 1372, 4, 2, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_spambase(my_tmp_dir):
    _helper_test_data_loader(load_spambase, 4601, 57, 2, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_seeds(my_tmp_dir):
    _helper_test_data_loader(load_seeds, 210, 7, 3, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_skin(my_tmp_dir):
    _helper_test_data_loader(load_skin, 245057, 3, 2, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_soybean_small(my_tmp_dir):
    _helper_test_data_loader(load_soybean_small, 47, 35, 4, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_soybean_large(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_soybean_large, 562, 35, 15,
                             dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_soybean_large, 266, 35, 15,
                             dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_soybean_large, 296, 35, 15,
                             dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_pendigits(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_pendigits, 10992, 16, 10,
                             dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_pendigits, 7494, 16, 10,
                             dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_pendigits, 3498, 16, 10,
                             dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_ecoli(my_tmp_dir):
    _helper_test_data_loader(load_ecoli, 336, 7, 8, dataloader_params={"downloads_path": my_tmp_dir})
    # Check if ignoring small clusters works
    _helper_test_data_loader(load_ecoli, 327, 7, 5,
                             dataloader_params={"ignore_small_clusters": True, "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_hrtu2(my_tmp_dir):
    _helper_test_data_loader(load_htru2, 17898, 8, 2, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_letterrecognition(my_tmp_dir):
    _helper_test_data_loader(load_letterrecognition, 20000, 16, 26,
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_har(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_har, 10299, 561, 6,
                             dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_har, 7352, 561, 6,
                             dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_har, 2947, 561, 6,
                             dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_statlog_shuttle(my_tmp_dir):
    # 7z probably not installed! -> data and labels can be None
    dataset = load_statlog_shuttle(downloads_path=my_tmp_dir)
    if dataset is not None:
        # Full data set
        _helper_test_data_loader(load_statlog_shuttle, 58000, 9, 7,
                                 dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
        # Train data set
        _helper_test_data_loader(load_statlog_shuttle, 43500, 9, 7,
                                 dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
        # Test data set
        _helper_test_data_loader(load_statlog_shuttle, 14500, 9, 7,
                                 dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_mice_protein(my_tmp_dir):
    _helper_test_data_loader(load_mice_protein, 1077, 68, 8, dataloader_params={"downloads_path": my_tmp_dir})
    # Check if additional labels work
    _helper_test_data_loader(load_mice_protein, 1077, 68, [8, 72, 2, 2, 2],
                             dataloader_params={"return_additional_labels": True, "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_user_knowledge(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_user_knowledge, 403, 5, 4,
                             dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_user_knowledge, 258, 5, 4,
                             dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_user_knowledge, 145, 5, 4,
                             dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_breast_tissue(my_tmp_dir):
    _helper_test_data_loader(load_breast_tissue, 106, 9, 6, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_forest_types(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_forest_types, 523, 27, 4,
                             dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
    # Train data set
    _helper_test_data_loader(load_forest_types, 198, 27, 4,
                             dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Test data set
    _helper_test_data_loader(load_forest_types, 325, 27, 4,
                             dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_dermatology(my_tmp_dir):
    _helper_test_data_loader(load_dermatology, 358, 34, 6, dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_multiple_features(my_tmp_dir):
    _helper_test_data_loader(load_multiple_features, 2000, 649, 10,
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_statlog_australian_credit_approval(my_tmp_dir):
    _helper_test_data_loader(load_statlog_australian_credit_approval, 690, 14, 2,
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_breast_cancer_wisconsin_original(my_tmp_dir):
    _helper_test_data_loader(load_breast_cancer_wisconsin_original, 683, 9, 2,
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_optdigits(my_tmp_dir):
    # Full data set
    dataset = _helper_test_data_loader(load_optdigits, 5620, 64, 10,
                                       dataloader_params={"subset": "all", "downloads_path": my_tmp_dir})
    # Non-flatten
    assert dataset.images.shape == (5620, 8, 8)
    assert dataset.image_format == "HW"
    # Train data set
    dataset = _helper_test_data_loader(load_optdigits, 3823, 64, 10,
                                       dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Non-flatten
    assert dataset.images.shape == (3823, 8, 8)
    assert dataset.image_format == "HW"
    # Test data set
    dataset = _helper_test_data_loader(load_optdigits, 1797, 64, 10,
                                       dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})
    # Non-flatten
    assert dataset.images.shape == (1797, 8, 8)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_semeion(my_tmp_dir):
    dataset = _helper_test_data_loader(load_semeion, 1593, 256, 10,
                                       dataloader_params={"downloads_path": my_tmp_dir})
    # Non-flatten
    assert dataset.images.shape == (1593, 16, 16)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_cmu_faces(my_tmp_dir):
    dataset = _helper_test_data_loader(load_cmu_faces, 624, 960, [20, 4, 4, 2],
                                       dataloader_params={"downloads_path": my_tmp_dir})
    # Non-flatten
    assert dataset.images.shape == (624, 30, 32)
    assert dataset.image_format == "HW"


@pytest.mark.data
def test_load_gene_expression_cancer_rna_seq(my_tmp_dir):
    _helper_test_data_loader(load_gene_expression_cancer_rna_seq, 801, 20531, 5,
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_sport_articles(my_tmp_dir):
    _helper_test_data_loader(load_sport_articles, 1000, 55, 2,
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_wholesale_customers(my_tmp_dir):
    _helper_test_data_loader(load_wholesale_customers, 440, 6, [2, 3],
                             dataloader_params={"downloads_path": my_tmp_dir})


@pytest.mark.data
def test_load_reuters21578(my_tmp_dir):
    # Full data set
    _helper_test_data_loader(load_reuters21578, 8367, 2000, 5,
                             dataloader_params={"downloads_path": my_tmp_dir})
    # Lewis train data
    _helper_test_data_loader(load_reuters21578, 5791, 2000, 5,
                             dataloader_params={"subset": "train", "downloads_path": my_tmp_dir})
    # Lewis test data
    _helper_test_data_loader(load_reuters21578, 2300, 2000, 5,
                             dataloader_params={"subset": "test", "downloads_path": my_tmp_dir})
    # cgi train data
    _helper_test_data_loader(load_reuters21578, 8091, 2000, 5,
                             dataloader_params={"subset": "train-cgi", "downloads_path": my_tmp_dir})
    # cgi test data
    _helper_test_data_loader(load_reuters21578, 276, 2000, 5,
                             dataloader_params={"subset": "test-cgi", "downloads_path": my_tmp_dir})

from .synthetic_data_creator import create_subspace_data, create_nr_data
from .real_world_data import load_har, load_letterrecognition, \
    load_optdigits, load_pendigits, load_newsgroups, load_iris, load_wine, load_breast_cancer, load_reuters, \
    load_banknotes, load_htru2, load_mice_protein, load_ecoli, load_spambase, load_seeds, \
    load_statlog_shuttle, load_forest_types, load_breast_tissue, load_soybean_large, load_soybean_small, load_skin, \
    load_user_knowledge, load_dermatology, load_multiple_features, load_statlog_australian_credit_approval, \
    load_breast_cancer_wisconsin_original, load_semeion, load_imagenet_dog, load_imagenet10
from .real_timeseries_data import load_motestrain, load_olive_oil, load_symbols, load_diatom_size_reduction, \
    load_proximal_phalanx_outline, load_plane, load_sony_aibo_robot_surface, load_two_patterns, load_lsst
from .real_nr_data import load_aloi_small, load_fruit, load_nrletters, load_stickfigures, load_webkb, load_cmu_faces
from .real_torchvision_data import load_usps, load_mnist, load_fmnist, load_kmnist, load_svhn, load_cifar10, load_stl10
from .real_medical_mnist_data import load_path_mnist, load_chest_mnist, load_derma_mnist, load_oct_mnist, \
    load_pneumonia_mnist, load_retina_mnist, load_breast_mnist, load_blood_mnist, load_tissue_mnist, load_organ_a_mnist, \
    load_organ_c_mnist, load_organ_s_mnist, load_organ_mnist_3d, load_nodule_mnist_3d, load_adrenal_mnist_3d, \
    load_fracture_mnist_3d, load_vessel_mnist_3d, load_synapse_mnist_3d

__all__ = ['create_subspace_data',
           'create_nr_data',
           'load_har',
           'load_usps',
           'load_mnist',
           'load_fmnist',
           'load_kmnist',
           'load_letterrecognition',
           'load_optdigits',
           'load_pendigits',
           'load_newsgroups',
           'load_iris',
           'load_wine',
           'load_breast_cancer',
           'load_reuters',
           'load_banknotes',
           'load_htru2',
           'load_motestrain',
           'load_cmu_faces',
           'load_mice_protein',
           'load_webkb',
           'load_ecoli',
           'load_spambase',
           'load_seeds',
           'load_statlog_shuttle',
           'load_olive_oil',
           'load_symbols',
           'load_diatom_size_reduction',
           'load_proximal_phalanx_outline',
           'load_forest_types',
           'load_breast_tissue',
           'load_soybean_large',
           'load_soybean_small',
           'load_skin',
           'load_user_knowledge',
           'load_plane',
           'load_aloi_small',
           'load_stickfigures',
           'load_nrletters',
           'load_fruit',
           'load_stl10',
           'load_cifar10',
           'load_svhn',
           'load_dermatology',
           'load_multiple_features',
           'load_statlog_australian_credit_approval',
           'load_breast_cancer_wisconsin_original',
           'load_semeion',
           'load_sony_aibo_robot_surface',
           'load_two_patterns',
           'load_lsst',
           'load_path_mnist',
           'load_chest_mnist',
           'load_derma_mnist',
           'load_oct_mnist',
           'load_pneumonia_mnist',
           'load_retina_mnist',
           'load_breast_mnist',
           'load_blood_mnist',
           'load_tissue_mnist',
           'load_organ_a_mnist',
           'load_organ_c_mnist',
           'load_organ_s_mnist',
           'load_organ_mnist_3d',
           'load_nodule_mnist_3d',
           'load_adrenal_mnist_3d',
           'load_fracture_mnist_3d',
           'load_vessel_mnist_3d',
           'load_synapse_mnist_3d',
           'load_imagenet_dog',
           'load_imagenet10']

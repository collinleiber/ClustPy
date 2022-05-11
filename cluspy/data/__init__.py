from .synthetic_data_creator import create_subspace_data, create_nr_data
from .real_world_data import load_har, load_letterrecognition, \
    load_optdigits, load_pendigits, load_newsgroups, load_iris, load_wine, load_breast_cancer, load_reuters, \
    load_banknotes, load_htru2, load_motestrain, load_mice_protein, load_ecoli, load_spambase, load_seeds, \
    load_statlog_shuttle, load_olive_oil, load_symbols, load_diatom_size_reduction, load_proximal_phalanx_outline, \
    load_forest_types, load_breast_tissue, load_soybean_large, load_soybean_small, load_skin, load_user_knowledge, \
    load_plane
from .real_nr_data import load_aloi_small, load_fruit, load_nrletters, load_stickfigures, load_webkb, load_cmu_faces
from .real_torchvision_data import load_usps, load_mnist, load_fmnist, load_kmnist, load_svhn, load_cifar10, load_stl10

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
           'load_svhn']

from clustpy.deep import DEC, IDEC, get_default_augmented_dataloaders
from clustpy.data import create_subspace_data, load_optdigits
import numpy as np
import torch
from sklearn.utils.estimator_checks import check_estimator


def test_dec_estimator():
    check_estimator(DEC(3, pretrain_epochs=3, clustering_epochs=3), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_idec_estimator():
    check_estimator(IDEC(3, pretrain_epochs=3, clustering_epochs=3), 
                    {"check_complex_data": "this check is expected to fail because complex values are not supported"})


def test_simple_dec():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    dec = DEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(dec, "labels_")
    dec.fit(X)
    assert dec.labels_.dtype == np.int32
    assert dec.labels_.shape == labels.shape
    X_embed = dec.transform(X)
    assert X_embed.shape == (X.shape[0], dec.embedding_size)
    # Test if random state is working
    dec2 = DEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    dec2.fit(X)
    assert np.array_equal(dec.labels_, dec2.labels_)
    assert np.allclose(dec.cluster_centers_, dec2.cluster_centers_, atol=1e-1)
    assert np.array_equal(dec.dec_labels_, dec2.dec_labels_)
    assert np.allclose(dec.dec_cluster_centers_, dec2.dec_cluster_centers_, atol=1e-1)
    # Test predict
    labels_predict = dec.predict(X)
    assert np.array_equal(dec.labels_, labels_predict)


def test_simple_idec():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    idec = IDEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(idec, "labels_")
    idec.fit(X)
    assert idec.labels_.dtype == np.int32
    assert idec.labels_.shape == labels.shape
    X_embed = idec.transform(X)
    assert X_embed.shape == (X.shape[0], idec.embedding_size)
    # Test if random state is working
    # TODO Does not work every time -> Check why
    # idec2 = IDEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    # idec2.fit(X)
    # assert np.array_equal(idec.labels_, idec2.labels_)
    # assert np.allclose(idec.cluster_centers_, idec2.cluster_centers_, atol=5e-1)
    # assert np.array_equal(idec.dec_labels_, idec2.dec_labels_)
    # assert np.allclose(idec.dec_cluster_centers_, idec2.dec_cluster_centers_, atol=5e-1)
    # Test predict
    labels_predict = idec.predict(X)
    assert np.array_equal(idec.labels_, labels_predict)


def test_dec_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = get_default_augmented_dataloaders(data)
    dec = DEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
              custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(dec, "labels_")
    dec.fit(data)
    assert dec.labels_.dtype == np.int32
    assert dec.labels_.shape == labels.shape


def test_idec_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = get_default_augmented_dataloaders(data)
    idec = IDEC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(idec, "labels_")
    idec.fit(data)
    assert idec.labels_.dtype == np.int32
    assert idec.labels_.shape == labels.shape

from clustpy.deep import ENRC, ACeDeC, get_default_augmented_dataloaders
from clustpy.data import create_nr_data, create_subspace_data, load_optdigits
import numpy as np
import torch
from clustpy.utils.checks import check_clustpy_estimator
import pytest


@pytest.mark.skip(reason="Checks are not completly implemented yet")
def test_acedec_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(ACeDeC(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


@pytest.mark.skip(reason="Checks are not completly implemented yet")
def test_enrc_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_enrc():
    torch.use_deterministic_algorithms(True)
    X, labels = create_nr_data(1000, subspace_features=(3, 3, 50), random_state=1)
    labels = labels[:, :-1]  # ignore noise space
    enrc = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(enrc, "labels_")
    enrc.fit(X)
    assert enrc.labels_.dtype == np.int32
    assert enrc.labels_.shape == labels.shape
    X_embed = enrc.transform(X)
    assert X_embed.shape == (X.shape[0], enrc.embedding_size)
    # Test if random state is working
    enrc2 = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, random_state=1, debug=True)
    enrc2.fit(X)
    assert np.array_equal(enrc.labels_, enrc2.labels_)
    for i in range(len(enrc.cluster_centers_)):
        assert np.allclose(enrc.cluster_centers_[i], enrc2.cluster_centers_[i])
    # Test if sgd as init is working
    enrc = ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3, init="sgd")
    enrc.fit(X)
    assert enrc.labels_.dtype == np.int32
    assert enrc.labels_.shape == labels.shape
    # Test predict
    labels_predict = enrc.predict(X)
    assert np.array_equal(enrc.labels_, labels_predict)


def test_simple_acedec():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 50), random_state=1)
    acedec = ACeDeC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    assert not hasattr(acedec, "labels_")
    acedec.fit(X)
    assert acedec.labels_.dtype == np.int32
    assert acedec.labels_.shape == labels.shape
    X_embed = acedec.transform(X)
    assert X_embed.shape == (X.shape[0], acedec.embedding_size)
    # Test if random state is working
    # TODO Does not work every time -> Check why
    # acedec2 = ACeDeC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1)
    # acedec2.fit(X)
    # assert np.array_equal(acedec.labels_, acedec2.labels_)
    # assert np.allclose(acedec.cluster_centers_[0], acedec2.cluster_centers_[0], atol=1e-1)
    # Test predict
    labels_predict = acedec.predict(X)
    assert np.array_equal(acedec.labels_, labels_predict)


def test_acedec_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = get_default_augmented_dataloaders(data)
    acedec = ACeDeC(3, pretrain_epochs=3, clustering_epochs=3, random_state=1,
                    custom_dataloaders=[aug_dl, orig_dl],
                    augmentation_invariance=True)
    assert not hasattr(acedec, "labels_")
    acedec.fit(data)
    assert acedec.labels_.dtype == np.int32
    assert acedec.labels_.shape == labels.shape

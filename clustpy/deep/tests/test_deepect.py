from clustpy.deep import DeepECT, get_default_augmented_dataloaders
from clustpy.deep.deepect import _DeepECT_Module, _DeepECT_ClusterTreeNode
from clustpy.hierarchical._cluster_tree import BinaryClusterTree
from clustpy.deep.tests._helpers_for_tests import _TestAutoencoder
from clustpy.deep._data_utils import get_dataloader
import numpy as np
import torch
from clustpy.data import create_subspace_data, load_optdigits


def test_DeepECT_ClusterTreeNode():
    cluster_tree = BinaryClusterTree(_DeepECT_ClusterTreeNode)
    dummy_ae = _TestAutoencoder(2, 2)
    optimizer = torch.optim.Adam(list(dummy_ae.parameters()), lr=1e-4)
    device = torch.device("cpu")
    # test set_center_weight_and_torch_labels
    left_node, right_node = cluster_tree.split_cluster(0)
    left_node.set_center_weight_and_torch_labels(np.array([0., 1.]), 1, optimizer, device)
    assert isinstance(left_node.center, torch.nn.Parameter)
    assert torch.equal(left_node.center, torch.tensor([0., 1.]))
    assert left_node.weight == 1
    assert torch.equal(left_node.torch_labels, torch.tensor([0]))
    right_node.set_center_weight_and_torch_labels(np.array([10., 11.]), 1.5, optimizer, device)
    assert isinstance(right_node.center, torch.nn.Parameter)
    assert torch.equal(right_node.center, torch.tensor([10., 11.]))
    assert right_node.weight == 1.5
    assert torch.equal(right_node.torch_labels, torch.tensor([1]))
    # second split -> test update_parents_torch_labels
    left_node_2, right_node_2 = cluster_tree.split_cluster(0)
    assert left_node.labels == [0, 2]
    assert torch.equal(left_node.torch_labels, torch.tensor([0]))
    assert not hasattr(cluster_tree.root_node_, "torch_labels")
    left_node_2.update_parents_torch_labels(device)
    assert torch.equal(left_node.torch_labels, torch.tensor([0, 2]))
    assert torch.equal(cluster_tree.root_node_.torch_labels, torch.tensor([0, 1, 2]))


def test_DeepECT_Module():
    cluster_tree = BinaryClusterTree(_DeepECT_ClusterTreeNode)
    dummy_ae = _TestAutoencoder(3, 2)
    optimizer = torch.optim.Adam(list(dummy_ae.parameters()), lr=1e-4)
    device = torch.device("cpu")
    deepect_module = _DeepECT_Module(cluster_tree, 20, 2, 0.1, False)
    # Prepare tree
    left_node, right_node = cluster_tree.split_cluster(0)
    left_node.set_center_weight_and_torch_labels(np.array([0., 2.]), 1, optimizer, device)
    right_node.set_center_weight_and_torch_labels(np.array([10., 11.]), 2, optimizer, device)
    left_left_node, left_right_node = cluster_tree.split_cluster(0, 5)
    left_left_node.set_center_weight_and_torch_labels(np.array([0., 0.]), 0., optimizer, device)
    left_right_node.set_center_weight_and_torch_labels(np.array([0., 4.]), 0.1, optimizer, device)
    # Check predict_hard
    embedded = torch.tensor([[0, 0], [10, 10], [3, 3], [12, 12], [1, 1]])
    predicted = deepect_module.predict_hard(embedded)
    assert torch.equal(predicted, torch.tensor([0, 1, 5, 1, 0]))
    # Check get_labels_from_leafs
    leaf_nodes = [right_node, left_left_node, left_right_node]
    leaf_centers, cluster_center_assignments, labels = deepect_module._get_labels_from_leafs(embedded, leaf_nodes)
    assert torch.equal(leaf_centers, torch.tensor([[10., 11.], [0., 0.], [0., 4.]]))
    assert torch.equal(cluster_center_assignments, torch.tensor([1, 0, 2, 0, 1]))
    assert torch.equal(labels, torch.tensor([0, 1, 5, 1, 0]))
    # Check grow_tree
    data = np.array([[0, 0, 0], [2, 3, 5], [1, 0, 2], [4, 3, 5], [1, 0, 0]])
    testloader = get_dataloader(data, 5)
    assert right_node.is_leaf_node()
    assert isinstance(right_node.center, torch.nn.Parameter)
    deepect_module._grow_tree(testloader, dummy_ae, leaf_nodes, 6, optimizer, device, np.random.RandomState(1))
    assert not right_node.is_leaf_node()
    assert not isinstance(right_node.center, torch.nn.Parameter) and isinstance(right_node.center, torch.Tensor)
    right_left_node = right_node.left_node_
    right_right_node = right_node.right_node_
    assert right_node.labels == [1, 6]
    assert torch.equal(right_node.torch_labels, torch.tensor([1, 6]))
    assert right_left_node.labels == [1]
    assert torch.equal(right_left_node.torch_labels, torch.tensor([1]))
    assert right_right_node.labels == [6]
    assert torch.equal(right_right_node.torch_labels, torch.tensor([6]))
    # Check _update_split_node_centers
    leaf_nodes = [left_left_node, left_right_node, right_left_node, right_right_node]
    leaf_node_centers = torch.stack([node.center.data for node in leaf_nodes], dim=0)
    leaf_node_weights = [node.weight for node in leaf_nodes]
    assert leaf_node_weights == [0, 0.1, 1, 1]
    split_nodes = [cluster_tree.root_node_, left_node, right_node]
    split_node_weights = [None if not hasattr(node, "weight") else node.weight for node in split_nodes]
    assert split_node_weights == [None, 1, 2]
    labels = torch.tensor([0, 1, 1, 6, 0])
    nodes_to_prune = deepect_module._update_split_node_centers(split_nodes, leaf_nodes, labels)
    assert torch.equal(leaf_node_centers, torch.stack([node.center.data for node in leaf_nodes], dim=0))
    assert torch.allclose(
        torch.tensor([[(1 * 0 + 2 * 10) / 3, (1 * 2 + 2 * 11) / 3], [(0 * 0 + 0.1 * 0) / 0.1, (0 * 0 + 0.1 * 4) / 0.1],
                      [(1 * 10 + 1 * 12) / 2, (1 * 10 + 1 * 12) / 2]], dtype=torch.double), torch.stack(
            [torch.tensor([-1, -1]) if not hasattr(node, "center") else node.center.data for node in split_nodes],
            dim=0))
    assert [node.weight for node in leaf_nodes] == [0.5 * 0 + 0.5 * 2, 0.5 * 0.1 + 0.5 * 0, 0.5 * 1 + 0.5 * 2,
                                                    0.5 * 1 + 0.5 * 1]
    assert [None if not hasattr(node, "weight") else node.weight for node in split_nodes] == [None, 0.5 * 1 + 0.5 * 2,
                                                                                              0.5 * 2 + 0.5 * 3]
    assert nodes_to_prune == [left_right_node]
    # Check _prune_tree
    assert cluster_tree.root_node_.labels == [0, 1, 5, 6]
    assert cluster_tree.root_node_.left_node_ == left_node
    deepect_module._prune_tree(nodes_to_prune, device)
    assert cluster_tree.root_node_.left_node_ == left_left_node
    assert cluster_tree.root_node_.labels == [0, 1, 6]


def test_simple_deepect():
    torch.use_deterministic_algorithms(True)
    X, labels = create_subspace_data(1000, subspace_features=(3, 10), random_state=1)
    deepect = DeepECT(pretrain_epochs=3, clustering_epochs=4, grow_interval=1, random_state=1)
    assert not hasattr(deepect, "labels_")
    deepect.fit(X)
    assert deepect.labels_.dtype == np.int32
    assert deepect.labels_.shape == labels.shape
    X_embed = deepect.transform(X)
    assert X_embed.shape == (X.shape[0], deepect.embedding_size)
    # Test if random state is working
    deepect2 = DeepECT(pretrain_epochs=3, clustering_epochs=4, grow_interval=1, random_state=1)
    deepect2.fit(X)
    assert np.array_equal(deepect.labels_, deepect2.labels_)
    # Test predict
    labels_predict = deepect.predict(X)
    assert np.array_equal(deepect.labels_, labels_predict)
    # Test flat clustering
    assert len(deepect.tree_.get_leaf_and_split_nodes()[0]) > 2
    assert np.unique(deepect.labels_).shape[0] > 2
    labels_flat = deepect.flat_clustering(2)
    assert np.unique(labels_flat).shape[0] == 2
    assert np.array_equal(labels_flat[(deepect.labels_ == 0) | (deepect.labels_ == 1)],
                          deepect.labels_[(deepect.labels_ == 0) | (deepect.labels_ == 1)])


def test_deepect_augmentation():
    torch.use_deterministic_algorithms(True)
    dataset = load_optdigits()
    data = dataset.images[:1000]
    labels = dataset.target[:1000]
    aug_dl, orig_dl = get_default_augmented_dataloaders(data)
    clusterer = DeepECT(pretrain_epochs=3, clustering_epochs=4, grow_interval=1, random_state=1,
                        custom_dataloaders=[aug_dl, orig_dl], augmentation_invariance=True)
    assert not hasattr(clusterer, "labels_")
    clusterer.fit(data)
    assert clusterer.labels_.dtype == np.int32
    assert clusterer.labels_.shape == labels.shape

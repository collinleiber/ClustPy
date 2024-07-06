from clustpy.deep.deepect import _DeepECT_Module
from clustpy.deep._deepect_utils import Cluster_Node
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from clustpy.data import load_mnist
import PIL


def test_Cluster_Node():
    root = Cluster_Node(np.array([0, 0]), "cpu")
    root.set_childs(None, np.array([-1, -1]), 0.5, np.array([1, 1]), 0.6, 0, 0)
    root.left_child.set_childs(
        None, np.array([-2, -2]), 0.7, np.array([-0.5, -0.5]), 0.8, 2, 1
    )

    # check if centers are stored correctly
    assert torch.all(torch.eq(root.center, torch.tensor([0, 0])))
    assert torch.all(torch.eq(root.left_child.center, torch.tensor([-1, -1])))
    assert torch.all(
        torch.eq(root.left_child.right_child.center, torch.tensor([-0.5, -0.5]))
    )
    # since the left node of the root node changed to a inner node, its weights must be non zero
    assert root.left_child.weight != None
    assert not root.left_child.is_leaf_node()
    # right child of root is still a leaf node
    assert root.right_child.is_leaf_node()
    assert root.left_child.right_child.is_leaf_node()
    # centers of leaf nodes must be trainable
    assert isinstance(root.left_child.right_child.center, torch.nn.Parameter)
    # centers of inner nodes are just tensors
    assert isinstance(root.left_child.center, torch.Tensor)
    assert root.left_child.weight.item() == torch.tensor(0.5, dtype=torch.float)
    assert root.left_child.right_child.weight.item() == torch.tensor(
        0.8, dtype=torch.float
    )


def sample_cluster_tree(get_deep_ect_module=False):
    """
    Helper method for creating a sample cluster tree
    """
    deep_ect = _DeepECT_Module(
        np.array([[0, 0], [1, 1]]),
        "cpu",
        random_state=np.random.RandomState(42),
    )
    tree = deep_ect.cluster_tree
    tree.root.left_child.set_childs(
        None,
        np.array([-2, -2]),
        0.7,
        np.array([-0.5, -0.5]),
        0.8,
        max_id=2,
        max_split_id=1,
    )

    if get_deep_ect_module:
        return deep_ect
    return tree


def encode_requires_grad(batch: torch.Tensor):
    batch.requires_grad = True
    return batch


def sample_cluster_tree_with_assignments():
    """
    Helper method for creating a sample cluster tree with assignments
    """
    # create mock-autoencoder, which represents just an identity function
    autoencoder = type("Autoencoder", (), {"encode": encode_requires_grad})

    tree = sample_cluster_tree()
    minibatch = torch.tensor([[-3, -3], [10, 10], [-0.4, -0.4], [0.4, 0.3]])
    tree.assign_to_nodes(autoencoder.encode(minibatch))
    tree._assign_to_splitnodes(tree.root)
    return tree


def test_cluster_tree():
    tree = sample_cluster_tree()
    # we should have 3 leaf nodes in this example
    assert len(tree.leaf_nodes) == 3

    # check if the returned nodes are really the leaf nodes by checking the stored center
    leaf_nodes = tree.leaf_nodes
    assert (
        torch.all(
            leaf_nodes[0].center
            == torch.nn.Parameter(torch.tensor([-2, -2], dtype=torch.float16))
        ).item()
        and torch.all(
            leaf_nodes[1].center
            == torch.nn.Parameter(torch.tensor([-0.5, -0.5], dtype=torch.float16))
        ).item()
        and torch.all(
            leaf_nodes[2].center
            == torch.nn.Parameter(torch.tensor([1, 1], dtype=torch.float16))
        ).item()
    )


def test_cluster_tree_growth():
    tree = sample_cluster_tree()
    optimizer = torch.optim.Adam([{"params": leaf.center} for leaf in tree.leaf_nodes])
    encode = lambda x: x
    autoencoder = type("Autoencoder", (), {"encode": encode})
    dataset = torch.tensor([[-3, -3], [10, 10], [-0.4, -0.4], [0.4, 0.3]], device="cpu")
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor([0, 1, 2, 3]), dataset),
        batch_size=2,
    )
    tree.grow_tree(dataloader, autoencoder, optimizer, "cpu")
    assert torch.allclose(
        torch.tensor([10.0, 10.0]), tree.root.right_child.right_child.center
    ) or torch.allclose(
        torch.tensor([10.0, 10.0]), tree.root.right_child.left_child.center
    )
    assert torch.allclose(
        torch.tensor([0.4, 0.3]), tree.root.right_child.right_child.center
    ) or torch.allclose(
        torch.tensor([0.4, 0.3]), tree.root.right_child.left_child.center
    )


def test_cluster_tree_assignment():

    tree = sample_cluster_tree_with_assignments()

    # check if all assignments made correct
    assert torch.all(
        torch.eq(tree.root.left_child.left_child.assignments, torch.tensor([[-3, -3]]))
    )
    assert torch.all(
        torch.eq(
            tree.root.left_child.right_child.assignments,
            torch.tensor([[-0.4, -0.4]]),
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.right_child.assignments, torch.tensor([[10, 10], [0.4, 0.3]])
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.assignments,
            torch.tensor([[-3, -3], [-0.4, -0.4], [10, 10], [0.4, 0.3]]),
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.left_child.assignments, torch.tensor([[-3, -3], [-0.4, -0.4]])
        )
    )

    # check if indexes are set correctly
    assert torch.all(
        torch.eq(
            tree.root.left_child.left_child.assignment_indices.sort()[0],
            torch.tensor([0]),
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.left_child.right_child.assignment_indices.sort()[0],
            torch.tensor([2]),
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.right_child.assignment_indices.sort()[0], torch.tensor([1, 3])
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.assignment_indices.sort()[0],
            torch.tensor([0, 1, 2, 3]),
        )
    )
    assert torch.all(
        torch.eq(
            tree.root.left_child.assignment_indices.sort()[0], torch.tensor([0, 2])
        )
    )


def test_predict():
    # get initial setting
    tree = sample_cluster_tree(get_deep_ect_module=True)
    # create mock data set
    dataset = torch.tensor([[-3, -3], [10, 10], [-0.4, -0.4], [0.4, 0.3]], device="cpu")
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor([0, 1, 2, 3]), dataset),
        batch_size=4,
    )
    # create mock-autoencoder, which represents just an identity function
    encode = lambda x: x
    autoencoder = type("Autoencoder", (), {"encode": encode})
    # predict 3 classes
    pred_tree = tree.predict(dataloader, autoencoder)
    assert pred_tree.flat_accuracy(np.array([0, 2, 1, 2]), 3) == 1
    # predict 2 classes
    pred_tree = tree.predict(dataloader, autoencoder)
    assert pred_tree.flat_accuracy(np.array([0, 1, 0, 1]), 2) == 1
    # predict 2 classes with batches
    dataset = torch.tensor([[-3, -3], [10, 10], [-0.4, -0.4], [0.4, 0.3]], device="cpu")
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor([0, 1, 2, 3]), dataset),
        batch_size=2,
    )
    pred_tree = tree.predict(dataloader, autoencoder)
    assert pred_tree.flat_accuracy(np.array([0, 1, 0, 1]), 2) == 1


def test_nc_loss():

    tree = sample_cluster_tree_with_assignments()

    # calculate nc loss for the above example
    loss = tree.nc_loss()

    # calculate nc loss for the above example manually
    loss_left_left_node = torch.sqrt(
        torch.sum((torch.tensor([-2, -2]) - torch.tensor([-3, -3])) ** 2)
    )
    loss_left_right_node = torch.sqrt(
        torch.sum((torch.tensor([-0.5, -0.5]) - torch.tensor([-0.4, -0.4])) ** 2)
    )
    loss_right_node = torch.sqrt(
        torch.sum(
            (
                torch.tensor([1, 1])
                - (torch.tensor([10, 10]) + torch.tensor([0.4, 0.3])) / 2
            )
            ** 2
        )
    )
    loss_test = (loss_left_left_node + loss_left_right_node + loss_right_node) / 3

    assert torch.all(torch.eq(loss, loss_test))

def test_adaption_inner_nodes():
    tree = sample_cluster_tree_with_assignments()

    tree.adapt_inner_nodes(tree.root)

    # calculate adaption manually
    root_weigth = torch.tensor([0.5 * (1 + 2), 0.5 * (1 + 2)])
    root_left_weight = torch.tensor([0.5 * (0.7 + 1), 0.5 * (0.8 + 1)])
    new_root_left = (
        root_left_weight[0] * torch.tensor([-2, -2])
        + root_left_weight[1] * torch.tensor([-0.5, -0.5])
    ) / (torch.sum(root_left_weight))
    new_root = (
        root_weigth[0] * new_root_left + root_weigth[1] * torch.tensor([1, 1])
    ) / (torch.sum(root_weigth))

    # compare results
    assert torch.all(torch.eq(tree.root.center, new_root))
    assert torch.all(torch.eq(tree.root.left_child.center, new_root_left))


def test_augmentation():

    dataset = load_mnist()
    data = dataset["data"] * 0.02

    degrees = (-15, 15)
    translation = (0.14, 0.14)
    image_size = 28

    image_max_value = np.max(data)
    image_min_value = np.min(data)

    augmentation_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x - image_min_value),
            transforms.Lambda(lambda x: x / image_max_value),  # [0,1]
            transforms.Lambda(lambda x: x.reshape(image_size, image_size)),
            transforms.ToPILImage(),  # [0,255]
            transforms.RandomAffine(
                degrees=degrees,
                shear=degrees,
                translate=translation,
                interpolation=PIL.Image.BILINEAR,
            ),
            transforms.ToTensor(),  # back to [0,1] again
            transforms.Lambda(lambda x: x.reshape(image_size**2)),
            transforms.Lambda(
                lambda x: x * image_max_value
            ),  # back to original data range
            transforms.Lambda(lambda x: x + image_min_value),
        ]
    )

    class Original_Dataset(Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            original_image = self.original_dataset[idx]
            return idx, original_image

    class Augmented_Dataset(Dataset):

        def __init__(self, original_dataset, augmentation_transform):
            self.original_dataset = original_dataset
            self.augmentation_transform = augmentation_transform

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            original_image = self.original_dataset[idx]
            augmented_image = self.augmentation_transform(original_image)
            return idx, original_image, augmented_image

    # Create an instance of the datasets
    original_dataset = Original_Dataset(data)
    augmented_dataset = Augmented_Dataset(data, augmentation_transform)

    # Create the dataloaders
    trainloader = DataLoader(augmented_dataset, batch_size=256, shuffle=True)
    testloader = DataLoader(original_dataset, batch_size=256, shuffle=False)

    idx, M, M_aug = next(iter(trainloader))
    _, M_test = next(iter(testloader))
    assert len(idx) == len(M)
    assert M.shape[0] == 256 and M.shape[1] == 784
    assert M.shape == M_aug.shape
    assert M_test.shape == M.shape
    # check if scaling is consistent
    assert torch.max(M) == torch.max(M_aug)
    assert torch.min(M) == torch.min(M_aug)

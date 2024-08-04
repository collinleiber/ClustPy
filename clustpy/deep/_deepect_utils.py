from typing import List, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

import os
import sys

sys.path.append(os.getcwd())

from clustpy.metrics.hierarchical_metrics import (
    PredictionClusterNode,
    PredictionClusterTree,
)


class Cluster_Node:
    """
    This class represents a cluster node within a binary cluster tree.
    Each node in a cluster tree represents a cluster. The cluster is
    stored through its center (self.center). During the assignment of
    a new minibatch to the cluster tree, each node stores the samples
    which are nearest to its center (self.assignments).
    The centers of leaf nodes are optimized through autograd, whereas
    the center of inner nodes are adapted analytically with weights
    for each of its child stored in self.weights.
    """

    def __init__(
        self,
        center: np.ndarray,
        device: torch.device,
        id: int = 0,
        parent: "Cluster_Node" = None,
        split_id: int = 0,
        weight: int = 1,
    ) -> "Cluster_Node":
        """
        Constructor for the Cluster_Node class.

        Parameters
        ----------
        center : np.ndarray
            The initial center for this node.
        device : torch.device
            The device to be trained on.
        id : int, optional
            The ID of the node, by default 0.
        parent : Cluster_Node, optional
            The parent node, by default None.
        split_id : int, optional
            The split ID of the node, by default 0.
        weight : int, optional
            The weight of the node, by default 1.
        """
        self.device = device
        self.left_child = None
        self.right_child = None
        self.weight = torch.tensor(weight, dtype=torch.float)
        self.center = torch.nn.Parameter(
            torch.tensor(
                center, requires_grad=True, device=self.device, dtype=torch.float
            )
        )
        self.assignments: Union[torch.Tensor, None] = None
        self.assignment_indices: Union[torch.Tensor, None] = None
        self.sum_squared_dist: Union[torch.Tensor, None] = None
        self.id = id
        self.split_id = split_id
        self.parent = parent

    def clear_assignments(self):
        """
        Clears the assignments for the node and its children.
        """
        if self.left_child is not None:
            self.left_child.clear_assignments()
        if self.right_child is not None:
            self.right_child.clear_assignments()
        self.assignments = None
        self.assignment_indices = None
        self.sum_squared_dist = None

    def is_leaf_node(self):
        """
        Checks whether the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return self.left_child is None and self.right_child is None

    def from_leaf_to_inner(self):
        """
        Converts a leaf node to an inner node. Weights for
        its child are initialized and the centers are not trainable anymore.
        """
        self.center.requires_grad = False
        self.assignments = None
        self.sum_squared_dist = None

    def prune(self):
        """
        Prunes the node and its children by converting them to inner nodes.
        """
        if self.left_child is not None:
            self.left_child.prune()
        if self.right_child is not None:
            self.right_child.prune()
        self.from_leaf_to_inner()

    def set_childs(
        self,
        optimizer: Union[torch.optim.Optimizer, None],
        left_child_centers: np.ndarray,
        left_child_weight: int,
        right_child_centers: np.ndarray,
        right_child_weight: int,
        max_id: int = 0,
        max_split_id: int = 0,
    ):
        """
        Sets new children to this cluster node and changes
        this node to an inner node.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer, optional
            The optimizer to add the new child centers to.
        left_child_centers : np.ndarray
            Initial centers for the left child.
        left_child_weight : int
            Initial weight for the left child.
        right_child_centers : np.ndarray
            Initial centers for the right child.
        right_child_weight : int
            Initial weight for the right child.
        max_id : int, optional
            The maximum ID used for the new children, by default 0.
        max_split_id : int, optional
            The maximum split ID used for the new children, by default 0.
        """
        self.left_child = Cluster_Node(
            left_child_centers,
            self.device,
            max_id + 1,
            self,
            max_split_id + 1,
            left_child_weight,
        )
        self.right_child = Cluster_Node(
            right_child_centers,
            self.device,
            max_id + 2,
            self,
            max_split_id + 1,
            right_child_weight,
        )
        self.from_leaf_to_inner()

        if optimizer is not None:
            optimizer.add_param_group({"params": self.left_child.center})
            optimizer.add_param_group({"params": self.right_child.center})


class Cluster_Tree:
    """
    This class represents a binary cluster tree. It provides multiple
    functionalities used for improving the cluster tree, like calculating
    the DC and NC losses for the tree and assigning samples of a minibatch
    to the appropriate nodes. Furthermore, it provides methods for
    growing and pruning the tree as well as the analytical adaptation
    of the inner nodes.
    """

    def __init__(
        self,
        init_leafnode_centers: np.ndarray,
        device: torch.device,
    ) -> "Cluster_Tree":
        """
        Constructor for the Cluster_Tree class.

        Parameters
        ----------
        init_leafnode_centers : np.ndarray
            The centers of the two initial leaf nodes of the tree
            given as an array of shape (2, #embedd_features).
        device : torch.device
            The device to be trained on.
        """
        # center of root can be a dummy-center since it's never needed
        self.root = Cluster_Node(np.zeros(init_leafnode_centers.shape[1]), device)
        # assign the 2 initial leaf nodes with its initial centers
        self.root.set_childs(
            None,
            init_leafnode_centers[0],
            1,
            init_leafnode_centers[1],
            1,
        )

    @property
    def number_nodes(self):
        """
        Calculates the total number of nodes in the tree.

        Returns
        -------
        int
            The total number of nodes.
        """

        def count_recursive(node: Cluster_Node):
            if node.is_leaf_node():
                return 1
            return (
                1 + count_recursive(node.left_child) + count_recursive(node.right_child)
            )

        return count_recursive(self.root)

    @property
    def nodes(self) -> List[Cluster_Node]:
        """
        Gets the list of all nodes in the tree.

        Returns
        -------
        List[Cluster_Node]
            The list of all nodes.
        """

        def get_nodes_recursive(node: Cluster_Node):
            result = [node]
            if node.is_leaf_node():
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    @property
    def leaf_nodes(self) -> List[Cluster_Node]:
        """
        Gets the list of all leaf nodes in the tree.

        Returns
        -------
        List[Cluster_Node]
            The list of all leaf nodes.
        """

        def get_nodes_recursive(node: Cluster_Node):
            result = []
            if node.is_leaf_node():
                result.append(node)
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    def clear_node_assignments(self):
        """
        Clears the assignments for all nodes in the tree.
        """
        self.root.clear_assignments()

    def assign_to_nodes(
        self, minibatch_embedded: torch.Tensor, compute_sum_dist: bool = False
    ):
        """
        Assigns all samples in the minibatch to their nearest nodes in the cluster tree.
        It is performed bottom-up, so each sample is first assigned to its nearest
        leaf node. Afterwards, the samples are assigned recursively to
        the inner nodes by merging the assignments of the child node.

        Parameters
        ----------
        minibatch_embedded : torch.Tensor
            The minibatch with shape (#samples, #emb_features).
        compute_sum_dist : bool, optional
            Whether to compute the sum of squared distances, by default False.
        """
        # transform it into a list of leaf node centers and stack it into a tensor of shape (#leafnodes, #emb_features)
        leafnode_centers = list(map(lambda node: node.center.data, self.leaf_nodes))
        leafnode_tensor = torch.stack(leafnode_centers, dim=0)  # (k, d)

        #  calculate the distance from each sample in the minibatch to all leaf nodes
        with torch.no_grad():
            distance_matrix = torch.cdist(
                minibatch_embedded, leafnode_tensor, p=2
            )  # kmeans uses L_2 norm (euclidean distance) (b, k)
        # the sample gets the nearest node assigned
        min_dists, assignments = torch.min(distance_matrix, dim=1)

        # for each leafnode, check which samples it has got assigned and store the assigned samples in the leafnode
        for i, node in enumerate(self.leaf_nodes):
            indices = (assignments == i).nonzero()
            if len(indices) < 1:
                node.assignments = (
                    None  # store None (perhaps overwrite previous assignment)
                )
                node.assignment_indices = None
                node.sum_squared_dist = None
            else:
                leafnode_data = minibatch_embedded[indices.squeeze()]
                if leafnode_data.ndim == 1:
                    leafnode_data = leafnode_data[None]
                node.assignments = leafnode_data
                node.assignment_indices = indices.reshape(
                    indices.nelement()
                )  # one dimensional tensor containing the indices
                if compute_sum_dist:
                    node.sum_squared_dist = torch.sum(
                        min_dists[indices.squeeze()].pow(2)
                    )
                    
        self._assign_to_splitnodes(self.root)

    def _assign_to_splitnodes(self, node: Cluster_Node):
        """
        Recursively assigns samples to inner nodes by merging the assignments of its two children.

        Parameters
        ----------
        node : Cluster_Node
            The node where the assignments should be stored.
        """
        if node.is_leaf_node():
            return node.assignment_indices, node.assignments
        else:
            # get assignments of left child
            left_assignment_indices, left_assignments = self._assign_to_splitnodes(
                node.left_child
            )
            # get assignments of right child
            right_assignment_indices, right_assignments = self._assign_to_splitnodes(
                node.right_child
            )
            # if one of the assignments is empty, then just use the assignments of the other node
            if left_assignments is None or right_assignments is None:
                node.assignments = (
                    left_assignments if right_assignments is None else right_assignments
                )
                node.assignment_indices = (
                    left_assignment_indices
                    if right_assignments is None
                    else right_assignment_indices
                )
            else:
                # merge the assignments of the child nodes and store it in the nodes
                node.assignments = torch.cat(
                    (left_assignments, right_assignments), dim=0
                )
                node.assignment_indices = torch.cat(
                    (left_assignment_indices, right_assignment_indices), dim=0
                )
            return node.assignment_indices, node.assignments

    def nc_loss(self, augmented_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the nc loss used for adapting the leaf node centers.

        Parameters
        ----------
        augmented_batch : torch.Tensor, optional
            The augmented batch for calculating the loss, by default None.

        Returns
        -------
        torch.Tensor
            The nc loss.
        """
        leafnodes = [node for node in self.leaf_nodes if node.assignments is not None]
        if len(leafnodes) == 0:
            return torch.tensor(
                0.0, dtype=torch.float, device=self.leaf_nodes[0].device
            )

        # reformat list of tensors to one single tensor of shape (#leafnodes, #emb_features)
        leafnode_center_tensor = torch.stack([node.center for node in leafnodes], dim=0)

        # calculate the center of the assignments from the current minibatch for each leaf node
        with torch.no_grad():  # embedded space should not be optimized in this loss
            leafnode_assignments = [
                (node.assignments, node.assignment_indices) for node in leafnodes
            ]

            def calc_assignment_center(assignment):
                assignments, indices = assignment
                sum_assignments = torch.sum(assignments, dim=0)
                if augmented_batch is not None:
                    sum_assignments_aug = torch.sum(augmented_batch[indices], dim=0)
                    sum_assignments = torch.add(sum_assignments, sum_assignments_aug)
                    return sum_assignments / (2 * len(assignments))
                else:
                    return sum_assignments / len(assignments)

            leafnode_minibatch_centers = list(
                map(calc_assignment_center, leafnode_assignments)
            )

        leafnode_minibatch_centers_tensor = torch.stack(
            leafnode_minibatch_centers, dim=0
        )

        # calculate the distance between the current leaf node centers and the center of its assigned embeddings averaged over all leaf nodes
        normed_dist = torch.linalg.vector_norm(
            leafnode_center_tensor - leafnode_minibatch_centers_tensor, dim=1
        )
        loss = torch.sum(normed_dist) / len(self.leaf_nodes)
        return loss

    def dc_loss(
        self, batch_size: int, encoded_augmented_batch: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculates the overall dc loss used for improving the embedded space for a better clustering result.

        Parameters
        ----------
        batchsize : int
            The batch size used for normalization here.
        augmented_batch : torch.Tensor, optional
            The augmented batch for calculating the loss, by default None.

        Returns
        -------
        torch.Tensor
            The dc loss.
        """
        sibling_losses = []  # storing losses for each node in the tree
        sibling_aug_losses = []
        self._calculate_sibling_loss(
            self.root,
            sibling_losses,
            sibling_aug_losses,
            encoded_augmented_batch,
        )

        # transform list of losses for each node to a tensor
        # calculate overall dc loss
        if encoded_augmented_batch is None:
            loss = (
                torch.stack(sibling_losses, dim=0)
                .sum()
                .div(len(sibling_losses) * batch_size)
            )
        else:
            loss = (
                torch.stack(sibling_losses, dim=0)
                .add(torch.stack(sibling_aug_losses, dim=0))
                .sum()
                .div(len(sibling_losses) * batch_size)
            )
        return loss

    def _calculate_sibling_loss(
        self,
        root: Cluster_Node,
        sibling_loss: List[torch.Tensor],
        sibling_aug_loss: List[torch.Tensor],
        encoded_augmented_batch: torch.Tensor,
    ):
        """
        Recursively calculates the dc loss for each node. The losses are stored in the given list <sibling_loss>.

        Parameters
        ----------
        root : Cluster_Node
            The node for which children the dc loss should be calculated.
        sibling_loss : List[torch.Tensor]
            Stores the loss for each node.
        augmented_batch : torch.Tensor, optional
            The augmented batch for calculating the loss, by default None.
        """
        if root is None:
            return

        # Traverse the left subtree
        self._calculate_sibling_loss(
            root.left_child,
            sibling_loss,
            sibling_aug_loss,
            encoded_augmented_batch,
        )

        # Traverse the right subtree
        self._calculate_sibling_loss(
            root.right_child,
            sibling_loss,
            sibling_aug_loss,
            encoded_augmented_batch,
        )

        # Calculate dc loss for siblings if they exist
        if root.left_child and root.right_child:
            loss_left, loss_aug_left = self._single_sibling_loss(
                root.left_child, root.right_child, encoded_augmented_batch
            )
            loss_right, loss_aug_right = self._single_sibling_loss(
                root.right_child, root.left_child, encoded_augmented_batch
            )
            sibling_loss.extend([loss_left + loss_right])
            if encoded_augmented_batch is not None:
                sibling_aug_loss.extend([loss_aug_left + loss_aug_right])

    def _single_sibling_loss(
        self,
        node: Cluster_Node,
        sibling_node: Cluster_Node,
        encoded_augmented_batch: Union[torch.Tensor | None],
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Calculates a single dc loss for the node <node> with respect to its sibling <sibling>.

        Parameters
        ----------
        node : Cluster_Node
            The node for which the dc loss should be calculated.
        sibling : Cluster_Node
            The sibling of the node for which the dc loss should be calculated.
        augmented_batch : torch.Tensor, optional
            The augmented batch for calculating the loss, by default None.

        Returns
        -------
        torch.Tensor
            The dc loss for <node>.
        """
        if node.assignments is None:
            if encoded_augmented_batch is None:
                return torch.tensor(0.0, dtype=torch.float, device=node.device), None
            else:
                return torch.tensor(
                    0.0, dtype=torch.float, device=node.device
                ), torch.tensor(0.0, dtype=torch.float, device=node.device)
        with torch.no_grad():
            # calculate projection direction
            diff = node.center - sibling_node.center
            projection_dir = (diff / torch.linalg.vector_norm(diff)).unsqueeze(1)
        # project each sample assigned to <node> in the direction of its sibling and sum up the absolute projection values for each sample
        fixed_center_tensor = node.center.detach().unsqueeze(0).data
        projected_diff = (fixed_center_tensor - node.assignments).matmul(projection_dir)
        absolute_projections = torch.abs(projected_diff).sum(1)

        if encoded_augmented_batch is not None:
            projected_augmented_diff = (
                fixed_center_tensor - encoded_augmented_batch[node.assignment_indices]
            ).matmul(projection_dir)
            absolute_projections_aug = torch.abs(projected_augmented_diff).sum(1)
            return torch.sum(absolute_projections), torch.sum(absolute_projections_aug)
        return torch.sum(absolute_projections), None

    def adapt_inner_nodes(self, root: Cluster_Node):
        """
        Recursively assigns samples to inner nodes by merging the assignments of its two children.

        Parameters
        ----------
        root : Cluster_Node
            The node where the assignments should be stored.
        """
        if root is None:
            return

        # Traverse the left subtree
        self.adapt_inner_nodes(root.left_child)

        # Traverse the right subtree
        self.adapt_inner_nodes(root.right_child)

        # adapt node based on its 2 children
        if root.left_child and root.right_child:
            left_child_len_assignments = len(
                root.left_child.assignments
                if root.left_child.assignments is not None
                else []
            )
            root.left_child.weight = 0.5 * (
                root.left_child.weight + left_child_len_assignments
            )
            right_child_len_assignments = len(
                root.right_child.assignments
                if root.right_child.assignments is not None
                else []
            )
            root.right_child.weight = 0.5 * (
                root.right_child.weight + right_child_len_assignments
            )
            # adapt center of parent based on the new weights
            with torch.no_grad():
                child_centers = torch.stack(
                    (
                        root.left_child.weight * root.left_child.center,
                        root.right_child.weight * root.right_child.center,
                    ),
                    dim=0,
                )
                root.center = torch.sum(child_centers, axis=0) / torch.add(
                    root.left_child.weight, root.right_child.weight
                )
            root.assignments = torch.zeros(
                left_child_len_assignments + right_child_len_assignments,
                dtype=torch.int8,
                device=root.device,
            )

    def prune_tree(self, pruning_threshold: float):
        """
        Prunes the tree by removing nodes with weights below the pruning threshold.

        Parameters
        ----------
        pruning_threshold : float
            The threshold value for pruning. Nodes with weights below this threshold will be removed.

        Returns
        -------
        bool
            Returns True if pruning occurred, otherwise False.
        """

        def prune_node(parent: Cluster_Node, child_attr: str):
            """
            Prunes a node from the tree by replacing it with its child or sibling node.

            Parameters
            ----------
            parent : Cluster_Node
                The parent node from which the child or sibling node will be pruned.
            child_attr : str
                The attribute name of the child node to be pruned.

            Returns
            -------
            None
            """
            child_node: Cluster_Node = getattr(parent, child_attr)
            sibling_attr = (
                "left_child" if child_attr == "right_child" else "right_child"
            )
            sibling_node: Cluster_Node = getattr(parent, sibling_attr)

            if sibling_node is None:
                raise ValueError(sibling_node)
            else:
                if parent == self.root:
                    self.root = sibling_node
                    self.root.parent = None
                else:
                    grandparent = parent.parent
                    if grandparent.left_child == parent:
                        grandparent.left_child = sibling_node
                    else:
                        grandparent.right_child = sibling_node
                    sibling_node.parent = grandparent
                sibling_node.split_id = parent.split_id
                sibling_node.weight = parent.weight
                child_node.prune()
                del child_node
                del parent
                for leaf in self.leaf_nodes:
                    leaf.center.requires_grad = True
                print(
                    f"Tree size after pruning: {self.number_nodes}, leaf nodes: {len(self.leaf_nodes)}"
                )

        def prune_recursive(node: Cluster_Node) -> bool:
            """
            Recursively prunes the tree starting from the given node.

            Parameters
            ----------
            node : Cluster_Node
                The node from which to start pruning.

            Returns
            -------
            bool
                Returns True if pruning occurred, otherwise False.
            """
            result = False
            if node.left_child:
                result = prune_recursive(node.left_child)
            if node.right_child:
                result = prune_recursive(node.right_child)

            if node.weight < pruning_threshold:
                if node.parent is not None:
                    if node.parent.left_child == node:
                        prune_node(node.parent, "left_child")
                        result = True
                    else:
                        prune_node(node.parent, "right_child")
                        result = True
                else:
                    if (
                        self.root.left_child
                        and self.root.left_child.weight < pruning_threshold
                    ):
                        prune_node(self.root, "left_child")
                        result = True
                    elif (
                        self.root.right_child
                        and self.root.right_child.weight < pruning_threshold
                    ):
                        prune_node(self.root, "right_child")
                        result = True
            return result

        return prune_recursive(self.root)

    def grow_tree(
        self,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Union[torch.device, str],
    ) -> None:
        """
        Grows the tree at the leaf node with the highest squared distance
        between its assignments and center. The distance is not normalized,
        so larger clusters will be chosen.

        We transform the dataset (or a representative sub-sample of it)
        onto the embedded space. Then, we determine the leaf node
        with the highest sum of squared distances between its center
        and the assigned data points. We selected this rule because it
        provides a good balance between the number of data points
        and data variance for this cluster.
        Next, we split the selected node and attach two new leaf
        nodes to it as children. We determine the initial centers for
        these new leaf nodes by applying two-means (k-means with
        k = 2) to the assigned data points.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader for the dataset.
        autoencoder : torch.nn.Module
            The autoencoder model for embedding the data.
        optimizer : torch.optim.Optimizer
            The optimizer for the autoencoder.
        device : Union[torch.device, str]
            The device to perform calculations on.
        """
        with torch.no_grad():
            if len(self.nodes) > 1:
                leaf_node_dist_sums = torch.zeros(
                    len(self.leaf_nodes), dtype=torch.float, device="cpu"
                )
                for batch in dataloader:
                    x = batch[1]
                    embed = autoencoder.encode(x.to(device))
                    self.assign_to_nodes(embed, compute_sum_dist=True)
                    leaf_node_dist_sums += torch.stack(
                        [
                            (
                                leaf.sum_squared_dist.cpu()
                                if leaf.sum_squared_dist is not None
                                else torch.tensor(0, dtype=torch.float32, device="cpu")
                            )
                            for leaf in self.leaf_nodes
                        ],
                        dim=0,
                    )

                idx = leaf_node_dist_sums.argmax()
                highest_dist_leaf_node = self.leaf_nodes[idx]
                # get all assignments for highest dist leaf node
                assignments = []
                for batch in dataloader:
                    x = batch[1]
                    embed = autoencoder.encode(x.to(device))
                    self.assign_to_nodes(embed)
                    if highest_dist_leaf_node.assignments is not None:
                        assignments.append(highest_dist_leaf_node.assignments.cpu())
                child_assignments = KMeans(n_clusters=2, n_init=20).fit(
                    torch.cat(assignments, dim=0).numpy()
                )
            else:
                assignments = []
                highest_dist_leaf_node = self.root
                for batch in dataloader:
                    x = batch[1]
                    embed = autoencoder.encode(x.to(device)).cpu()
                    assignments.append(autoencoder.encode(x.to(device)).cpu())
                child_assignments = KMeans(n_clusters=2, n_init=20).fit(
                    torch.cat(assignments, dim=0).numpy()
                )

            print(f"Leaf assignments: {len(child_assignments.labels_)}")
            child_weight = 10 if len(self.nodes) < 3 else 1
            highest_dist_leaf_node.set_childs(
                optimizer,
                child_assignments.cluster_centers_[0],
                child_weight,
                child_assignments.cluster_centers_[1],
                child_weight,
                max([leaf.id for leaf in self.leaf_nodes]),
                max([node.split_id for node in self.nodes]),
            )
            print(
                f"Tree size after growing: {self.number_nodes}, leaf nodes: {len(self.leaf_nodes)}"
            )


def transform_cluster_tree_to_pred_tree(tree: Cluster_Tree) -> PredictionClusterTree:
    """
    Transforms a Cluster_Tree to a PredictionClusterTree.

    Parameters
    ----------
    tree : Cluster_Tree
        The cluster tree to transform.

    Returns
    -------
    PredictionClusterTree
        The transformed prediction cluster tree.
    """

    def transform_nodes(node: Cluster_Node):
        pred_node = PredictionClusterNode(
            node.id, node.split_id, node.center.detach().cpu().numpy()
        )
        if node.is_leaf_node():
            return pred_node
        pred_node.left_child = transform_nodes(node.left_child)
        pred_node.left_child.parent = pred_node
        pred_node.right_child = transform_nodes(node.right_child)
        pred_node.right_child.parent = pred_node
        return pred_node

    return PredictionClusterTree(transform_nodes(tree.root))

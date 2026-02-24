"""
@authors:
Collin Leiber
"""

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from clustpy.utils import dip_test
import torch
import numpy as np
from clustpy.partition.skinnydip import _dip_mirrored_data
from clustpy.deep._utils import detect_device, encode_batchwise, mean_squared_error
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep.neural_networks._resnet_ae_modules import EncoderBlock, DecoderBlock
import matplotlib.pyplot as plt
from clustpy.utils import plot_scatter_matrix
import tqdm
from collections.abc import Callable

"""
Dip module - holds backward functions
"""


class _Dip_Module(torch.nn.Module):
    """
    The _Dip_Module class is a wrapper for the _Dip_Gradient class.
    It saves the projection axes needed to calculate the Dip-values.

    Parameters
    ----------
    projection_axes : np.ndarray
        The initial projection axes. Should be of shape (k * (k-1) / 2 x dimensionality of embedding)

    Attributes
    ----------
    projection_axes : torch.Tensor
        The current projection axes
    """

    def __init__(self, projection_axes: np.ndarray):
        super(_Dip_Module, self).__init__()
        self.projection_axes = torch.nn.Parameter(torch.from_numpy(projection_axes).float())

    def forward(self, X: torch.Tensor, projection_axis_index: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate and return the Dip-value of the input data projected onto the projection axes at the specified index.
        The actual calculations will happen within the _Dip_Gradient class.

        Parameters
        ----------
        X : torch.Tensor
            The data set
        projection_axis_index : int
            The index of the projection axis within the DipModule

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor)
            The Dip-value, the modal inveral ids, the modal triangle ids
        """
        dip_value, modal_interval, modal_triangle = _Dip_Gradient.apply(X, self.projection_axes[projection_axis_index])
        return dip_value, modal_interval, modal_triangle


class _Dip_Gradient(torch.autograd.Function):
    """
    The _Dip_Gradient class is the essential class for the calculation of the Dip-test.
    This calculation will be executed in the forward function.
    The backward function calculates the gradients of the Dip-value.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function._ContextMethodMixin, X: torch.Tensor,
                projection_vector: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Execute the forward method which will return the Dip-value of the input data set projected onto the specified projection axis.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            A context object used to stash information for the backward method.
        X : torch.Tensor
            The data set
        projection_vector : torch.Tensor
            The projection axis

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor)
            The Dip-value, the modal inveral ids, the modal triangle ids
        """
        # Project data onto projection vector
        X_proj = torch.matmul(X, projection_vector)
        # Sort data
        sorted_indices = X_proj.argsort()
        # Calculate dip
        sorted_data = X_proj[sorted_indices]
        sorted_data_numpy = sorted_data.detach().cpu().numpy()
        dip_value, modal_interval, modal_triangle = dip_test(sorted_data_numpy, is_data_sorted=True, just_dip=False)
        dip_value_torch = torch.tensor(dip_value)
        modal_interval_torch = torch.tensor(modal_interval, dtype=torch.long)
        modal_triangle_torch = torch.tensor(modal_triangle, dtype=torch.long) 
        # Save parameters for backward
        ctx.save_for_backward(X, X_proj, sorted_indices, projection_vector, modal_triangle_torch)
        return dip_value_torch, sorted_indices[modal_interval_torch], sorted_indices[modal_triangle_torch]

    @staticmethod
    def backward(ctx: torch.autograd.function._ContextMethodMixin, grad_output_dip: torch.Tensor, grad_output_modal_interval: torch.Tensor, 
                 grad_output_modal_triangle: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Execute the backward method which will return the gradients of the Dip-value calculated in the forward method.
        First gradient corresponds the data, second gradient corresponds to the projection axis.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            A context object used to load information from the forward method.
        grad_output : torch.Tensor
            Corresponds to the factor that the Dip-value has been multiplied by after it has been returned be the _Dip_Module

        Returns
        -------
        gradient : (torch.Tensor, torch.Tensor)
            The gradient of the Dip-value with respect to the data and with respect to the projection axis
        """
        # Load parameters from forward
        X, X_proj, sorted_indices, projection_vector, modal_triangle = ctx.saved_tensors
        device = detect_device(projection_vector.get_device())
        if -1 in modal_triangle:
            return torch.zeros((X_proj.shape[0], projection_vector.shape[0])).to(device), torch.zeros(
                projection_vector.shape).to(device)
        # Calculate the partial derivative for all dimensions
        data_index_i1, data_index_i2, data_index_i3 = sorted_indices[modal_triangle]
        # Get A and c
        A = modal_triangle[0] - modal_triangle[1] + \
            (modal_triangle[2] - modal_triangle[0]) * (X_proj[data_index_i2] - X_proj[data_index_i1]) / (
                    X_proj[data_index_i3] - X_proj[data_index_i1])
        constant = torch.true_divide(modal_triangle[2] - modal_triangle[0], 2 * X.shape[0])
        # Check A
        if A < 0:
            constant = -constant
        # Calculate derivative of projection vector
        gradient_proj = _calculate_partial_derivative_proj(X, X_proj, data_index_i1, data_index_i2, data_index_i3)
        gradient_proj = gradient_proj * constant
        # Calculate derivative for projected datapoints
        gradient_x_tmp = _calculate_partial_derivative_x(X_proj, data_index_i1, data_index_i2, data_index_i3, device)
        gradient_x_tmp = gradient_x_tmp * constant
        # Mind the matrix multiplication of the data and the projection
        tmp_vec = torch.ones(X.shape).to(device) * projection_vector
        gradient_x = tmp_vec * gradient_x_tmp.reshape(-1, 1)
        # Return gradients
        return grad_output_dip * gradient_x, grad_output_dip * gradient_proj


def _calculate_partial_derivative_x(X_proj, data_index_i1: torch.long, data_index_i2: torch.long,
                                    data_index_i3: torch.long, device: torch.device) -> torch.Tensor:
    """
    Calculate the gradient of the Dip-value with respect to the data.

    Parameters
    ----------
    X_proj : torch.Tensor
        The projected data
    data_index_i1 : torch.long
        Index of the first full-dimensional object of the modal triangle (beware that the index of the projected and non-projected data differs)
    data_index_i2 : torch.long
        Index of the second full-dimensional object of the modal triangle (beware that the index of the projected and non-projected data differs)
    data_index_i3 : torch.long
        Index of the third full-dimensional object of the modal triangle (beware that the index of the projected and non-projected data differs)
    device : torch.device
        device to be trained on

    Returns
    -------
    gradient : torch.Tensor
        The gradient of the Dip-value with respect to the data
    """
    gradient = torch.zeros(X_proj.shape[0]).to(device)
    # derivative X[jb] = i1
    d_X_jb = (X_proj[data_index_i2] - X_proj[data_index_i3]) / (X_proj[data_index_i3] - X_proj[data_index_i1]) ** 2
    gradient[data_index_i1] = d_X_jb
    # derivative X[jj] = i2
    d_X_jj = 1 / (X_proj[data_index_i3] - X_proj[data_index_i1])
    gradient[data_index_i2] = d_X_jj
    # derivative X[je] = i3
    d_X_je = (X_proj[data_index_i1] - X_proj[data_index_i2]) / (X_proj[data_index_i3] - X_proj[data_index_i1]) ** 2
    gradient[data_index_i3] = d_X_je
    return gradient


def _calculate_partial_derivative_proj(X: torch.Tensor, X_proj: torch.Tensor, data_index_i1: torch.long,
                                       data_index_i2: torch.long, data_index_i3: torch.long) -> torch.Tensor:
    """
    Calculate the gradient of the Dip-value with respect to the projection axis.

    Parameters
    ----------
    X : torch.Tensor
        The data set
    X_proj : torch.Tensor
        The projected data
    data_index_i1 : torch.long
        Index of the first full-dimensional object of the modal triangle (beware that the index of the projected and non-projected data differs)
    data_index_i2 : torch.long
        Index of the second full-dimensional object of the modal triangle (beware that the index of the projected and non-projected data differs)
    data_index_i3 : torch.long
        Index of the third full-dimensional object of the modal triangle (beware that the index of the projected and non-projected data differs)

    Returns
    -------
    gradient : torch.Tensor
        The gradient of the Dip-value with respect to the projection axis
    """
    quotient = (X_proj[data_index_i3] - X_proj[data_index_i1])
    gradient = (X[data_index_i2] - X[data_index_i1]) / quotient - \
               (X[data_index_i3] - X[data_index_i1]) * (
                       X_proj[data_index_i2] - X_proj[data_index_i1]) / quotient ** 2
    return gradient


"""
Module-helpers
"""


def plot_dipencoder_embedding(X_embed: np.ndarray, n_clusters: int, labels: np.ndarray, projection_axes: np.ndarray,
                              index_dict: dict, edge_width: float = 0.1, show_legend: bool = False,
                              show_plot: bool = True) -> None:
    """
    Plot the current state of the DipEncoder.
    Uses the plot_scatter_matrix as a basis and adds projection axes in red.

    Parameters
    ----------
    X_embed : np.ndarray
        The embedded data set
    n_clusters : int
        Number of clusters
    labels : np.ndarray
        The cluster labels
    projection_axes : np.ndarray
        The projection axes between the clusters
    index_dict : dict
        A dictionary to match the indices of two clusters to a projection axis
    edge_width : float
        Specifies the width of the empty space (containung no points) at the edges of the plots
    show_legend : bool
        Specifies whether a legend should be added to the plot
    show_plot : bool
        Specifies whether the plot should be plotted, i.e. if plt.show() should be executed (default: True)
    """
    # Get cluster means do plot projection axes
    means = [np.mean(X_embed[labels == i], axis=0) for i in range(n_clusters)]
    # Get min and max values to scale the plots
    mins = np.array([np.min(X_embed[:, i]) for i in range(X_embed.shape[1])])
    maxs = np.array([np.max(X_embed[:, i]) for i in range(X_embed.shape[1])])
    max_min_diffs = maxs - mins
    arbitrary_high_value = 999999  # used to have infinite projection axis
    # Plot the scatter matrix
    axes = plot_scatter_matrix(X_embed, labels=labels, show_plot=False, show_legend=show_legend)
    # Add projection axes
    for m in range(X_embed.shape[1]):
        for n in range(X_embed.shape[1]):
            if m == n:
                continue
            ax = axes[m, n]
            for a in range(n_clusters - 1):
                for b in range(a + 1, n_clusters):
                    projection_axis = projection_axes[index_dict[(a, b)]]
                    ax.plot([(means[a] - arbitrary_high_value * projection_axis)[n],
                             (means[a] + arbitrary_high_value * projection_axis)[n]],
                            [(means[a] - arbitrary_high_value * projection_axis)[m],
                             (means[a] + arbitrary_high_value * projection_axis)[m]],
                            c="r", ls="--")
            # Set the limits -> will create the empty space (no points) at the edge of the plot
            ax.set_ylim([mins[m] - edge_width * max_min_diffs[m], maxs[m] + edge_width * max_min_diffs[m]])
            ax.set_xlim([mins[n] - edge_width * max_min_diffs[n], maxs[n] + edge_width * max_min_diffs[n]])
    if show_plot:
        plt.show()


def _get_ssl_loss_of_first_batch(trainloader: torch.utils.data.DataLoader, neural_network: torch.nn.Module,
                                 ssl_loss_fn: Callable | torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
    """
    Calculate the ssl loss of the first batch of data.
    Therefore, a new instance of the neural network will be created using the same architecture.

    Parameters
    ----------
    trainloader : torch.utils.data.DataLoader
        dataloader to be used for training
    neural network : torch.nn.Module
        the neural_network
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    device : torch.device
        device to be trained on

    Returns
    -------
    ssl_loss : torch.Tensor
        The ssl loss of the first batch of data
    """
    neural_network_class = type(neural_network)
    # Create new instance of the neural network
    if hasattr(neural_network, "encoder"):
        # In case of Feedforward-based architectures
        tmp_neural_network = neural_network_class(layers=neural_network.encoder.layers,
                                                  decoder_layers=neural_network.decoder.layers).to(device)
    else:
        # In case of Conv-based architectures
        conv_encoder_name = "resnet18" if type(neural_network.conv_encoder.layer1[0]) is EncoderBlock else "resnet50"
        conv_decoder_name = "resnet18" if type(neural_network.conv_decoder.layer1[0]) is DecoderBlock else "resnet50"
        tmp_neural_network = neural_network_class(input_height=neural_network.input_height,
                                                  fc_layers=neural_network.fc_encoder.layers,
                                                  conv_encoder_name=conv_encoder_name,
                                                  fc_decoder_layers=neural_network.fc_decoder.layers,
                                                  conv_decoder_name=conv_decoder_name).to(device)
    # Get first batch of data and calculate ssl loss
    batch_init = next(iter(trainloader))
    ssl_loss, _, _ = tmp_neural_network.loss(batch_init, ssl_loss_fn, device)
    return ssl_loss.detach()


def _predict_using_thresholds(X: np.ndarray, projections: np.ndarray, projection_thresholds: list,
            n_clusters: int, index_dict: dict) -> np.ndarray:
    """
    Predict the clustering labels using the embedding of the neural network and the saved threshold on the projection axes.
    If an object of X is left of this threshold it matches the left cluster. The same applies analogously to the right cluster.
    In the end the object will be assigned the label of the cluster that matched most often.

    Parameters
    ----------
    X : np.ndarray
        The data set used to predict the labels
    projections : np.ndarray
        Matrix containing all the projection axes
    projection_thresholds : list
        List containing the thresholds for each projection axis and a tuple indicating which cluster is left and right of the threshold
    n_clusters : int
        The total number of clusters
    index_dict : dict
        A dictionary to match the indices of two clusters to a projection axis

    Returns
    -------
    labels_pred : np.ndarray
        The predicted labels
    """
    labels_pred_matrix = np.zeros((X.shape[0], n_clusters))
    for m in range(n_clusters - 1):
        for n in range(m + 1, n_clusters):
            # Get correct projection vector
            projection_vector = projections[index_dict[(m, n)]]
            threshold, cluster_tuple = projection_thresholds[index_dict[(m, n)]]
            assert m in cluster_tuple and n in cluster_tuple
            # Project data
            X_proj = np.matmul(X, projection_vector)
            labels_pred_matrix[X_proj < threshold, cluster_tuple[0]] += 1
            labels_pred_matrix[X_proj >= threshold, cluster_tuple[1]] += 1
    # Get best matching cluster
    labels_pred = np.argmax(labels_pred_matrix, axis=1)
    return labels_pred


class _DipEncoder_Module(torch.nn.Module):
    """
    The _DipEncoder_Module. Contains most of the algorithm specific procedures like the loss function.

    Parameters
    ----------
    n_clusters : int
        nNumber of clusters
    index_dict : dict
        A dictionary to match the indices of two clusters to a projection axis
    dip_module : torch.nn.Module
        The DipModule
    init_np_labelss : np.ndarray
        The initial cluster labels
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) samples will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    use_gt : bool
        Use the ground truth to learn the embedding. In that case the labels will not change during optimization (default: False)

    Attributes
    ----------
    labels : float
        the cluster labels
    projection_thresholds_ : bool
        A list containing the thresholds for each projection axis and a tuple indicating which cluster is left and right of the threshold
    """

    def __init__(self, n_clusters: int, index_dict: dict, dip_module: torch.nn.Module, 
                 init_np_labels: np.ndarray, max_cluster_size_diff_factor: float, augmentation_invariance: bool = False, 
                 use_gt: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self.index_dict = index_dict
        self.dip_module = dip_module
        self.labels = init_np_labels
        self.max_cluster_size_diff_factor = max_cluster_size_diff_factor
        self.augmentation_invariance = augmentation_invariance
        self.use_gt = use_gt

    def _update_labels_and_thresholds(self, X: np.ndarray) -> (np.ndarray, list):
        """
        Predict the clustering labels using the current structure of the neural network and DipModule.
        Therefore, we determine the modal interval for two clusters on their corresponding projection axis using X.
        The center between the upper bound of the left cluster and the lower bound of the right cluster will be used as a threshold.
        If an object of X is left of this threshold it matches the left cluster. The same applies analogously to the right cluster.
        In the end the object will be assigned the label of the cluster that matched most often.

        Parameters
        ----------
        X : np.ndarray
            The data set used to retrieve the modal intervals

        Returns
        -------
        tuple : (np.ndarray, list)
            The new labels,
            A list containing the thresholds for each projection axis and a tuple indicating which cluster is left and right of the threshold
        """
        projections = self.dip_module.projection_axes.detach().cpu().numpy()
        points_in_all_clusters = [np.where(self.labels == clus)[0] for clus in range(self.n_clusters)]
        n_points_in_all_clusters = [points_in_cluster.shape[0] for points_in_cluster in points_in_all_clusters]
        labels_pred_matrix = np.zeros((X.shape[0], self.n_clusters))
        projection_thresholds = []
        for m in range(self.n_clusters - 1):
            for n in range(m + 1, self.n_clusters):
                # Get correct projection vector
                projection_vector = projections[self.index_dict[(m, n)]]
                # Project data
                X_proj = np.matmul(X, projection_vector)
                X_proj_m = X_proj[points_in_all_clusters[m]]
                X_proj_n = X_proj[points_in_all_clusters[n]]
                if n_points_in_all_clusters[m] < 4:
                    low_m_coor = np.mean(X_proj_m)
                    high_m_coor = low_m_coor
                else:
                    # Sort data
                    sorted_indices_m = X_proj_m.argsort()
                    # Execute mirrored dip
                    _, low_m, high_m = _dip_mirrored_data(X_proj_m[sorted_indices_m], None)
                    low_m_coor = X_proj_m[sorted_indices_m[low_m]]
                    high_m_coor = X_proj_m[sorted_indices_m[high_m]]
                if n_points_in_all_clusters[n] < 4:
                    low_n_coor = np.mean(X_proj_n)
                    high_n_coor = low_n_coor
                else:
                    # Sort data
                    sorted_indices_n = X_proj_n.argsort()
                    # Execute mirrored dip
                    _, low_n, high_n = _dip_mirrored_data(X_proj_n[sorted_indices_n], None)
                    low_n_coor = X_proj_n[sorted_indices_n[low_n]]
                    high_n_coor = X_proj_n[sorted_indices_n[high_n]]
                # Check if projected data is better captured by cluster m or n
                if low_m_coor > high_n_coor:  # cluster m right of cluster n
                    threshold = high_n_coor + (low_m_coor - high_n_coor) / 2
                    labels_pred_matrix[X_proj < threshold, n] += 1
                    labels_pred_matrix[X_proj >= threshold, m] += 1
                    projection_thresholds.append((threshold, (n, m)))
                elif low_n_coor > high_m_coor:  # cluster n right of cluster m
                    threshold = high_m_coor + (low_n_coor - high_m_coor) / 2
                    labels_pred_matrix[X_proj < threshold, m] += 1
                    labels_pred_matrix[X_proj >= threshold, n] += 1
                    projection_thresholds.append((threshold, (m, n)))
                else:
                    center_coor_m = (low_m_coor + high_m_coor) / 2
                    center_coor_n = (low_n_coor + high_n_coor) / 2
                    if center_coor_m > center_coor_n:  # cluster m right of cluster n
                        threshold = low_m_coor + (high_n_coor - low_m_coor) / 2
                        labels_pred_matrix[X_proj < threshold, n] += 1
                        labels_pred_matrix[X_proj >= threshold, m] += 1
                        projection_thresholds.append((threshold, (n, m)))
                    else:  # cluster n right of cluster m
                        threshold = low_n_coor + (high_m_coor - low_n_coor) / 2
                        labels_pred_matrix[X_proj < threshold, m] += 1
                        labels_pred_matrix[X_proj >= threshold, n] += 1
                        projection_thresholds.append((threshold, (m, n)))
        # Get best matching cluster
        labels_pred = np.argmax(labels_pred_matrix, axis=1)
        return labels_pred, projection_thresholds

    def _get_dip_error(self, X_embed: torch.Tensor, projection_axis_index: int,
                    points_in_m: torch.Tensor, points_in_n: torch.Tensor, n_points_in_m: int, n_points_in_n: int,
                    device: torch.device) -> torch.Tensor:
        """
        Calculate the dip error for the projection axis between cluster m and cluster n.
        In details it returns:
        0.5 * ((Dip-value of cluster m) + (Dip-value of cluster m)) - (Dip-value of cluster m and n)
        on this specific projeciton axis.

        Parameters
        ----------
        X_embed : torch.Tensor
            The embedded data set
        projection_axis_index : int
            The index of the projection axis within the DipModule 
        points_in_m : torch.Tensor
            Tensor containing the indices of the objects within cluster m
        points_in_n : torch.Tensor
            Tensor containing the indices of the objects within cluster n
        n_points_in_m : int
            Size of cluster m
        n_points_in_n : int
            Size of cluster n
        device : torch.device
            device to be trained on

        Returns
        -------
        dip_loss_new : torch.Tensor
            The final Dip loss on the specified projection axis
        """
        # Calculate dip cluster m
        dip_value_m, _, _ = self.dip_module(X_embed[points_in_m], projection_axis_index)
        dip_value_m = (dip_value_m.detach() * 4) * dip_value_m # weight by dip
        # Calculate dip cluster n
        dip_value_n, _, _ = self.dip_module(X_embed[points_in_n], projection_axis_index)
        dip_value_n = (dip_value_n.detach() * 4) * dip_value_n # weight by dip
        # Calculate dip combined clusters m and n
        if n_points_in_m > self.max_cluster_size_diff_factor * n_points_in_n:
            perm = torch.randperm(n_points_in_m).to(device)
            sampled_m = points_in_m[perm[:int(n_points_in_n * self.max_cluster_size_diff_factor)]]
            dip_value_mn, _, _ = self.dip_module(torch.cat([X_embed[sampled_m], X_embed[points_in_n]]),
                                    projection_axis_index)
        elif n_points_in_n > self.max_cluster_size_diff_factor * n_points_in_m:
            perm = torch.randperm(n_points_in_n).to(device)
            sampled_n = points_in_n[perm[:int(n_points_in_m * self.max_cluster_size_diff_factor)]]
            dip_value_mn, _, _ = self.dip_module(torch.cat([X_embed[points_in_m], X_embed[sampled_n]]),
                                    projection_axis_index)
        else:
            dip_value_mn, _, _ = self.dip_module(X_embed[torch.cat([points_in_m, points_in_n])], projection_axis_index)
        dip_value_mn = (0.25 - dip_value_mn.detach()) * 4 * dip_value_mn # weight by dip
        # We want to maximize dip between clusters => set mn loss to -dip
        dip_loss_new = 0.5 * (dip_value_m + dip_value_n) - dip_value_mn
        return dip_loss_new

    def _loss(self, batch: list, labels_torch: torch.Tensor, neural_network: torch.nn.Module, 
              ssl_loss_fn: Callable | torch.nn.modules.loss._Loss, ssl_loss_weight: float, 
              clustering_loss_weight: float, device: torch.device) -> torch.Tensor:
        """
        Calculate the complete DipEncoder + neural network loss.

        Parameters
        ----------
        batch : list
            the minibatch
        labels_torch : torch.Tensor
            the current cluster labels as torch tensor
        neural_network : torch.nn.Module
            the neural network
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss
        clustering_loss_weight : float
            weight of the clustering loss
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : torch.Tensor
            the final DipEncoder loss
        """
        MIN_NUMBER_OF_POINTS = 10
        ids = batch[0]
        # SSL Loss
        if self.augmentation_invariance:
            ssl_loss, embedded, _, embedded_aug, _ = neural_network.loss_augmentation(batch, ssl_loss_fn,
                                                                                    device)
        else:
            ssl_loss, embedded, _ = neural_network.loss(batch, ssl_loss_fn, device)
        # Get points within each cluster
        points_in_all_clusters = [torch.where(labels_torch[ids] == clus)[0].to(device) for clus in
                                range(self.n_clusters)]
        n_points_in_all_clusters = [points_in_cluster.shape[0] for points_in_cluster in points_in_all_clusters]
        if self.augmentation_invariance:
            # Regular embedded data will be combined with augmented data
            points_in_all_clusters = [torch.cat((p_c, p_c + embedded.shape[0])) for p_c in points_in_all_clusters]
            n_points_in_all_clusters = [2 * n_c for n_c in n_points_in_all_clusters]
            embedded = torch.cat((embedded, embedded_aug), 0)
        dip_loss = torch.tensor(0)
        for m in range(self.n_clusters - 1):
            if n_points_in_all_clusters[m] < MIN_NUMBER_OF_POINTS:
                continue
            for n in range(m + 1, self.n_clusters):
                if n_points_in_all_clusters[n] < MIN_NUMBER_OF_POINTS:
                    continue
                dip_loss_new = self._get_dip_error(
                    embedded, self.index_dict[(m, n)], points_in_all_clusters[m], points_in_all_clusters[n],
                    n_points_in_all_clusters[m], n_points_in_all_clusters[n], device)
                dip_loss = dip_loss + dip_loss_new
        final_dip_loss = torch.true_divide(dip_loss, len(self.index_dict))
        loss = clustering_loss_weight * final_dip_loss + ssl_loss * ssl_loss_weight
        return loss

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader, 
            testloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
            clustering_loss_weight: float, ssl_loss_weight: float) -> '_DipEncoder_Module':
        """
        Trains the _DipEncoder_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        testloader : torch.utils.data.DataLoader
            dataloader to be used for updating the clustering parameters
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss

        Returns
        -------
        self : _DipEncoder_Module
            this instance of the _DipEncoder_Module
        """
        labels_torch = torch.from_numpy(self.labels).int().to(device)
        # Start Optimization
        tbar = tqdm.trange(n_epochs + 1, desc="DipEncoder training")
        for iteration in tbar:
            # Update labels for clustering
            if not self.use_gt:
                X_embed = encode_batchwise(testloader, neural_network)
                self.labels, self.projection_thresholds_ = self._update_labels_and_thresholds(X_embed)
                labels_torch = torch.from_numpy(self.labels).int().to(device)
            if iteration == n_epochs:
                break
            total_loss = 0
            for batch in trainloader:
                loss = self._loss(batch, labels_torch, neural_network, ssl_loss_fn, ssl_loss_weight, clustering_loss_weight, device)
                total_loss += loss.item()
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            postfix_str = {"Loss": total_loss}
            tbar.set_postfix(postfix_str)
        # Get final labels
        self.labels = self.labels.astype(np.int32)
        if self.use_gt:
            X_embed = encode_batchwise(testloader, neural_network)
            _, self.projection_thresholds_ = self._update_labels_and_thresholds(X_embed)
        return self

"""
DipEncoder object
"""


class DipEncoder(_AbstractDeepClusteringAlgo):
    """
    The DipEncoder.
    Can be used either as a clustering procedure if no ground truth labels are given or as a supervised dimensionality reduction technique.
    First, a neural network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the DipEncoder loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, that can determine the number of clusters, e.g. DBSCAN (default: 8)
    batch_size : int
        size of the data batches for the actual training of the DipEncoder.
        Should be larger the more clusters we have. If it is None, it will be set to (25 x n_clusters) (default: None)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate. If None, it will be set to {"lr": 1e-3} (default: None)
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate. If None, it will be set to {"lr": 1e-4} (default: None)
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: mean_squared_error)
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used (default: 3)
    clustering_loss_weight : float
        weight of the clustering loss (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss.
        If None, it will be equal to 1/(4L), where L is the reconstruction loss of the first batch of an untrained neural network (default: None)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class. If None, it will be set to {} (default: None)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    projection_axes_ : np.ndarray
        The final projection axes between the clusters
    index_dict_ : dict
        A dictionary to match the indices of two clusters to a projection axis
    projection_thresholds_ : list
        A list containing the thresholds for each projection axis and a tuple indicating which cluster is left and right of the threshold
    neural_network_trained_ : torch.nn.Module
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

    Examples
    ----------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DipEncoder
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> dipencoder = DipEncoder(3, pretrain_epochs=3, clustering_epochs=3)
    >>> dipencoder.fit(data)

    References
    ----------
    Leiber, Collin and Bauer, Lena G. M. and Neumayr, Michael and Plant, Claudia and Böhm, Christian
    "The DipEncoder: Enforcing Multimodality in Autoencoders."
    Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2022.
    """

    def __init__(self, n_clusters: int = 8, batch_size: int = None, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 150, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, max_cluster_size_diff_factor: float = 3,
                 clustering_loss_weight: float = 1., ssl_loss_weight: float = None,
                 custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 initial_clustering_class: ClusterMixin = KMeans, initial_clustering_params: dict = None,
                 device: torch.device = None, random_state: np.random.RandomState | int = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters = n_clusters
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.max_cluster_size_diff_factor = max_cluster_size_diff_factor
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params

    def fit(self, X: np.ndarray, val_set: np.ndarray = None, y: np.ndarray = None) -> 'DipEncoder':
        """
        Initiate the actual clustering/dimensionality reduction process on the input data set.
        If no ground truth labels are given, the resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            The given (training) data set
        val_set : np.ndarray
            The validation data set (not used in DipEncoder, included for compatibility reasons) (default: None)
        y : np.ndarray
            The ground truth labels. If None, the DipEncoder will be used for clustering (default: None)

        Returns
        -------
        self : DipEncoder
            This instance of the DipEncoder
        """
        X, y, random_state, pretrain_optimizer_params, clustering_optimizer_params, initial_clustering_params = self._check_parameters(X, y=y)
        assert self.batch_size is not None or self.n_clusters is not None, "n_clusters and batch_size can not both be None"
        batch_size = 25 * self.n_clusters if self.batch_size is None else self.batch_size
        # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
        device, trainloader, testloader, _, neural_network, X_embed, n_clusters, init_labels, init_centers, _ = get_default_deep_clustering_initialization(
            X, val_set, self.n_clusters, batch_size, pretrain_optimizer_params, self.pretrain_epochs, self.optimizer_class, self.ssl_loss_fn,
            self.neural_network, self.embedding_size, self.custom_dataloaders, self.initial_clustering_class if y is None else None, 
            initial_clustering_params, self.device, random_state, neural_network_weights=self.neural_network_weights)
        if y is not None:
            class_labels = np.unique(y)
            if len(class_labels) != self.n_clusters:
                print("WARNING: If y is specified, the number of labels must match n_clusters. Therefore, n_clusters was changed from {0} to {1}".format(self.n_clusters, len(class_labels)))
                n_clusters = len(class_labels)
            # If y is give, overwrite labels and centers
            init_labels = y.astype(int)
            init_centers = np.array([np.mean(X_embed[y == i], axis=0) for i in range(n_clusters)])
        n_cluster_combinations = int((n_clusters - 1) * n_clusters / 2)
        # Get factor for AE loss
        # rand_samples = torch.rand((batch_size, X.shape[1]))
        # data_min = np.min(X)
        # data_max = np.max(X)
        # rand_samples_resized = (rand_samples * (data_max - data_min) + data_min).to(device)
        # rand_samples_reconstruction = neural_network.forward(rand_samples_resized)
        # reconstruction_loss_weight = loss_fn(rand_samples_reconstruction, rand_samples_resized).detach()
        if self.ssl_loss_weight is None:
            ssl_loss_weight = _get_ssl_loss_of_first_batch(trainloader, neural_network, self.ssl_loss_fn, device)
            ssl_loss_weight = 1 / (4 * ssl_loss_weight)
            print("INFO: Setting ssl_loss_weight automatically; set to", ssl_loss_weight)
        else:
            ssl_loss_weight = self.ssl_loss_weight
        # Create initial projections vectors by using difference between cluster centers
        index_dict = {}
        projections = np.zeros((n_cluster_combinations, self.embedding_size))
        for m in range(n_clusters - 1):
            for n in range(m + 1, n_clusters):
                mean_1 = init_centers[m]
                mean_2 = init_centers[n]
                v = mean_1 - mean_2
                projections[len(index_dict)] = v
                index_dict[(m, n)] = len(index_dict)
        # Create DipModule
        dip_module = _Dip_Module(projections).to(device)
        # Create SGD Optimizer
        optimizer = self.optimizer_class(list(neural_network.parameters()) + list(dip_module.parameters()),
                                    **clustering_optimizer_params)
        dipencoder_module = _DipEncoder_Module(n_clusters, index_dict, dip_module, init_labels, self.max_cluster_size_diff_factor, self.augmentation_invariance, y is not None)
        dipencoder_module.fit(neural_network, trainloader, testloader, self.clustering_epochs, device, optimizer, self.ssl_loss_fn, 
                              self.clustering_loss_weight, ssl_loss_weight)
        # Save values
        self.labels_ = dipencoder_module.labels
        self.projection_axes_ = dip_module.projection_axes.detach().cpu().numpy()
        self.projection_thresholds_ = dipencoder_module.projection_thresholds_
        self.index_dict_ = index_dict
        self.neural_network_trained_ = neural_network
        self.n_clusters_out_ = n_clusters
        self.set_n_featrues_in(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        X_embed = self.transform(X)
        labels_pred = _predict_using_thresholds(X_embed, self.projection_axes_, self.projection_thresholds_, self.n_clusters_out_, self.index_dict_)
        return labels_pred.astype(np.int32)

    def plot(self, X: np.ndarray, edge_width: float = 0.2, show_legend: bool = True) -> None:
        """
        Plot the current state of the DipEncoder.
        First the data set will be encoded using the neural network, afterwards the plot will be created.
        Uses the plot_scatter_matrix as a basis and adds projection axes in red.

        Parameters
        ----------
        X : np.ndarray
            The data set
        edge_width : float
            Specifies the width of the empty space (containung no points) at the edges of the plots
        show_legend : bool
            Specifies whether a legend should be added to the plot
        """
        X_embed = self.transform(X)
        plot_dipencoder_embedding(X_embed, self.n_clusters, self.labels_, self.projection_axes_, self.index_dict_,
                                  edge_width, show_legend=show_legend)

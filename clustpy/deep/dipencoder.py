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
from clustpy.deep._utils import detect_device, encode_batchwise, run_initial_clustering
from clustpy.deep._data_utils import get_dataloader, augmentation_invariance_check
from clustpy.deep._train_utils import get_trained_autoencoder
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep.autoencoders._resnet_ae_modules import EncoderBlock, DecoderBlock
import matplotlib.pyplot as plt
from clustpy.utils import plot_scatter_matrix

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

    def forward(self, X: torch.Tensor, projection_axis_index: int) -> torch.Tensor:
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
        dip_value : torch.Tensor
            The Dip-value
        """
        dip_value = _Dip_Gradient.apply(X, self.projection_axes[projection_axis_index])
        return dip_value


class _Dip_Gradient(torch.autograd.Function):
    """
    The _Dip_Gradient class is the essential class for the calculation of the Dip-test.
    This calculation will be executed in the forward function.
    The backward function calculates the gradients of the Dip-value.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function._ContextMethodMixin, X: torch.Tensor,
                projection_vector: torch.Tensor) -> torch.Tensor:
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
        torch_dip : torch.Tensor
            The Dip-value
        """
        # Project data onto projection vector
        X_proj = torch.matmul(X, projection_vector)
        # Sort data
        sorted_indices = X_proj.argsort()
        # Calculate dip
        sorted_data = X_proj[sorted_indices].detach().cpu().numpy()
        dip_value, _, modal_triangle = dip_test(sorted_data, is_data_sorted=True, just_dip=False)
        torch_dip = torch.tensor(dip_value)
        # Save parameters for backward
        ctx.save_for_backward(X, X_proj, sorted_indices, projection_vector,
                              torch.tensor(modal_triangle, dtype=torch.long), torch_dip)
        return torch_dip

    @staticmethod
    def backward(ctx: torch.autograd.function._ContextMethodMixin, grad_output: torch.Tensor) -> (
            torch.Tensor, torch.Tensor):
        """
        Execute the backward method which will return the gradients of the Dip-value calculated in the forward method.
        First gradient corresponds the the data, second gradient corresponds to the projection axis.

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
        device = detect_device()
        # Load parameters from forward
        X, X_proj, sorted_indices, projection_vector, modal_triangle, dip_value = ctx.saved_tensors
        if -1 in modal_triangle:
            return torch.zeros((X_proj.shape[0], projection_vector.shape[0])).to(device), torch.zeros(
                projection_vector.shape).to(device)
        # Grad_output equals gradient of outer operations. Update grad_output to consider dip
        if grad_output > 0:
            grad_output = grad_output * dip_value * 4
        else:
            grad_output = grad_output * (0.25 - dip_value) * 4
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
        return grad_output * gradient_x, grad_output * gradient_proj


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


def _get_dip_error(dip_module: _Dip_Module, X_embed: torch.Tensor, projection_axis_index: int,
                   points_in_m: torch.Tensor, points_in_n: torch.Tensor, n_points_in_m: int, n_points_in_n: int,
                   max_cluster_size_diff_factor: float, device: torch.device) -> torch.Tensor:
    """
    Calculate the dip error for the projection axis between cluster m and cluster n.
    In details it returns:
    0.5 * ((Dip-value of cluster m) + (Dip-value of cluster m)) - (Dip-value of cluster m and n)
    on this specific projeciton axis.

    Parameters
    ----------
    dip_module : _Dip_Module
        The DipModule
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
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used
    device : torch.device
        device to be trained on

    Returns
    -------
    dip_loss_new : torch.Tensor
        The final Dip loss on the specified projection axis
    """
    # Calculate dip cluster m
    dip_value_m = dip_module(X_embed[points_in_m], projection_axis_index)

    # Calculate dip cluster n
    dip_value_n = dip_module(X_embed[points_in_n], projection_axis_index)

    # Calculate dip combined clusters m and n
    if n_points_in_m > max_cluster_size_diff_factor * n_points_in_n:
        perm = torch.randperm(n_points_in_m).to(device)
        sampled_m = points_in_m[perm[:int(n_points_in_n * max_cluster_size_diff_factor)]]
        dip_value_mn = dip_module(torch.cat([X_embed[sampled_m], X_embed[points_in_n]]),
                                  projection_axis_index)
    elif n_points_in_n > max_cluster_size_diff_factor * n_points_in_m:
        perm = torch.randperm(n_points_in_n).to(device)
        sampled_n = points_in_n[perm[:int(n_points_in_m * max_cluster_size_diff_factor)]]
        dip_value_mn = dip_module(torch.cat([X_embed[points_in_m], X_embed[sampled_n]]),
                                  projection_axis_index)
    else:
        dip_value_mn = dip_module(X_embed[torch.cat([points_in_m, points_in_n])], projection_axis_index)
    # We want to maximize dip between clusters => set mn loss to -dip
    dip_loss_new = 0.5 * (dip_value_m + dip_value_n) - dip_value_mn
    return dip_loss_new


def _predict(X_train: np.ndarray, X_test: np.ndarray, labels_train: np.ndarray, projections: np.ndarray,
             n_clusters: int, index_dict: dict) -> np.ndarray:
    """
    Predict the clustering labels using the current structure of the autoencoder and DipModule.
    Therefore, we determine the modal interval for two clusters on their corresponding projection axis using X_train.
    The center between the upper bound of the left cluster and the lower bound of the right cluster will be used as a threshold.
    If an object of X_test is left of this threshold it will be assigned to the left cluster. The same applies analogously to the right cluster.
    In the end the object will be assigned the label of the cluster that matched most often.

    Parameters
    ----------
    X_train : np.ndarray
        The data set used to retrieve the modal intervals
    X_test : np.ndarray
        The data set for which we want to retrieve the labels
    labels_train : np.ndarray
        The labels of X_train
    projections : np.ndarray
        Matrix containing all the projection axes
    n_clusters : int
        The total number of clusters
    index_dict : dict
        A dictionary to match the indices of two clusters to a projection axis

    Returns
    -------
    labels_pred : np.ndarray
        The predicted labels for X_test
    """
    points_in_all_clusters = [np.where(labels_train == clus)[0] for clus in range(n_clusters)]
    n_points_in_all_clusters = [points_in_cluster.shape[0] for points_in_cluster in points_in_all_clusters]
    labels_pred_matrix = np.zeros((X_test.shape[0], n_clusters))
    for m in range(n_clusters - 1):
        if n_points_in_all_clusters[m] < 4:
            continue
        for n in range(m + 1, n_clusters):
            if n_points_in_all_clusters[n] < 4:
                continue
            # Get correct projection vector
            projection_vector = projections[index_dict[(m, n)]]
            # Project data
            X_train_m = X_train[points_in_all_clusters[m]]
            X_train_n = X_train[points_in_all_clusters[n]]
            x_proj_m = np.matmul(X_train_m, projection_vector)
            x_proj_n = np.matmul(X_train_n, projection_vector)
            # Sort data
            sorted_indices_m = x_proj_m.argsort()
            sorted_indices_n = x_proj_n.argsort()
            # Execute mirrored dip
            _, low_m, high_m = _dip_mirrored_data(x_proj_m[sorted_indices_m], None)
            low_m_coor = x_proj_m[sorted_indices_m[low_m]]
            high_m_coor = x_proj_m[sorted_indices_m[high_m]]
            _, low_n, high_n = _dip_mirrored_data(x_proj_n[sorted_indices_n], None)
            low_n_coor = x_proj_n[sorted_indices_n[low_n]]
            high_n_coor = x_proj_n[sorted_indices_n[high_n]]
            # Project testdata onto projection line
            x_test_proj = np.matmul(X_test, projection_vector)
            # Check if projected test data matches cluster structure
            if low_m_coor > high_n_coor:  # cluster m right of cluster n
                threshold = (low_m_coor - high_n_coor) / 2
                labels_pred_matrix[x_test_proj <= low_m_coor - threshold, n] += 1
                labels_pred_matrix[x_test_proj >= high_n_coor + threshold, m] += 1
            elif low_n_coor > high_m_coor:  # cluster n right of cluster m
                threshold = (low_n_coor - high_m_coor) / 2
                labels_pred_matrix[x_test_proj <= low_n_coor - threshold, m] += 1
                labels_pred_matrix[x_test_proj >= high_m_coor + threshold, n] += 1
            else:
                center_coor_m = (low_m_coor + high_m_coor) / 2
                center_coor_n = (low_n_coor + high_n_coor) / 2
                if center_coor_m > center_coor_n:  # cluster m right of cluster n
                    threshold = (high_n_coor - low_m_coor) / 2
                    labels_pred_matrix[x_test_proj <= low_m_coor + threshold, n] += 1
                    labels_pred_matrix[x_test_proj >= high_n_coor - threshold, m] += 1
                else:  # cluster n right of cluster m
                    threshold = (high_m_coor - low_n_coor) / 2
                    labels_pred_matrix[x_test_proj <= low_n_coor + threshold, m] += 1
                    labels_pred_matrix[x_test_proj >= high_m_coor - threshold, n] += 1
    # Get best matching cluster
    labels_pred = np.argmax(labels_pred_matrix, axis=1)
    return labels_pred


def _get_rec_loss_of_first_batch(trainloader: torch.utils.data.DataLoader, autoencoder: torch.nn.Module,
                                 loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
    """
    Calculate the reconstruction loss of the first batch of data.
    Therefore, a new instance of the autoencoder will be created using the same architecture.

    Parameters
    ----------
    trainloader : torch.utils.data.DataLoader
        dataloader to be used for training
    autoencoder : torch.nn.Module
        the autoencoder
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction
    device : torch.device
        device to be trained on

    Returns
    -------
    ae_loss : torch.Tensor
        The reconstruction loss of the first batch of data
    """
    autoencoder_class = type(autoencoder)
    # Create new instance of the autoencoder
    if hasattr(autoencoder, "encoder"):
        # In case of Feedforward-based architectures
        tmp_autoencoder = autoencoder_class(layers=autoencoder.encoder.layers,
                                            decoder_layers=autoencoder.decoder.layers).to(device)
    else:
        # In case of Conv-based architectures
        conv_encoder_name = "resnet18" if type(autoencoder.conv_encoder.layer1[0]) is EncoderBlock else "resnet50"
        conv_decoder_name = "resnet18" if type(autoencoder.conv_decoder.layer1[0]) is DecoderBlock else "resnet50"
        tmp_autoencoder = autoencoder_class(input_height=autoencoder.input_height,
                                            fc_layers=autoencoder.fc_encoder.layers,
                                            conv_encoder_name=conv_encoder_name,
                                            fc_decoder_layers=autoencoder.fc_decoder.layers,
                                            conv_decoder_name=conv_decoder_name).to(device)
    # Get first batch of data and calculate reconstruction loss
    batch_init = next(iter(trainloader))
    ae_loss, _, _ = tmp_autoencoder.loss(batch_init, loss_fn, device)
    return ae_loss.detach()


def _dipencoder(X: np.ndarray, n_clusters: int, embedding_size: int, batch_size: int,
                optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss, clustering_epochs: int,
                clustering_optimizer_params: dict, pretrain_epochs: int,
                pretrain_optimizer_params: dict, autoencoder: torch.nn.Module, max_cluster_size_diff_factor: float,
                reconstruction_loss_weight: float, custom_dataloaders: tuple, augmentation_invariance: bool,
                initial_clustering_class: ClusterMixin, initial_clustering_params: dict, labels_gt: np.ndarray,
                random_state: np.random.RandomState, debug: bool) -> (np.ndarray, np.ndarray, dict, torch.nn.Module):
    """
    Start the actual DipEncoder procedure on the input data set.
    If labels_gt is None this method will act as a clustering algorithm else it will only be used to learn an embedding.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    embedding_size : int
        size of the embedding within the autoencoder
    batch_size : int
        size of the data batches for the actual training of the DipEncoder
    optimizer_class : torch.optim.Optimizer
        the optimizer class
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction
    clustering_epochs : int
        number of epochs for the actual clustering procedure
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used
    reconstruction_loss_weight : float
        weight of the reconstruction loss compared to the clustering loss.
        If None it will be equal to 1/(4L), where L is the reconstruction loss of the first batch of an untrained autoencoder
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    labels_gt : no.ndarray
        Ground truth labels. If None, the DipEncoder will be used for clustering
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    debug : bool
        If true, additional information will be printed to the console

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, dict, torch.nn.Module)
        The labels as identified by the DipEncoder,
        The final projection axes between the clusters,
        A dictionary to match the indices of two clusters to a projection axis,
        The final autoencoder
    """
    MIN_NUMBER_OF_POINTS = 10
    # Deep Learning stuff
    device = detect_device()
    # sample random mini-batches from the data -> shuffle = True
    if custom_dataloaders is None:
        trainloader = get_dataloader(X, batch_size, True, False)
        testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
    # Get initial AE
    autoencoder = get_trained_autoencoder(trainloader, n_epochs=pretrain_epochs,
                                          optimizer_params=pretrain_optimizer_params, optimizer_class=optimizer_class,
                                          device=device, loss_fn=loss_fn, embedding_size=embedding_size,
                                          autoencoder=autoencoder)
    # Get factor for AE loss
    # rand_samples = torch.rand((batch_size, X.shape[1]))
    # data_min = np.min(X)
    # data_max = np.max(X)
    # rand_samples_resized = (rand_samples * (data_max - data_min) + data_min).to(device)
    # rand_samples_reconstruction = autoencoder.forward(rand_samples_resized)
    # reconstruction_loss_weight = loss_fn(rand_samples_reconstruction, rand_samples_resized).detach()
    if reconstruction_loss_weight is None:
        reconstruction_loss_weight = _get_rec_loss_of_first_batch(trainloader, autoencoder, loss_fn, device)
        reconstruction_loss_weight = 1 / (4 * reconstruction_loss_weight)
        if debug:
            print("Choose reconstruction_loss_weight automatically; set to", reconstruction_loss_weight)
    # Create initial projections
    X_embed = encode_batchwise(testloader, autoencoder)
    if labels_gt is None:
        # Execute intitial clustering to get labels and centers
        n_clusters, labels_new, centers, _ = run_initial_clustering(X_embed, n_clusters,
                                                                    initial_clustering_class,
                                                                    initial_clustering_params, random_state)
        labels_torch = torch.from_numpy(labels_new)
    else:
        labels_torch = torch.from_numpy(labels_gt)
        centers = np.array([np.mean(X_embed[labels_gt == i], axis=0) for i in range(n_clusters)])
    n_cluster_combinations = int((n_clusters - 1) * n_clusters / 2)
    projections = np.zeros((n_cluster_combinations, embedding_size))
    # Create initial projections vectors by using difference between cluster centers
    index_dict = {}
    for m in range(n_clusters - 1):
        for n in range(m + 1, n_clusters):
            mean_1 = centers[m]
            mean_2 = centers[n]
            v = mean_1 - mean_2
            projections[len(index_dict)] = v
            index_dict[(m, n)] = len(index_dict)
    # Create DipModule
    dip_module = _Dip_Module(projections).to(device)
    # Create SGD Optimizer
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(dip_module.parameters()),
                                **clustering_optimizer_params)
    # Start Optimization
    for iteration in range(clustering_epochs + 1):
        # Update labels for clustering
        if labels_gt is None:
            X_embed = encode_batchwise(testloader, autoencoder)
            labels_new = _predict(X_embed, X_embed,
                                  labels_new,
                                  dip_module.projection_axes.detach().cpu().numpy(),
                                  n_clusters, index_dict)
            labels_torch = torch.from_numpy(labels_new).int().to(device)
        if iteration == clustering_epochs:
            break
        if debug:
            print("iteration:", iteration, "/", clustering_epochs)
            dip_losses = []
            ae_losses = []
        for batch in trainloader:
            ids = batch[0]
            # Reconstruction Loss
            if augmentation_invariance:
                ae_loss_tmp, embedded, _ = autoencoder.loss([batch[0], batch[2]], loss_fn, device)
                ae_loss_tmp_aug, embedded_aug, _ = autoencoder.loss([batch[0], batch[1]], loss_fn, device)
                ae_loss_tmp = (ae_loss_tmp + ae_loss_tmp_aug) / 2
            else:
                ae_loss_tmp, embedded, _ = autoencoder.loss(batch, loss_fn, device)
            ae_loss = ae_loss_tmp * reconstruction_loss_weight
            # Get points within each cluster
            points_in_all_clusters = [torch.where(labels_torch[ids] == clus)[0].to(device) for clus in
                                      range(n_clusters)]
            n_points_in_all_clusters = [points_in_cluster.shape[0] for points_in_cluster in points_in_all_clusters]
            if augmentation_invariance:
                # Regular embedded data will be combined with augmented data
                points_in_all_clusters = [torch.cat((p_c, p_c + embedded.shape[0])) for p_c in points_in_all_clusters]
                n_points_in_all_clusters = [2 * n_c for n_c in n_points_in_all_clusters]
                embedded = torch.cat((embedded, embedded_aug), 0)
            dip_loss = torch.tensor(0)
            for m in range(n_clusters - 1):
                if n_points_in_all_clusters[m] < MIN_NUMBER_OF_POINTS:
                    continue
                for n in range(m + 1, n_clusters):
                    if n_points_in_all_clusters[n] < MIN_NUMBER_OF_POINTS:
                        continue
                    dip_loss_new = _get_dip_error(
                        dip_module, embedded, index_dict[(m, n)], points_in_all_clusters[m], points_in_all_clusters[n],
                        n_points_in_all_clusters[m], n_points_in_all_clusters[n], max_cluster_size_diff_factor,
                        device)
                    dip_loss = dip_loss + dip_loss_new
            final_dip_loss = torch.true_divide(dip_loss, n_cluster_combinations)
            total_loss = final_dip_loss + ae_loss
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # Just for printing
            if debug:
                dip_losses.append(final_dip_loss.item())
                ae_losses.append(ae_loss.item())
        if debug:
            mean_dip_losses = np.mean(dip_losses)
            mean_ae_losses = np.mean(ae_losses)
            print("total loss: {0} (dip loss: {1} / ae loss: {2})".format(mean_dip_losses + mean_ae_losses,
                                                                          mean_dip_losses, mean_ae_losses))
    # Get final labels
    if labels_gt is None:
        X_embed = encode_batchwise(testloader, autoencoder)
        labels_final = _predict(X_embed, X_embed,
                                labels_new,
                                dip_module.projection_axes.detach().cpu().numpy(),
                                n_clusters, index_dict)
        labels_final = labels_final.astype(np.int32)
    else:
        labels_final = labels_gt
    return labels_final, dip_module.projection_axes.detach().cpu().numpy(), index_dict, autoencoder


"""
DipEncoder
"""


class DipEncoder(_AbstractDeepClusteringAlgo):
    """
    The DipEncoder.
    Can be used either as a clustering procedure if no ground truth labels are given or as a supervised dimensionality reduction technique.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DipEncoder loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    batch_size : int
        size of the data batches for the actual training of the DipEncoder.
        Should be larger the more clusters we have. If it is None, it will be set to (25 x n_clusters) (default: None)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 100)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
         loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    max_cluster_size_diff_factor : float
        The maximum different in size when comparing two clusters regarding the number of samples.
        If one cluster surpasses this difference factor, only the max_cluster_size_diff_factor*(size of smaller cluster) closest samples will be used (default: 3)
    reconstruction_loss_weight : float
        weight of the reconstruction loss compared to the clustering loss.
        If None it will be equal to 1/(4L), where L is the reconstruction loss of the first batch of an untrained autoencoder (default: None)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        If true, additional information will be printed to the console (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    projection_axes_ : np.ndarray
        The final projection axes between the clusters
    index_dict_ : dict
        A dictionary to match the indices of two clusters to a projection axis
    autoencoder : torch.nn.Module
        The final autoencoder

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

    def __init__(self, n_clusters: int, batch_size: int = None, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100,
                 clustering_epochs: int = 100, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, max_cluster_size_diff_factor: float = 3,
                 reconstruction_loss_weight: float = None, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, random_state: np.random.RandomState = None,
                 debug: bool = False):
        super().__init__(25 * n_clusters if batch_size is None else batch_size, autoencoder, embedding_size,
                         random_state)
        self.n_clusters = n_clusters
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {
            "lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.max_cluster_size_diff_factor = max_cluster_size_diff_factor
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = {} if initial_clustering_params is None else initial_clustering_params
        self.debug = debug

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DipEncoder':
        """
        Initiate the actual clustering/dimensionality reduction process on the input data set.
        If no ground truth labels are given, the resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            The given (training) data set
        y : np.ndarray
            The ground truth labels. If None, the DipEncoder will be used for clustering (default: None)

        Returns
        -------
        self : DipEncoder
            This instance of the DipEncoder
        """
        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        if y is not None:
            assert len(np.unique(y)) == self.n_clusters, "n_clusters must match number of unique labels in y."
        labels, projection_axes, index_dict, autoencoder = _dipencoder(X, self.n_clusters, self.embedding_size,
                                                                       self.batch_size, self.optimizer_class,
                                                                       self.loss_fn, self.clustering_epochs,
                                                                       self.clustering_optimizer_params,
                                                                       self.pretrain_epochs,
                                                                       self.pretrain_optimizer_params, self.autoencoder,
                                                                       self.max_cluster_size_diff_factor,
                                                                       self.reconstruction_loss_weight,
                                                                       self.custom_dataloaders,
                                                                       self.augmentation_invariance,
                                                                       self.initial_clustering_class,
                                                                       self.initial_clustering_params,
                                                                       y, self.random_state, self.debug)
        self.labels_ = labels
        self.projection_axes_ = projection_axes
        self.index_dict_ = index_dict
        self.autoencoder = autoencoder
        return self

    def predict(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the X_test dataset using the information gained by the fit function and the X_train dataset.
        Beware that the current labels influence the labels obtained by predict(). Therefore, it can occur that the outcome of
        dipencoder.fit(X) does not match dipencoder.predict(X).

        Parameters
        ----------
        X_train : np.ndarray
            The data set used to train the DipEncoder (i.e. to retrieve the projection axes, modal intervals, ...)
        X_test : np.ndarray
            The data set for which we want to retrieve the labels

        Returns
        -------
        labels_pred : np.ndarray
            The predicted labels for X_test
        """
        X_train = self.transform(X_train)
        X_test = self.transform(X_test)
        labels_pred = _predict(X_train, X_test, self.labels_, self.projection_axes_, self.n_clusters,
                               self.index_dict_)
        return labels_pred.astype(np.int32)

    def plot(self, X: np.ndarray, edge_width: float = 0.2, show_legend: bool = True) -> None:
        """
        Plot the current state of the DipEncoder.
        First the data set will be encoded using the autoencoder, afterwards the plot will be created.
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

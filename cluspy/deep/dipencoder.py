from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from cluspy.utils import dip_test
from cluspy.deep.dipdeck import _DipDECK_Autoencoder
import torch
import numpy as np
from cluspy.partition.skinnydip import _dip_mirrored_data
from cluspy.deep._utils import detect_device, encode_batchwise, get_trained_autoencoder

"""
Dip module - holds backward functions
"""


class _Dip_Module(torch.nn.Module):

    def __init__(self, projection_vectors):
        super(_Dip_Module, self).__init__()
        self.projection_vectors = torch.nn.Parameter(torch.from_numpy(projection_vectors).float())

    def forward(self, X, projection_vector_index):
        mydip = _Dip_Gradient.apply(X, self.projection_vectors[projection_vector_index])
        return mydip


class _Dip_Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, projection_vector):
        # Project data onto projection vector
        x_proj = torch.matmul(X, projection_vector)
        # Sort data
        sortedIndices = x_proj.argsort()
        # Calculate dip
        sorted_data = x_proj[sortedIndices].detach().cpu().numpy()
        dip_value, _, modal_triangle = dip_test(sorted_data, is_data_sorted=True, just_dip=False)
        if modal_triangle is None:
            modal_triangle = (-1, -1, -1)
        my_torch_dip = torch.tensor(dip_value)
        # Save parameters for backward
        ctx.save_for_backward(X, x_proj, sortedIndices, projection_vector,
                              torch.tensor(modal_triangle, dtype=torch.long),
                              my_torch_dip)
        return my_torch_dip

    @staticmethod
    def backward(ctx, grad_output):
        device = detect_device()
        # Load parameters from forward
        X, x_proj, sortedIndices, projection_vector, modal_triangle, dip_value = ctx.saved_tensors
        if modal_triangle[0] == -1:
            return torch.zeros((x_proj.shape[0], projection_vector.shape[0])).to(device), torch.zeros(
                projection_vector.shape).to(device)
        # Grad_output equals gradient of outer operations. Update grad_output to consider dip
        if grad_output > 0:
            grad_output = grad_output * dip_value
        else:
            grad_output = grad_output * (0.25 - dip_value)
        # Calculate the partial derivative for all dimensions
        data_index_i1, data_index_i2, data_index_i3 = sortedIndices[modal_triangle]
        # Get A and c
        A = modal_triangle[0] - modal_triangle[1] + \
            (modal_triangle[2] - modal_triangle[0]) * (x_proj[data_index_i2] - x_proj[data_index_i1]) / (
                    x_proj[data_index_i3] - x_proj[data_index_i1])
        constant = torch.true_divide(modal_triangle[2] - modal_triangle[0], 2 * X.shape[0])
        # Check A
        if A < 0:
            constant = -constant
        # Calculate derivative of projection vector
        gradient_proj = _calculate_partial_derivative_proj(X, x_proj, data_index_i1, data_index_i2, data_index_i3)
        gradient_proj = gradient_proj * constant
        # Calculate derivative for projected datapoints
        gradient_x_tmp = _calculate_partial_derivative_x(x_proj, data_index_i1, data_index_i2, data_index_i3, device)
        gradient_x_tmp = gradient_x_tmp * constant
        # Mind the matrix multiplication of the data and the projection
        tmp_vec = torch.ones(X.shape).to(device) * projection_vector
        gradient_x = tmp_vec * gradient_x_tmp.reshape(-1, 1)
        # Return gradients
        return grad_output * gradient_x, grad_output * gradient_proj


def _calculate_partial_derivative_x(X_proj, data_index_i1, data_index_i2, data_index_i3, device):
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


def _calculate_partial_derivative_proj(X, x_proj, data_index_i1, data_index_i2, data_index_i3):
    quotient = (x_proj[data_index_i3] - x_proj[data_index_i1])
    gradient = (X[data_index_i2] - X[data_index_i1]) / quotient - \
               (X[data_index_i3] - X[data_index_i1]) * (
                       x_proj[data_index_i2] - x_proj[data_index_i1]) / quotient ** 2
    return gradient


"""
Module-helpers
"""


def _get_dip_error(dip_module, embedded, j, points_in_m, points_in_n, n_points_in_m, n_points_in_n, device):
    # Calculate dip cluster m
    dip_value_m = dip_module(embedded[points_in_m], j)

    # Calculate dip cluster n
    dip_value_n = dip_module(embedded[points_in_n], j)

    # Calculate dip combined clusters m and n
    diff_size_factor = 3
    if n_points_in_m > diff_size_factor * n_points_in_n:
        perm = torch.randperm(n_points_in_m).to(device)
        sampled_m = points_in_m[perm[:n_points_in_n * diff_size_factor]]
        dip_value_mn = dip_module(torch.cat([embedded[sampled_m], embedded[points_in_n]]), j)
    elif n_points_in_n > diff_size_factor * n_points_in_m:
        perm = torch.randperm(n_points_in_n).to(device)
        sampled_n = points_in_n[perm[:n_points_in_m * diff_size_factor]]
        dip_value_mn = dip_module(torch.cat([embedded[points_in_m], embedded[sampled_n]]), j)
    else:
        dip_value_mn = dip_module(embedded[torch.cat([points_in_m, points_in_n])], j)
    # We want to maximize dip between clusters => set mn loss to -dip
    dip_loss_new = 0.5 * (dip_value_m + dip_value_n) - dip_value_mn
    return dip_loss_new


def _predict(X_train, X_test, labels_train, projections, n_clusters, index_dict):
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
            sortedIndices_m = x_proj_m.argsort()
            sortedIndices_n = x_proj_n.argsort()
            # Execute mirrored dip
            _, low_m, high_m = _dip_mirrored_data(x_proj_m[sortedIndices_m], None)
            low_m_coor = x_proj_m[sortedIndices_m[low_m]]
            high_m_coor = x_proj_m[sortedIndices_m[high_m]]
            _, low_n, high_n = _dip_mirrored_data(x_proj_n[sortedIndices_n], None)
            low_n_coor = x_proj_n[sortedIndices_n[low_n]]
            high_n_coor = x_proj_n[sortedIndices_n[high_n]]
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


def _dipmodule(X, n_clusters, embedding_size, batch_size, optimizer_class, loss_fn, clustering_epochs,
               clustering_learning_rate, pretrain_epochs, pretrain_learning_rate, autoencoder=None,
               labels_gt=None, debug=False):
    MIN_NUMBER_OF_POINTS = 10
    # Deep Learning stuff
    device = detect_device()
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.from_numpy(X).float(), torch.arange(0, X.shape[0]))),
        batch_size=batch_size,
        # sample random mini-batches from the data
        shuffle=True,
        drop_last=False)
    # create a Dataloader to test the autoencoder in mini-batch fashion (Important for validation)
    testloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
                                             batch_size=batch_size,
                                             # Note that we deactivate the shuffling
                                             shuffle=False,
                                             drop_last=False)
    # Get initial AE
    if autoencoder is None:
        autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                              optimizer_class, loss_fn, X.shape[1], embedding_size,
                                              _DipDECK_Autoencoder)
    else:
        autoencoder.to(device)
    # Get factor for AE loss
    rand_samples = torch.rand((batch_size, X.shape[1]))
    data_min = np.min(X)
    data_max = np.max(X)
    rand_samples_resized = (rand_samples * (data_max - data_min) + data_min).to(device)
    rand_samples_reconstruction = autoencoder.forward(rand_samples_resized)
    ae_factor = loss_fn(rand_samples_reconstruction, rand_samples_resized).detach()
    # Create initial projections
    n_cluste_combinations = int((n_clusters - 1) * n_clusters / 2)
    projections = np.zeros((n_cluste_combinations, embedding_size))
    X_embed = encode_batchwise(testloader, autoencoder, device)
    if labels_gt is None:
        # Execute kmeans to get initial clusters
        km = KMeans(n_clusters)
        km.fit(X_embed)
        labels = km.labels_
        centers = km.cluster_centers_
        labels_torch = torch.from_numpy(labels)
    else:
        labels_torch = torch.from_numpy(labels_gt)
        centers = np.array([np.mean(X_embed[labels_gt == i], axis=0) for i in range(n_clusters)])
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
                                lr=clustering_learning_rate)
    # Start Optimization
    for iteration in range(clustering_epochs + 1):
        # Update labels for clustering
        if labels_gt is None:
            X_embed = encode_batchwise(testloader, autoencoder, device)
            labels_new = _predict(X_embed, X_embed,
                                  labels_torch.detach().cpu().numpy(),
                                  dip_module.projection_vectors.detach().cpu().numpy(),
                                  n_clusters, index_dict)
            labels_torch = torch.from_numpy(labels_new).int().to(device)
        if iteration == clustering_epochs:
            break
        if debug:
            print("iteration:", iteration, "/", clustering_epochs)
            losses = []
        for batch, ids in trainloader:
            batch_data = batch.to(device)
            embedded = autoencoder.encode(batch_data)
            # Reconstruction Loss
            reconstruction = autoencoder.decode(embedded)
            ae_loss_tmp = loss_fn(reconstruction, batch_data)
            ae_loss = ae_loss_tmp / ae_factor / 4
            # Get points within each cluster
            points_in_all_clusters = [torch.where(labels_torch[ids] == clus)[0].to(device) for clus in
                                      range(n_clusters)]
            n_points_in_all_clusters = [points_in_cluster.shape[0] for points_in_cluster in points_in_all_clusters]
            dip_loss = torch.tensor(0)
            for m in range(n_clusters - 1):
                if n_points_in_all_clusters[m] < MIN_NUMBER_OF_POINTS:
                    continue
                for n in range(m + 1, n_clusters):
                    if n_points_in_all_clusters[n] < MIN_NUMBER_OF_POINTS:
                        continue
                    dip_loss_new = _get_dip_error(
                        dip_module, embedded, index_dict[(m, n)], points_in_all_clusters[m], points_in_all_clusters[n],
                        n_points_in_all_clusters[m], n_points_in_all_clusters[n], device)
                    dip_loss = dip_loss + dip_loss_new
            final_dip_loss = torch.true_divide(dip_loss, n_cluste_combinations)
            total_loss = final_dip_loss + ae_loss
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # Just for printing
            if debug:
                losses.append(final_dip_loss.item())
        if debug:
            print("dip loss:", np.mean(losses), " / ae loss:", ae_loss.item())
    return labels_torch.detach().cpu().numpy(), autoencoder, dip_module.projection_vectors.detach().cpu().numpy(), index_dict


"""
DipEncoder
"""


class DipEncoder(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, batch_size=None, pretrain_learning_rate=1e-3, clustering_learning_rate=1e-4,
                 pretrain_epochs=100, clustering_epochs=100, optimizer_class=torch.optim.Adam,
                 loss_fn=torch.nn.MSELoss(), autoencoder=None, embedding_size=10, debug=False):
        self.n_clusters = n_clusters
        if batch_size is None:
            batch_size = 30 * n_clusters
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.debug = debug

    def fit(self, X, y=None):
        if y is not None:
            assert len(np.unique(y)) == self.n_clusters, "n_clusters must match number of unique labels in y."
        labels, autoencoder, projection_vecotrs, index_dict = _dipmodule(X, self.n_clusters, self.embedding_size,
                                                                         self.batch_size, self.optimizer_class,
                                                                         self.loss_fn, self.clustering_epochs,
                                                                         self.clustering_learning_rate,
                                                                         self.pretrain_epochs,
                                                                         self.pretrain_learning_rate, self.autoencoder,
                                                                         y, self.debug)
        self.labels_ = labels
        self.projection_vecotrs_ = projection_vecotrs
        self.index_dict_ = index_dict
        self.autoencoder = autoencoder
        return self

    def predict(self, X, X_test):
        testloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
                                                 batch_size=self.batch_size,
                                                 # Note that we deactivate the shuffling
                                                 shuffle=False,
                                                 drop_last=False)
        testloader_supervised = torch.utils.data.DataLoader(torch.from_numpy(X_test).float(),
                                                            batch_size=self.batch_size,
                                                            # Note that we deactivate the shuffling
                                                            shuffle=False,
                                                            drop_last=False)
        device = detect_device()
        X_train = encode_batchwise(testloader, self.autoencoder, device)
        X_test = encode_batchwise(testloader_supervised, self.autoencoder, device)
        labels_pred = _predict(X_train, X_test, self.labels_, self.projection_vecotrs_, self.n_clusters, self.index_dict_)
        return labels_pred

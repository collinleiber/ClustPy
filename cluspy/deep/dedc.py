from scipy.spatial.distance import cdist
import numpy as np
from cluspy.utils import dip_test, PVAL_BY_FUNCTION
import torch
from cluspy.deep._utils import detect_device, encode_batchwise, Simple_Autoencoder, \
    squared_euclidean_distance, int_to_one_hot, get_trained_autoencoder
from sklearn.cluster import KMeans


def _dedc(X, n_clusters_start, dip_merge_threshold, cluster_loss_weight, n_clusters_max, n_clusters_min, batch_size,
          learning_rate, pretrain_epochs, dedc_epochs, update_pause_epochs, optimizer_class, loss_fn, autoencoder,
          embedding_size, debug):
    if n_clusters_max < n_clusters_min:
        raise Exception("n_clusters_max can not be smaller than n_clusters_min")
    if n_clusters_min <= 0:
        raise Exception("n_clusters_min must be greater than zero")
    if n_clusters_start < n_clusters_min:
        raise Exception("n_clusters can not be smaller than n_clusters_min")
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
    if autoencoder is None:
        autoencoder = get_trained_autoencoder(trainloader, learning_rate, pretrain_epochs, device,
                                              optimizer_class, loss_fn, X.shape[1], embedding_size, _DEDC_Autoencoder)
    # Execute kmeans in embedded space - initial clustering
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters_start)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    cluster_labels_cpu = kmeans.labels_
    # Get nearest points to optimal centers
    centers_cpu, embedded_centers_cpu = _get_nearest_points_to_optimal_centers(X, init_centers, embedded_data)
    # Initial dip values
    dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_start)
    # Reduce learning_rate from pretraining by a magnitude of 10
    dedc_learning_rate = learning_rate * 0.1
    optimizer = optimizer_class(autoencoder.parameters(), lr=dedc_learning_rate)
    # Start training
    cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder = _dedc_training(X, n_clusters_start,
                                                                                  dip_merge_threshold,
                                                                                  cluster_loss_weight, centers_cpu,
                                                                                  cluster_labels_cpu,
                                                                                  dip_matrix_cpu, n_clusters_max,
                                                                                  n_clusters_min, dedc_epochs,
                                                                                  update_pause_epochs, optimizer,
                                                                                  loss_fn, autoencoder, device,
                                                                                  trainloader, testloader, debug)
    # Return results
    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder


def _dedc_training(X, n_clusters_current, dip_merge_threshold, cluster_loss_weight, centers_cpu, cluster_labels_cpu,
                   dip_matrix_cpu, n_clusters_max, n_clusters_min, dedc_epochs, update_pause_epochs,
                   optimizer, loss_fn, autoencoder, device, trainloader, testloader, debug):
    i = 0
    while i < dedc_epochs:
        cluster_labels_torch = torch.from_numpy(cluster_labels_cpu).long().to(device)
        centers_torch = torch.from_numpy(centers_cpu).float().to(device)
        dip_matrix_torch = torch.from_numpy(dip_matrix_cpu).float().to(device)
        for batch, ids in trainloader:
            batch_data = batch.to(device)
            embedded = autoencoder.encode(batch_data)
            reconstruction = autoencoder.decode(embedded)
            embedded_centers_torch = autoencoder.encode(centers_torch)
            # Reconstruction Loss
            ae_loss = loss_fn(reconstruction, batch_data)
            # Get distances between points and centers. Get nearest center
            squared_diffs = squared_euclidean_distance(embedded_centers_torch, embedded)
            # Update labels? Pause is needed, so cluster labels can adjust to the new structure
            if i >= update_pause_epochs:
                # Update labels
                current_labels = squared_diffs.argmin(1)
                # cluster_labels_torch[ids] = current_labels
            else:
                current_labels = cluster_labels_torch[ids]
            onehot_labels = int_to_one_hot(current_labels, n_clusters_current).float()
            # Get dip costs
            dip_matrix_eye = dip_matrix_torch + torch.eye(n_clusters_current, device=device)
            dip_matrix_eye /= dip_matrix_eye.sum(1).reshape((-1,1))
            cluster_relationships = torch.matmul(onehot_labels, dip_matrix_eye)
            escaped_diffs = cluster_relationships * squared_diffs
            # Normalize loss by cluster distances
            squared_center_diffs = squared_euclidean_distance(embedded_centers_torch, embedded_centers_torch)
            # Ignore zero values (diagonal)
            mask = torch.where(squared_center_diffs != 0)
            masked_center_diffs = squared_center_diffs[mask[0], mask[1]]
            sqrt_masked_center_diffs = masked_center_diffs.sqrt()
            masked_center_diffs_std = sqrt_masked_center_diffs.std() if len(sqrt_masked_center_diffs) > 2 else 0
            # Loss function
            cluster_loss = escaped_diffs.sum(1).mean() * (
                    1 + masked_center_diffs_std) / sqrt_masked_center_diffs.mean()
            cluster_loss *= cluster_loss_weight
            loss = ae_loss + cluster_loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # cluster_labels_cpu = cluster_labels_torch.detach().cpu().numpy()
        # Update centers
        embedded_data = encode_batchwise(testloader, autoencoder, device)
        embedded_centers_cpu = autoencoder.encode(centers_torch).detach().cpu().numpy()
        if i >= update_pause_epochs:
            cluster_labels_cpu = np.argmin(cdist(embedded_centers_cpu, embedded_data), axis=0)
        optimal_centers = np.array([np.mean(embedded_data[cluster_labels_cpu == cluster_id], axis=0) for cluster_id in
                                    range(n_clusters_current)])
        centers_cpu, embedded_centers_cpu = _get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data)
        # Update Dips
        dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_current)

        if debug:
            print(
                "Iteration {0}  (n_clusters = {4}) - reconstruction loss: {1} / cluster loss: {2} / total loss: {3}".format(
                    i, ae_loss.item(), cluster_loss.item(), loss.item(), n_clusters_current))
            print("max dip", np.max(dip_matrix_cpu), " at ",
                  np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape))
        # i is increased here. Else next iteration will start with i = 1 instead of 0 after a merge
        i += 1
        # Start merging procedure
        if i > update_pause_epochs:
            # Is merge possible?
            dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)
            while dip_matrix_cpu[dip_argmax] >= dip_merge_threshold and n_clusters_current > n_clusters_min:
                if debug:
                    print("Start merging in iteration {0}.\nMerging clusters {1} with dip value {2}.".format(i,
                                                                                                             dip_argmax,
                                                                                                             dip_matrix_cpu[
                                                                                                                 dip_argmax]))
                # Reset iteration and reduce number of cluster
                i = 0
                n_clusters_current -= 1
                cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu = \
                    _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current, centers_cpu,
                                        embedded_centers_cpu)
                dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)
        # Optional: Force merging of clusters
        if i == dedc_epochs and n_clusters_current > n_clusters_max:
            # Get smallest cluster
            _, cluster_sizes = np.unique(cluster_labels_cpu, return_counts=True)
            smallest_cluster_id = np.argmin(cluster_sizes)
            smallest_cluster_size = cluster_sizes[smallest_cluster_id]
            i = 0
            n_clusters_current -= 1
            # Is smallest cluster small enough for deletion?
            if smallest_cluster_size < 0.2 * np.mean(cluster_sizes):
                if debug:
                    print(
                        "Remove smallest cluster {0} with size {1}".format(smallest_cluster_id, smallest_cluster_size))
                distances_to_clusters = cdist(embedded_centers_cpu,
                                              embedded_data[cluster_labels_cpu == smallest_cluster_id])
                # Set dist to center which is being removed to inf
                distances_to_clusters[smallest_cluster_id, :] = np.inf
                cluster_labels_cpu[cluster_labels_cpu == smallest_cluster_id] = np.argmin(distances_to_clusters, axis=0)
                cluster_labels_cpu[cluster_labels_cpu >= smallest_cluster_id] -= 1
                optimal_centers = np.array(
                    [np.mean(embedded_data[cluster_labels_cpu == cluster_id], axis=0) for cluster_id in
                     range(n_clusters_current)])
                centers_cpu, embedded_centers_cpu = _get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data)
                # Update dip values
                dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu,
                                                  n_clusters_current)
            else:
                # Else: merge clusters with hightest dip
                if debug:
                    print("Force merge of clusters {0} with dip value {1}".format(dip_argmax,
                                                                                  dip_matrix_cpu[dip_argmax]))

                cluster_labels_cpu, centers_cpu, _, dip_matrix_cpu = \
                    _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current, centers_cpu,
                                        embedded_centers_cpu)
        if n_clusters_current == 1:
            if debug:
                print("Only one cluster left")
            break
    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder


def _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current, centers_cpu,
                        embedded_centers_cpu):
    # Get points in clusters
    points_in_center_1 = len(cluster_labels_cpu[cluster_labels_cpu == dip_argmax[0]])
    points_in_center_2 = len(cluster_labels_cpu[cluster_labels_cpu == dip_argmax[1]])
    # update labels
    for j, l in enumerate(cluster_labels_cpu):
        if l == dip_argmax[0] or l == dip_argmax[1]:
            cluster_labels_cpu[j] = n_clusters_current - 1
        elif l < dip_argmax[0] and l < dip_argmax[1]:
            cluster_labels_cpu[j] = l
        elif l > dip_argmax[0] and l > dip_argmax[1]:
            cluster_labels_cpu[j] = l - 2
        else:
            cluster_labels_cpu[j] = l - 1
    # Find new center position
    optimal_new_center = (embedded_centers_cpu[dip_argmax[0]] * points_in_center_1 +
                          embedded_centers_cpu[dip_argmax[1]] * points_in_center_2) / (
                                 points_in_center_1 + points_in_center_2)
    new_center_cpu, new_embedded_center_cpu = _get_nearest_points_to_optimal_centers(X, [optimal_new_center], embedded_data)
    # Remove the two old centers and add the new one
    centers_cpu_tmp = np.delete(centers_cpu, dip_argmax, axis=0)
    centers_cpu = np.append(centers_cpu_tmp, new_center_cpu, axis=0)
    embedded_centers_cpu_tmp = np.delete(embedded_centers_cpu, dip_argmax, axis=0)
    embedded_centers_cpu = np.append(embedded_centers_cpu_tmp, new_embedded_center_cpu, axis=0)
    # Update dip values
    dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu,
                                      n_clusters_current)
    return cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu


def _get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data):
    best_center_points = np.argmin(cdist(optimal_centers, embedded_data), axis=1)
    centers_cpu = X[best_center_points, :]
    embedded_centers_cpu = embedded_data[best_center_points, :]
    return centers_cpu, embedded_centers_cpu


def _get_nearest_points(all_points, center, sample_size, min_n_dip_samples):
    distances = cdist(all_points, [center])
    nearest_points = np.argsort(distances, axis=0)
    # Check if more points should be taken because the other cluster is too small
    if sample_size < min_n_dip_samples:
        sample_size = min(min_n_dip_samples, len(all_points))
    # OLD: n_points = max(number_of_points, min(min_number_of_points, len(all_points)))
    subset_all_points = all_points[nearest_points[:sample_size, 0]]
    return subset_all_points


def _get_dip_matrix(data, dip_centers, dip_labels, n_clusters, max_cluster_size_diff_factor=2, min_n_dip_samples=100):
    dip_matrix = np.zeros((n_clusters, n_clusters))
    # Loop over all combinations of centers
    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            center_diff = dip_centers[i] - dip_centers[j]
            points_in_i = data[dip_labels == i]
            points_in_j = data[dip_labels == j]
            points_in_i_or_j = np.append(points_in_i, points_in_j, axis=0)
            proj_points = np.dot(points_in_i_or_j, center_diff)
            _, dip_p_value = dip_test(proj_points, pval_strategy=PVAL_BY_FUNCTION)
            # Check if clusters sizes differ heavily
            if points_in_i.shape[0] > points_in_j.shape[0] * max_cluster_size_diff_factor or \
                    points_in_j.shape[0] > points_in_i.shape[0] * max_cluster_size_diff_factor:
                if points_in_i.shape[0] > points_in_j.shape[0] * max_cluster_size_diff_factor:
                    points_in_i = _get_nearest_points(points_in_i, dip_centers[j], points_in_j.shape[0] *
                                                      max_cluster_size_diff_factor, min_n_dip_samples)
                elif points_in_j.shape[0] > points_in_i.shape[0] * max_cluster_size_diff_factor:
                    points_in_j = _get_nearest_points(points_in_j, dip_centers[i], points_in_i.shape[0] *
                                                      max_cluster_size_diff_factor, min_n_dip_samples)
                points_in_i_or_j = np.append(points_in_i, points_in_j, axis=0)
                proj_points = np.dot(points_in_i_or_j, center_diff)
                _, dip_p_value_2 = dip_test(proj_points, pval_strategy=PVAL_BY_FUNCTION)
                dip_p_value = min(dip_p_value, dip_p_value_2)
            # Add pval to dip matrix
            dip_matrix[i][j] = dip_p_value
            dip_matrix[j][i] = dip_p_value
    return dip_matrix


class _DEDC_Autoencoder(Simple_Autoencoder):

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        for _ in range(n_epochs):
            for batch, _ in trainloader:
                # load batch on device
                batch_data = batch.to(device)
                reconstruction = self.forward(batch_data)
                loss = loss_fn(reconstruction, batch_data)
                # reset gradients from last iteration
                optimizer.zero_grad()
                # calculate gradients and reset the computation graph
                loss.backward()
                # update the internal params (weights, etc.)
                optimizer.step()


class DEDC():

    def __init__(self, n_clusters_start=35, dip_merge_threshold=0.9, cluster_loss_weight=1, n_clusters_max=np.inf,
                 n_clusters_min=1, batch_size=256, learning_rate=1e-3, pretrain_epochs=100, dedc_epochs=50,
                 update_pause_epochs=5, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(),
                 autoencoder=None, embedding_size=5, debug=False):
        self.n_clusters_start = n_clusters_start
        self.dip_merge_threshold = dip_merge_threshold
        self.cluster_loss_weight = cluster_loss_weight
        self.n_clusters_max = n_clusters_max
        self.n_clusters_min = n_clusters_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.dedc_epochs = dedc_epochs
        self.update_pause_epochs = update_pause_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.debug = debug

    def fit(self, X):
        labels, n_clusters, centers, autoencoder = _dedc(X, self.n_clusters_start, self.dip_merge_threshold,
                                                         self.cluster_loss_weight, self.n_clusters_max,
                                                         self.n_clusters_min, self.batch_size, self.learning_rate,
                                                         self.pretrain_epochs, self.dedc_epochs,
                                                         self.update_pause_epochs, self.optimizer_class,
                                                         self.loss_fn, self.autoencoder, self.embedding_size,
                                                         self.debug)
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.autoencoder = autoencoder

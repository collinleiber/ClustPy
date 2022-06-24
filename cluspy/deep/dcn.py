"""
Yang, Bo, et al. "Towards k-means-friendly spaces:
Simultaneous deep learning and clustering." international
conference on machine learning. PMLR, 2017.

@authors Lukas Miklautz, Dominik Mautz
"""

from cluspy.deep._utils import detect_device, encode_batchwise, \
    squared_euclidean_distance, predict_batchwise
from ._data_utils import get_dataloader
from ._train_utils import get_trained_autoencoder
import torch
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin


def _dcn(X, n_clusters, batch_size, pretrain_learning_rate, clustering_learning_rate, pretrain_epochs,
         clustering_epochs, optimizer_class, loss_fn, autoencoder, embedding_size, degree_of_space_distortion,
         degree_of_space_preservation):
    device = detect_device()
    trainloader = get_dataloader(X, batch_size, True, False)
    testloader = get_dataloader(X, batch_size, False, False)

    autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, X.shape[1], embedding_size, autoencoder)
    # Execute kmeans in embedded space
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    # Setup DCN Module
    dcn_module = _DCN_Module(init_centers).to_device(device)
    # Use DCN learning_rate (usually pretrain_learning_rate reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()), lr=clustering_learning_rate)
    # DEC Training loop
    dcn_module.start_training(autoencoder, trainloader, clustering_epochs, device, optimizer, loss_fn,
                              degree_of_space_distortion, degree_of_space_preservation)
    # Get labels
    dcn_labels = predict_batchwise(testloader, autoencoder, dcn_module, device)
    dcn_centers = dcn_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dcn_labels, dcn_centers, autoencoder


def _compute_centroids(centers, embedded, count, s):
    n = embedded.shape[0]
    ta = embedded.unsqueeze(1)
    for i in range(n):
        c = s[i].item()
        count[c] += 1
        eta = 1.0 / count[c].item()
        centers[c] = (1 - eta) * centers[c] + eta * ta[i]
    return centers, count


class _DCN_Module(torch.nn.Module):
    def __init__(self, init_np_centers):
        super().__init__()
        self.centers = torch.tensor(init_np_centers)

    def compression_loss(self, embedded, weights=None) -> torch.Tensor:
        dist = squared_euclidean_distance(self.centers, embedded, weights=weights)
        loss = (dist.min(dim=1)[0]).mean()
        return loss

    def prediction_hard(self, embedded, weights=None) -> torch.Tensor:
        dist = squared_euclidean_distance(self.centers, embedded, weights=weights)
        s = (dist.min(dim=1)[1])
        return s

    def update_centroids(self, embedded, count, s) -> torch.Tensor:
        self.centers, count = _compute_centroids(self.centers, embedded, count, s)
        return count

    def to_device(self, device):
        self.centers = self.centers.to(device)
        self.to(device)
        return self

    def start_training(self, autoencoder, trainloader, n_epochs, device, optimizer, loss_fn,
                       degree_of_space_distortion, degree_of_space_preservation):
        # DCN training loop
        # Init for count from original DCN code (not reported in Paper)
        # This means centroid learning rate at the beginning is scaled by a hundred
        count = torch.ones(self.centers.shape[0], dtype=torch.int32) * 100
        for _ in range(n_epochs):
            # Update Network
            for batch in trainloader:
                batch_data = batch[1].to(device)
                embedded = autoencoder.encode(batch_data)
                reconstruction = autoencoder.decode(embedded)

                # compute reconstruction loss
                ae_loss = loss_fn(batch_data, reconstruction)
                # compute cluster loss
                cluster_loss = self.compression_loss(embedded)
                # compute total loss
                loss = degree_of_space_preservation * ae_loss + 0.5 * degree_of_space_distortion * cluster_loss
                # Backward pass - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update Assignments and Centroids
            with torch.no_grad():
                for batch in trainloader:
                    batch_data = batch[1].to(device)
                    embedded = autoencoder.encode(batch_data)

                    ## update centroids [on gpu] About 40 seconds for 1000 iterations
                    ## No overhead from loading between gpu and cpu
                    # count = cluster_module.update_centroid(embedded, count, s)

                    # update centroids [on cpu] About 30 Seconds for 1000 iterations
                    # with additional overhead from loading between gpu and cpu
                    embedded = embedded.cpu()
                    self.centers = self.centers.cpu()

                    # update assignments
                    s = self.prediction_hard(embedded)

                    # update centroids
                    count = self.update_centroids(embedded, count.cpu(), s.cpu())
                    count = count.to(device)
                    self.centers = self.centers.to(device)


class DCN(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, batch_size=256, pretrain_learning_rate=1e-3, clustering_learning_rate=1e-4,
                 pretrain_epochs=100, clustering_epochs=150, optimizer_class=torch.optim.Adam,
                 loss_fn=torch.nn.MSELoss(), degree_of_space_distortion=0.05, degree_of_space_preservation=1.0,
                 autoencoder=None, embedding_size=10):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size

    def fit(self, X, y=None):
        kmeans_labels, kmeans_centers, dcn_labels, dcn_centers, autoencoder = _dcn(X, self.n_clusters, self.batch_size,
                                                                                   self.pretrain_learning_rate,
                                                                                   self.clustering_learning_rate,
                                                                                   self.pretrain_epochs,
                                                                                   self.clustering_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.degree_of_space_distortion,
                                                                                   self.degree_of_space_preservation)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dcn_labels_ = dcn_labels
        self.dcn_cluster_centers_ = dcn_centers
        self.autoencoder = autoencoder
        return self

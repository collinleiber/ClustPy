from cluspy.deep._utils import detect_device, _get_trained_autoencoder, encode_batchwise, squared_euclidean_distance, \
    predict_batchwise
import torch
from sklearn.cluster import KMeans


def _dcn(X, n_clusters, batch_size, learning_rate, pretrain_iterations, dcn_iterations, optimizer_class,
         loss_fn, autoencoder, embedding_size, degree_of_space_distortion, degree_of_space_preservation):
    device = detect_device()
    trainloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
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
        autoencoder = _get_trained_autoencoder(trainloader, learning_rate, pretrain_iterations, device,
                                               optimizer_class, loss_fn, X.shape[1], embedding_size)
    # Execute kmeans in embedded space
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    # Setup DCN Module
    dcn_module = _DCN_Module(init_centers)
    # Reduce learning_rate from pretraining by a magnitude of 10
    dcn_learning_rate = learning_rate * 0.1
    optimizer = optimizer_class(list(autoencoder.parameters()), lr=dcn_learning_rate)
    # DEC Training loop
    dcn_module.train(autoencoder, trainloader, dcn_iterations, device, optimizer, loss_fn,
                     degree_of_space_distortion, degree_of_space_preservation)
    # Get labels
    labels = predict_batchwise(testloader, autoencoder, dcn_module, device)
    return labels


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

    def train(self, autoencoder, trainloader, training_iterations, device, optimizer, loss_fn,
              degree_of_space_distortion, degree_of_space_preservation):
        # DCN training loop
        i = 0
        # Init for count from original DCN code (not reported in Paper)
        # This means centroid learning rate at the beginning is scaled by a hundred
        count = torch.ones(self.centers.shape[0], dtype=torch.int32) * 100
        while (i < training_iterations):  # each iteration is equal to an epoch
            # Update Network
            for batch in trainloader:
                batch_data = batch.to(device)
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
                    batch_data = batch.to(device)
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
                    i += 1


class DCN():

    def __init__(self, n_clusters, batch_size=256, learning_rate=1e-3, pretrain_iterations=50000,
                 dcn_iterations=40000, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(),
                 degree_of_space_distortion=0.05, degree_of_space_preservation=1.0, autoencoder=None, embedding_size=10):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_iterations = pretrain_iterations
        self.dcn_iterations = dcn_iterations
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size

    def fit(self, X):
        labels = _dcn(X, self.n_clusters, self.batch_size, self.learning_rate, self.pretrain_iterations,
                      self.dcn_iterations, self.optimizer_class, self.loss_fn, self.autoencoder, self.embedding_size,
                      self.degree_of_space_distortion, self.degree_of_space_preservation)
        self.labels_ = labels

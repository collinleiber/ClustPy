"""
Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised
deep embedding for clustering analysis." International
conference on machine learning. 2016.

and

Guo, Xifeng, et al. "Improved deep embedded clustering with
local structure preservation." IJCAI. 2017.

@authors Lukas Miklautz, Dominik Mautz
"""

from cluspy.deep._utils import detect_device, get_trained_autoencoder, encode_batchwise, \
    squared_euclidean_distance, predict_batchwise
import torch
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin


def _dec(X, n_clusters, alpha, batch_size, pretrain_learning_rate, dec_learning_rate, pretrain_epochs, dec_epochs, 
         optimizer_class, loss_fn, autoencoder, embedding_size, use_reconstruction_loss, cluster_loss_weight):
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
        autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                              optimizer_class, loss_fn, X.shape[1], embedding_size)
    else:
        autoencoder.to(device)
    # Execute kmeans in embedded space - initial clustering
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    init_centers = kmeans.cluster_centers_
    # Setup DEC Module
    dec_module = _DEC_Module(init_centers, alpha).to(device)
    # Use DEC learning_rate (usually pretrain_learning_rate reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(dec_module.parameters()), lr=dec_learning_rate)
    # DEC Training loop
    dec_module.start_training(autoencoder, trainloader, dec_epochs, device, optimizer, loss_fn, use_reconstruction_loss,
                              cluster_loss_weight)
    # Get labels
    dec_labels = predict_batchwise(testloader, autoencoder, dec_module, device)
    dec_centers = dec_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dec_labels, dec_centers, autoencoder


def _dec_prediction(centers, embedded, alpha=1.0, weights=None):
    squared_diffs = squared_euclidean_distance(centers, embedded, weights)
    numerator = (1.0 + squared_diffs / alpha).pow(-1.0 * (alpha + 1.0) / 2.0)
    denominator = numerator.sum(1)
    prob = numerator / denominator.unsqueeze(1)
    return prob


def _dec_compression_value(pred_labels):
    soft_freq = pred_labels.sum(0)
    squared_pred = pred_labels.pow(2)
    normalized_squares = squared_pred / soft_freq.unsqueeze(0)
    sum_normalized_squares = normalized_squares.sum(1)
    p = normalized_squares / sum_normalized_squares.unsqueeze(1)
    return p


def _dec_compression_loss_fn(q):
    p = _dec_compression_value(q).detach().data
    loss = -1.0 * torch.mean(torch.sum(p * torch.log(q + 1e-8), dim=1))
    return loss


class _DEC_Module(torch.nn.Module):
    def __init__(self, init_np_centers, alpha):
        super().__init__()
        self.alpha = alpha
        # Centers are learnable parameters
        self.centers = torch.nn.Parameter(torch.tensor(init_np_centers), requires_grad=True)

    def prediction(self, embedded, weights=None) -> torch.Tensor:
        """Soft prediction $q$"""
        return _dec_prediction(self.centers, embedded, self.alpha, weights=weights)

    def prediction_hard(self, embedded, weights=None) -> torch.Tensor:
        """Hard prediction"""
        return self.prediction(embedded, weights=weights).argmax(1)

    def compression_loss(self, embedded, weights=None) -> torch.Tensor:
        """Loss of DEC"""
        prediction = _dec_prediction(self.centers, embedded, self.alpha, weights=weights)
        loss = _dec_compression_loss_fn(prediction)
        return loss

    def start_training(self, autoencoder, trainloader, n_epochs, device, optimizer, loss_fn, use_reconstruction_loss,
                       cluster_loss_weight):
        for _ in range(n_epochs):
            for batch in trainloader:
                batch_data = batch.to(device)
                embedded = autoencoder.encode(batch_data)

                cluster_loss = self.compression_loss(embedded)
                loss = cluster_loss * cluster_loss_weight
                # Reconstruction loss is not included in DEC
                if use_reconstruction_loss:
                    reconstruction = autoencoder.decode(embedded)
                    ae_loss = loss_fn(batch_data, reconstruction)
                    loss += ae_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class DEC(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, alpha=1.0, batch_size=256, pretrain_learning_rate=1e-3, dec_learning_rate=1e-4,
                 pretrain_epochs=100, dec_epochs=150, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(),
                 autoencoder=None, embedding_size=10, use_reconstruction_loss=False, cluster_loss_weight=1):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_learning_rate
        self.dec_learning_rate = dec_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.dec_epochs = dec_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.use_reconstruction_loss = use_reconstruction_loss
        self.cluster_loss_weight = cluster_loss_weight

    def fit(self, X, y=None):
        kmeans_labels, kmeans_centers, dec_labels, dec_centers, autoencoder = _dec(X, self.n_clusters, self.alpha,
                                                                                   self.batch_size,
                                                                                   self.pretrain_learning_rate,
                                                                                   self.dec_learning_rate,
                                                                                   self.pretrain_epochs,
                                                                                   self.dec_epochs,
                                                                                   self.optimizer_class, self.loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.use_reconstruction_loss, 
                                                                                   self.cluster_loss_weight)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dec_labels_ = dec_labels
        self.dec_cluster_centers_ = dec_centers
        self.autoencoder = autoencoder
        return self


class IDEC(DEC):

    def __init__(self, n_clusters, alpha=1.0, batch_size=256, pretrain_learning_rate=1e-3, idec_learning_rate=1e-4,
                 pretrain_epochs=100, idec_epochs=150, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(),
                 autoencoder=None, embedding_size=10):
        super().__init__(n_clusters, alpha, batch_size, pretrain_learning_rate, idec_learning_rate, pretrain_epochs,
              idec_epochs, optimizer_class, loss_fn, autoencoder, embedding_size, True, 0.1)

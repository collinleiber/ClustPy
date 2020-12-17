"""
Jiang, Zhuxi, et al. "Variational deep embedding: An
unsupervised and generative approach to clustering." arXiv
preprint arXiv:1611.05148 (2016).
"""

import torch
from cluspy.deep._utils import detect_device
import numpy as np
from sklearn.mixture import GaussianMixture


def _vade(X, n_clusters, batch_size, learning_rate, pretrain_epochs, vade_epochs, optimizer_class,
          loss_fn, autoencoder, embedding_size):
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
        autoencoder = _VadeAutoencoder(input_dim=X.shape[1], embedding_size=embedding_size).to(device)
        ae_optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
        autoencoder.start_training(trainloader, pretrain_epochs, device, ae_optimizer, loss_fn)
    # Execute EM in embedded space
    embedded_data = _vade_encode_batchwise(testloader, autoencoder, device)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=100)
    gmm.fit(embedded_data)
    # Initialize VaDE
    vade_module = _VaDE_Module(autoencoder, n_clusters=n_clusters, embedding_size=10, pi=gmm.weights_,
                               mean=gmm.means_, var=gmm.covariances_, device=device).to(device)
    # Reduce learning_rate from pretraining by a magnitude of 10
    vade_learning_rate = learning_rate * 0.1
    optimizer = optimizer_class(vade_module.parameters(), lr=vade_learning_rate)
    # Vade Training loop
    vade_module.start_training(trainloader, vade_epochs, device, optimizer, loss_fn)
    # Get labels
    vade_labels = _vade_predict_batchwise(testloader, vade_module, device)
    vade_centers = vade_module.p_mean.detach().cpu().numpy()
    vade_covariances = vade_module.p_var.detach().cpu().numpy()
    # Do reclustering with GMM
    embedded_data = _vade_encode_batchwise(testloader, vade_module, device)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=100)
    gmm_labels = gmm.fit_predict(embedded_data)
    # Return results
    return gmm_labels, gmm.means_, gmm.covariances_, vade_labels, vade_centers, vade_covariances, autoencoder


def _sampling(q_mean, q_var):
    std = torch.exp(0.5 * q_var)
    eps = torch.randn_like(std)
    z = q_mean + eps * std
    return z


def _get_gamma(pi, p_mean, p_var, z):
    z = z.unsqueeze(1)
    p_var = p_var.unsqueeze(0)
    pi = pi.unsqueeze(0)

    p_z_c = -torch.sum(0.5 * (np.log(2 * np.pi)) + p_var + ((z - p_mean).pow(2) / (2. * torch.exp(p_var))), dim=2)
    p_c_z_c = torch.exp(torch.log(pi) + p_z_c) + 1e-10
    p_c_z = p_c_z_c / torch.sum(p_c_z_c, dim=1, keepdim=True)

    return p_c_z


def _compute_loss(pi, p_mean, p_var, q_mean, q_var, batch_data, p_c_z, reconstruction, loss_fn):
    q_mean = q_mean.unsqueeze(1)
    p_var = p_var.unsqueeze(0)

    p_x_z = loss_fn(reconstruction, batch_data)

    p_z_c = torch.sum(p_c_z * (0.5 * np.log(2 * np.pi) + 0.5 * (
            torch.sum(p_var, dim=2) + torch.sum(torch.exp(q_var.unsqueeze(1)) / torch.exp(p_var),
                                                dim=2) + torch.sum((q_mean - p_mean).pow(2) / torch.exp(p_var),
                                                                   dim=2))))
    p_c = torch.sum(p_c_z * torch.log(pi))
    q_z_x = 0.5 * (np.log(2 * np.pi)) + 0.5 * torch.sum(1 + q_var)
    q_c_x = torch.sum(p_c_z * torch.log(p_c_z))

    loss = p_x_z + p_z_c - p_c - q_z_x + q_c_x
    loss /= batch_data.size(0)
    return loss


def _vade_predict_batchwise(dataloader, vade_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch.to(device)
        q_mean, q_var = vade_module.encode(batch_data)
        prediction = vade_module.predict(q_mean, q_var).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()


def _vade_encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    for batch in dataloader:
        batch_data = batch.to(device)
        q_mean, q_var = model.encode(batch_data)
        embeddings.append(q_mean.detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


class _VadeAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_size: int):
        super(_VadeAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 2000),
            torch.nn.ReLU(inplace=True),
        )

        # naming is used for later correspondence in VaDE
        self.mean_ = torch.nn.Linear(2000, embedding_size)
        # is only initialized
        self.variance_ = torch.nn.Linear(2000, embedding_size)

        self.classify = torch.nn.LogSoftmax(dim=1)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 2000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2000, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, input_dim),
            torch.nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.encoder(x)
        q_mean = self.mean_(embedded)
        q_var = self.variance_(embedded)
        return q_mean, q_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_mean, q_var = self.encode(x)
        reconstruction = self.decode(q_mean)
        return q_mean, reconstruction

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        # training loop
        for _ in range(n_epochs):
            self.train()
            for batch in trainloader:
                # load batch on device
                batch_data = batch.to(device)
                q_mean, reconstruction = self.forward(batch_data)
                loss = loss_fn(reconstruction, batch_data) / batch_data.size(0)
                # reset gradients from last iteration
                optimizer.zero_grad()
                # calculate gradients and reset the computation graph
                loss.backward()
                # update the internal params (weights, etc.)
                optimizer.step()


class _VaDE_Module(torch.nn.Module):
    def __init__(self, autoencoder, n_clusters, embedding_size, pi, mean, var, device):
        super(_VaDE_Module, self).__init__()

        self.pi = torch.nn.Parameter(torch.ones(n_clusters) / n_clusters, requires_grad=True)
        self.p_mean = torch.nn.Parameter(torch.randn(n_clusters, embedding_size),
                                         requires_grad=True)  # if not initialized then use torch.randn
        self.p_var = torch.nn.Parameter(torch.ones(n_clusters, embedding_size), requires_grad=True)

        self.pi.data = torch.from_numpy(pi).float().to(device)
        self.p_mean.data = torch.from_numpy(mean).float().to(device)
        self.p_var.data = torch.log(torch.from_numpy(var)).float().to(device)

        self.normalize_prob = torch.nn.Softmax(dim=0)
        self.autoencoder = autoencoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        q_mean, q_var = self.autoencoder.encode(x)
        return q_mean, q_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_mean, q_var = self.autoencoder.encode(x)
        z = _sampling(q_mean, q_var)
        reconstruction = self.autoencoder.decode(z)
        return z, q_mean, q_var, reconstruction

    def vae_loss(self, batch_data, reconstruction, q_mean, q_var, loss_fn) -> torch.Tensor:
        z = _sampling(q_mean, q_var)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        loss = _compute_loss(pi_normalized, self.p_mean, self.p_var, q_mean, q_var, batch_data, p_c_z, reconstruction,
                             loss_fn)
        return loss

    def predict(self, q_mean, q_var) -> torch.Tensor:
        z = _sampling(q_mean, q_var)
        pi_normalized = self.normalize_prob(self.pi)
        p_c_z = _get_gamma(pi_normalized, self.p_mean, self.p_var, z)
        pred = torch.argmax(p_c_z, dim=1)
        return pred

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        # lr_decrease = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # training loop
        for _ in range(n_epochs):
            self.train()
            for batch in trainloader:
                # load batch on device
                batch_data = batch.to(device)

                z, q_mean, q_var, reconstruction = self.forward(batch_data)
                loss = self.vae_loss(batch_data, reconstruction, q_mean, q_var, loss_fn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class VaDE():
    def __init__(self, n_clusters, batch_size=256, learning_rate=1e-3, pretrain_epochs=50,
                 vade_epochs=300, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.BCELoss(reduction='sum'),
                 autoencoder=None, embedding_size=10):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.vade_epochs = vade_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size

    def fit(self, X):
        gmm_labels, gmm_means, gmm_covariances, vade_labels, vade_centers, vade_covariances, autoencoder = _vade(X,
                                                                                                                 self.n_clusters,
                                                                                                                 self.batch_size,
                                                                                                                 self.learning_rate,
                                                                                                                 self.pretrain_epochs,
                                                                                                                 self.vade_epochs,
                                                                                                                 self.optimizer_class,
                                                                                                                 self.loss_fn,
                                                                                                                 self.autoencoder,
                                                                                                                 self.embedding_size)
        self.labels_ = gmm_labels
        self.cluster_centers_ = gmm_means
        self.covariances_ = gmm_covariances
        self.vade_labels_ = vade_labels
        self.vade_cluster_centers_ = vade_centers
        self.vade_covariances_ = vade_covariances
        self.autoencoder = autoencoder

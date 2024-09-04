"""
@authors:
Lukas Miklautz
"""

import torch
from sklearn.cluster import KMeans
import numpy as np
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep._utils import int_to_one_hot, squared_euclidean_distance, encode_batchwise, detect_device
from clustpy.deep._data_utils import get_dataloader, get_train_and_test_dataloader
from clustpy.deep._train_utils import get_trained_network
from clustpy.alternative import NrKmeans
from sklearn.utils import check_random_state
from sklearn.metrics import normalized_mutual_info_score
from clustpy.utils.plots import plot_scatter_matrix
from clustpy.alternative.nrkmeans import _get_total_cost_function
import tqdm


class _ENRC_Module(torch.nn.Module):
    """
    The ENRC torch.nn.Module.

    Parameters
    ----------
    centers : list
        list containing the cluster centers for each clustering
    P : list
        list containing projections for each clustering
    V : np.ndarray
        orthogonal rotation matrix
    beta_init_value : float
        initial values of beta weights. Is ignored if beta_weights is not None (default: 0.9)
    clustering_loss_weight : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    center_lr : float
        weight for updating the centers via mini-batch k-means. Has to be set between 0 and 1. If set to 1.0 than only the mini-batch centroid will be used,
        neglecting the past state and if set to 0 then no update is happening (default: 0.5)
    rotate_centers : bool
        if True then centers are multiplied with V before they are used, because ENRC assumes that the centers lie already in the V-rotated space (default: False)
    beta_weights : np.ndarray
        initial beta weights for the softmax (optional). If not None, then beta_init_value will be ignored (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    lonely_centers_count : list
        list of np.ndarrays, count indicating how often a center in a clustering has not received any updates, because no points were assigned to it.
        The lonely_centers_count of a center is reset if it has been reinitialized.
    mask_sum : list
        list of torch.tensors, contains the average number of points assigned to each cluster in each clustering over the training.
    reinit_threshold : int
        threshold that indicates when a cluster should be reinitialized. Starts with 1 and increases during training with int(np.sqrt(i+1)), where i is the number of mini-batch iterations.
    augmentation_invariance : bool (default: False)

    Raises
    ----------
    ValueError : if center_lr is not in [0,1]
    """

    def __init__(self, centers: list, P: list, V: np.ndarray, beta_init_value: float = 0.9,
                 clustering_loss_weight: float = 1.0, ssl_loss_weight: float = 1.0,
                 center_lr: float = 0.5, rotate_centers: bool = False, beta_weights: np.ndarray = None,
                 augmentation_invariance: bool = False):
        super().__init__()

        self.P = P
        self.m = [len(P_i) for P_i in self.P]
        if beta_weights is None:
            beta_weights = beta_weights_init(self.P, n_dims=centers[0].shape[1], high_value=beta_init_value)
        else:
            beta_weights = torch.tensor(beta_weights).float()
        self.beta_weights = torch.nn.Parameter(beta_weights, requires_grad=True)
        self.V = torch.nn.Parameter(torch.tensor(V, dtype=torch.float), requires_grad=True)
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight

        # Center specific initializations
        if rotate_centers:
            centers = [np.matmul(centers_sub, V) for centers_sub in centers]
        self.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in centers]
        if not (0 <= center_lr <= 1):
            raise ValueError(f"center_lr={center_lr}, but has to be in [0,1].")
        self.center_lr = center_lr
        self.lonely_centers_count = []
        self.mask_sum = []
        for centers_i in self.centers:
            self.lonely_centers_count.append(np.zeros((centers_i.shape[0], 1)).astype(int))
            self.mask_sum.append(torch.zeros((centers_i.shape[0], 1)))
        self.reinit_threshold = 1
        self.augmentation_invariance = augmentation_invariance

    def to_device(self, device: torch.device) -> '_ENRC_Module':
        """
        Loads all ENRC parameters to device that are needed during the training and prediction (including the learnable parameters).
        This function is preferred over the to(device) function.

        Parameters
        ----------
        device : torch.device
            device to be trained on

        Returns
        -------
        self : _ENRC_Module
            this instance of the ENRC_Module
        """
        self.to(device)
        self.centers = [c_i.to(device) for c_i in self.centers]
        self.mask_sum = [i.to(device) for i in self.mask_sum]
        return self

    def subspace_betas(self) -> torch.Tensor:
        """
        Returns a len(P) x d matrix with softmax weights, where d is the number of dimensions of the embedded space, indicating
        which dimensions belongs to which clustering.

        Returns
        -------
        self : torch.Tensor
            the dimension assignments
        """
        dimension_assignments = torch.nn.functional.softmax(self.beta_weights, dim=0)
        return dimension_assignments

    def get_P(self) -> list:
        """
        Converts the soft beta weights back to hard assignments P and returns them as a list.

        Returns
        -------
        P : list
            list containing indices for projections for each clustering
        """
        P = _get_P(betas=self.subspace_betas().detach().cpu(), centers=self.centers)
        return P

    def rotate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Rotate the embedded data ponint z using the orthogonal rotation matrix V.

        Parameters
        ----------
        z : torch.Tensor
            embedded data point, can also be a mini-batch of points

        Returns
        -------
        z_rot : torch.Tensor
            the rotated embedded data point
        """
        z_rot = _rotate(z, self.V)
        return z_rot

    def rotate_back(self, z_rot: torch.Tensor) -> torch.Tensor:
        """
        Rotate a rotated embedded data point back to its original state.

        Parameters
        ----------
        z_rot : torch.Tensor
            rotated and embedded data point, can also be a mini-batch of points

        Returns
        -------
        z : torch.Tensor
            the back-rotated embedded data point
        """
        z = _rotate_back(z_rot, self.V)
        return z

    def rotation_loss(self) -> torch.Tensor:
        """
        Computes how much the rotation matrix self.V diverges from an orthogonal matrix by calculating |V^T V - I|.
        For an orthogonal matrix this difference is 0, as V^T V=I.

        Returns
        -------
        rotation_loss : torch.Tensor
            the average absolute difference between V^T times V - the identity matrix I.
        """
        ident = torch.matmul(self.V.t(), self.V).detach().cpu()
        rotation_loss = (ident - torch.eye(n=ident.shape[0])).abs().mean()
        return rotation_loss

    def update_center(self, data: torch.Tensor, one_hot_mask: torch.Tensor, subspace_id: int) -> None:
        """
        Inplace update of centers of a clusterings in subspace=subspace_id in a mini-batch fashion.

        Parameters
        ----------
        data : torch.Tensor
            data points, can also be a mini-batch of points
        one_hot_mask : torch.Tensor
            one hot encoded matrix of cluster assignments
        subspace_id : int
            integer which indicates which subspace the cluster to be updated are in

        Raises
        ----------
        ValueError: If None values are encountered.
        """
        if self.centers[subspace_id].shape[0] == 1:
            # Shared space update with only one cluster
            self.centers[subspace_id] = self.centers[subspace_id] * 0.5 + data.mean(0).unsqueeze(0) * 0.5
        else:

            batch_cluster_sums = (data.unsqueeze(1) * one_hot_mask.unsqueeze(2)).sum(0)
            mask_sum = one_hot_mask.sum(0).unsqueeze(1)
            if (mask_sum == 0).sum().int().item() != 0:
                idx = (mask_sum == 0).nonzero()[:, 0].detach().cpu()
                self.lonely_centers_count[subspace_id][idx] += 1

            # In case mask sum is zero batch cluster sum is also zero so we can add a small constant to mask sum and center_lr
            # Avoid division by a small number
            mask_sum += 1e-8
            # Use weighted average
            nonzero_mask = (mask_sum.squeeze(1) != 0)
            self.mask_sum[subspace_id][nonzero_mask] = self.center_lr * mask_sum[nonzero_mask] + (1 - self.center_lr) * \
                                                       self.mask_sum[subspace_id][nonzero_mask]

            per_center_lr = 1.0 / (1 + self.mask_sum[subspace_id][nonzero_mask])
            self.centers[subspace_id] = (1.0 - per_center_lr) * self.centers[subspace_id][
                nonzero_mask] + per_center_lr * batch_cluster_sums[nonzero_mask] / mask_sum[nonzero_mask]
            if torch.isnan(self.centers[subspace_id]).sum() > 0:
                raise ValueError(
                    f"Found nan values\n self.centers[subspace_id]: {self.centers[subspace_id]}\n per_center_lr: {per_center_lr}\n self.mask_sum[subspace_id]: {self.mask_sum[subspace_id]}\n ")

    def update_centers(self, z_rot: torch.Tensor, assignment_matrix_dict: dict) -> None:
        """
        Inplace update of all centers in all clusterings in a mini-batch fashion.

        Parameters
        ----------
        z_rot : torch.Tensor
            rotated data point, can also be a mini-batch of points
        assignment_matrix_dict : dict
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
        """
        for subspace_i in range(len(self.centers)):
            self.update_center(z_rot.detach(),
                               assignment_matrix_dict[subspace_i],
                               subspace_id=subspace_i)

    def forward(self, z: torch.Tensor, assignment_matrix_dict: dict = None) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, dict):
        """
        Calculates the k-means loss and cluster assignments for each clustering.

        Parameters
        ----------
        z : torch.Tensor
            embedded input data point, can also be a mini-batch of embedded points
        assignment_matrix_dict : dict
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments (default: None)

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, dict)
            averaged sum of all k-means losses for each clustering,
            the rotated embedded point,
            the back rotated embedded point,
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
        """
        z_rot = self.rotate(z)
        z_rot_back = self.rotate_back(z_rot)

        subspace_betas = self.subspace_betas()
        subspace_losses = 0

        if assignment_matrix_dict is None:
            assignment_matrix_dict = {}
            overwrite_assignments = True
        else:
            overwrite_assignments = False

        for i, centers_i in enumerate(self.centers):
            weighted_squared_diff = squared_euclidean_distance(z_rot, centers_i.detach(), weights=subspace_betas[i, :])
            weighted_squared_diff /= z_rot.shape[0]

            if overwrite_assignments:
                assignments = weighted_squared_diff.detach().argmin(1)
                one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                assignment_matrix_dict[i] = one_hot_mask
            else:
                one_hot_mask = assignment_matrix_dict[i]
            weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask
            subspace_losses += weighted_squared_diff_masked.sum()

        subspace_losses = subspace_losses / subspace_betas.shape[0]
        return subspace_losses, z_rot, z_rot_back, assignment_matrix_dict

    def predict(self, z: torch.Tensor, use_P: bool = False) -> np.ndarray:
        """
        Predicts the labels for each clustering of an input z.

        Parameters
        ----------
        z : torch.Tensor
            embedded input data point, can also be a mini-batch of embedded points
        use_P: bool
            if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: False)

        Returns
        -------
        predicted_labels : np.ndarray
            n x c matrix, where n is the number of data points in z and c is the number of clusterings.
        """
        predicted_labels = enrc_predict(z=z, V=self.V, centers=self.centers, subspace_betas=self.subspace_betas(),
                                        use_P=use_P)
        return predicted_labels

    def predict_batchwise(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                          device: torch.device = torch.device("cpu"), use_P: bool = False) -> np.ndarray:
        """
        Predicts the labels for each clustering of a dataloader in a mini-batch manner.

        Parameters
        ----------
        model : torch.nn.Module
            the input model for encoding the data
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for prediction
        device : torch.device
            device to be predicted on (default: torch.device('cpu'))
        use_P: bool
            if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: False)

        Returns
        -------
        predicted_labels : np.ndarray
            n x c matrix, where n is the number of data points in z and c is the number of clusterings.
        """
        predicted_labels = enrc_predict_batchwise(V=self.V, centers=self.centers, model=model, dataloader=dataloader,
                                                  subspace_betas=self.subspace_betas(), device=device, use_P=use_P)
        return predicted_labels

    def recluster(self, dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, optimizer_params: dict,
                  optimizer_class: torch.optim.Optimizer = None,
                  device: torch.device = torch.device('cpu'), rounds: int = 1, reclustering_strategy="auto",
                  init_kwargs: dict = None) -> None:
        """
        Recluster ENRC inplace using NrKMeans or SGD (depending on the data set size, see init='auto' for details).
        Can lead to improved and more stable performance.
        Updates self.P, self.beta_weights, self.V and self.centers.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for prediction
        model : torch.nn.Module
            the input model for encoding the data
        optimizer_params: dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate
        optimizer_class : torch.optim.Optimizer
            optimizer for training. If None then torch.optim.Adam will be used (default: None)
        device : torch.device
            device to be predicted on (default: torch.device('cpu'))
        rounds : int
            number of repetitions of the reclustering procedure (default: 1)
        reclustering_strategy : string
            choose which initialization strategy should be used. Has to be one of 'nrkmeans', 'random' or 'sgd' (default: 'nrkmeans')
        init_kwargs : dict
            additional parameters that are used if reclustering_strategy is a callable (optional) (default: None)
        """

        # Extract parameters
        V = self.V.detach().cpu().numpy()
        n_clusters = [c.shape[0] for c in self.centers]

        # Encode data
        embedded_data = encode_batchwise(dataloader, model)
        embedded_rot = np.matmul(embedded_data, V)

        # Apply reclustering in the rotated space, because V does not have to be orthogonal, so it could learn a mapping that is not recoverable by nrkmeans.
        centers_reclustered, P, new_V, beta_weights = enrc_init(data=embedded_rot, n_clusters=n_clusters, rounds=rounds,
                                                                max_iter=300, optimizer_params=optimizer_params,
                                                                optimizer_class=optimizer_class,
                                                                init=reclustering_strategy, debug=False,
                                                                init_kwargs=init_kwargs,
                                                                batch_size=dataloader.batch_size,
                                                                )

        # Update V, because we applied the reclustering in the rotated space
        new_V = np.matmul(V, new_V)

        # Assign reclustered parameters
        self.P = P
        self.m = [len(P_i) for P_i in self.P]
        self.beta_weights = torch.nn.Parameter(torch.from_numpy(beta_weights).float(), requires_grad=True)
        self.V = torch.nn.Parameter(torch.tensor(new_V, dtype=torch.float), requires_grad=True)
        self.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in centers_reclustered]
        self.to_device(device)

    def fit(self, trainloader: torch.utils.data.DataLoader, evalloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, max_epochs: int, model: torch.nn.Module,
            batch_size: int, ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
            device: torch.device = torch.device("cpu"), debug: bool = True,
            scheduler: torch.optim.lr_scheduler = None, fix_rec_error: bool = False,
            tolerance_threshold: float = None, data: torch.Tensor | np.ndarray = None) -> (
            torch.nn.Module, '_ENRC_Module'):
        """
        Trains ENRC and the neural network in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        evalloader : torch.utils.data.DataLoader
            Evalloader is used for checking label change
        optimizer : torch.optim.Optimizer
            parameterized optimizer to be used
        max_epochs : int
            maximum number of epochs for training
        model : torch.nn.Module
            The underlying neural network
        batch_size: int
            batch size for dataloader
        ssl_loss_fn : torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        debug : bool
            if True than training errors will be printed (default: True)
        scheduler : torch.optim.lr_scheduler
            parameterized learning rate scheduler that should be used (default: None)
        fix_rec_error : bool
            if set to True than reconstruction loss is weighted proportionally to the cluster loss. Only used for init='sgd' (default: False)
        tolerance_threshold : float
            tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
            for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
            will train as long as max_epochs (default: None)
        data : torch.Tensor | np.ndarray
            dataset to be used for training (default: None)
        Returns
        -------
        tuple : (torch.nn.Module, _ENRC_Module)
            trained neural network,
            trained enrc module
        """
        # Deactivate Batchnorm and dropout
        model.eval()
        model.to(device)
        self.to_device(device)

        if trainloader is None and data is not None:
            trainloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif trainloader is None and data is None:
            raise ValueError("trainloader and data cannot be both None.")
        if evalloader is None and data is not None:
            # Evalloader is used for checking label change. Only difference to the trainloader here is that shuffle=False.
            evalloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)

        if fix_rec_error:
            if debug: print("Calculate initial reconstruction error")
            _, _, init_ssl_loss = enrc_encode_decode_batchwise_with_loss(V=self.V, centers=self.centers, model=model,
                                                                         dataloader=evalloader, device=device,
                                                                         ssl_loss_fn=ssl_loss_fn)
            # For numerical stability we add a small number
            init_ssl_loss += 1e-8
            if debug: print("Initial reconstruction error is ", init_ssl_loss)
        i = 0
        labels_old = None
        tbar = tqdm.trange(max_epochs, desc="ENRC training")
        for _ in tbar:
            total_loss = 0
            for batch in trainloader:
                if self.augmentation_invariance:
                    batch_data_aug = batch[1].to(device)
                    batch_data = batch[2].to(device)
                else:
                    batch_data = batch[1].to(device)

                z = model.encode(batch_data)
                subspace_loss, z_rot, z_rot_back, assignment_matrix_dict = self(z)

                reconstruction = model.decode(z_rot_back)
                ssl_loss = ssl_loss_fn(reconstruction, batch_data)

                if self.augmentation_invariance:
                    z_aug = model.encode(batch_data_aug)
                    # reuse assignments
                    subspace_loss_aug, _, z_rot_back_aug, _ = self(z_aug, assignment_matrix_dict=assignment_matrix_dict)
                    reconstruction_aug = model.decode(z_rot_back_aug)
                    ssl_loss_aug = ssl_loss_fn(reconstruction_aug, batch_data_aug)
                    ssl_loss = (ssl_loss + ssl_loss_aug) / 2
                    subspace_loss = (subspace_loss + subspace_loss_aug) / 2

                if fix_rec_error:
                    rec_weight = ssl_loss.item() / init_ssl_loss + subspace_loss.item() / ssl_loss.item()
                    if rec_weight < 1:
                        rec_weight = 1.0
                    ssl_loss *= rec_weight

                summed_loss = self.clustering_loss_weight * subspace_loss + self.ssl_loss_weight * ssl_loss
                total_loss += summed_loss.item()
                optimizer.zero_grad()
                summed_loss.backward()
                optimizer.step()

                # Update Assignments and Centroids on GPU
                with torch.no_grad():
                    self.update_centers(z_rot, assignment_matrix_dict)
                # Check if clusters have to be reinitialized
                for subspace_i in range(len(self.centers)):
                    reinit_centers(enrc=self, subspace_id=subspace_i, dataloader=trainloader, model=model,
                                   n_samples=512, kmeans_steps=10, debug=debug)

                # Increase reinit_threshold over time
                self.reinit_threshold = int(np.sqrt(i + 1))

                i += 1
            postfix_str = {"Loss": total_loss}
            with torch.no_grad():
                # Rotation loss is calculated to check if its deviation from an orthogonal matrix
                rotation_loss = self.rotation_loss()
                postfix_str["rotation_loss"] = rotation_loss.item()
            tbar.set_postfix(postfix_str)

            if scheduler is not None:
                scheduler.step()

            if tolerance_threshold is not None and tolerance_threshold > 0:
                # Check if labels have changed
                labels_new = self.predict_batchwise(model=model, dataloader=evalloader, device=device, use_P=True)
                if _are_labels_equal(labels_new=labels_new, labels_old=labels_old, threshold=tolerance_threshold):
                    # training has converged
                    if debug:
                        print("Clustering has converged")
                    break
                else:
                    labels_old = labels_new.copy()

        # Extract P and m
        self.P = self.get_P()
        self.m = [len(P_i) for P_i in self.P]
        return model, self


"""
===================== Helper Functions =====================
"""


class _IdentityAutoencoder(torch.nn.Module):
    """
    Convenience class to avoid reimplementation of the remaining ENRC pipeline for the initialization.
    Encoder and decoder are here just identity functions implemented via lambda x:x.

    Attributes
    ----------
    encoder : function
        the encoder part
    decoder : function
        the decoder part
    """

    def __init__(self):
        super(_IdentityAutoencoder, self).__init__()

        self.encoder = lambda x: x
        self.decoder = lambda x: x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        encoded : torch.Tensor
            the encoeded data point
        """
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        decoded : torch.Tensor
            returns the reconstruction of embedded
        """
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies both the encode and decode function.
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of embedded points

        Returns
        -------
        reconstruction : torch.Tensor
            returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction


def _get_P(betas: torch.Tensor, centers: list, shared_space_variation: float = 0.05) -> float:
    """
    Converts the softmax betas back to hard assignments P and returns them as a list.

    Parameters
    ----------
    betas : torch.Tensor
        c x d soft assignment weights matrix for c clusterings and d dimensions.
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    shared_space_variation : float
        specifies how much beta in the shared space is allowed to diverge from the uniform distribution. Only needed if a shared space (space with one cluster) exists (default: 0.05)

    Returns
    ----------
    P : list
        list containing indices for projections for each clustering
    """
    # Check if a shared space with a single cluster center exist
    shared_space_idx = [i for i, centers_i in enumerate(centers) if centers_i.shape[0] == 1]
    if shared_space_idx:
        # Specifies how much beta in the shared space is allowed to diverge from the uniform distribution
        shared_space_idx = shared_space_idx[0]
        equal_threshold = 1.0 / betas.shape[0]
        # Increase Weight of shared space dimensions that are close to the uniform distribution
        equal_threshold -= shared_space_variation
        betas[shared_space_idx][betas[shared_space_idx] > equal_threshold] += 1

    # Select highest assigned dimensions to P
    max_assigned_dims = betas.argmax(0)
    P = [[] for _ in range(betas.shape[0])]
    for dim_i, cluster_subspace_id in enumerate(max_assigned_dims):
        P[cluster_subspace_id].append(dim_i)
    return P


def _rotate(z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Rotate the embedded data ponint z using the orthogonal rotation matrix V.

    Parameters
    ----------
    V : torch.Tensor
        orthogonal rotation matrix
    z : torch.Tensor
        embedded data point, can also be a mini-batch of points
    
    Returns
    -------
    z_rot : torch.Tensor
        the rotated embedded data point
    """
    return torch.matmul(z, V)


def _rotate_back(z_rot: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Rotate a rotated embedded data point back to its original state.

    Parameters
    ----------
    z_rot : torch.Tensor
        rotated and embedded data point, can also be a mini-batch of points
    V : torch.Tensor
        orthogonal rotation matrix
    
    Returns
    -------
    z : torch.Tensor
        the back-rotated embedded data point
    """
    return torch.matmul(z_rot, V.t())


def enrc_predict(z: torch.Tensor, V: torch.Tensor, centers: list, subspace_betas: torch.Tensor,
                 use_P: bool = False) -> np.ndarray:
    """
    Predicts the labels for each clustering of an input z.

    Parameters
    ----------
    z : torch.Tensor
        embedded input data point, can also be a mini-batch of embedded points
    V : torch.tensor
        orthogonal rotation matrix
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    subspace_betas : torch.Tensor
        weights for each dimension per clustering. Calculated via softmax(beta_weights).
    use_P: bool
        if True then P will be used to hard select the dimensions for each clustering, else the soft subspace_beta weights are used (default: False)

    Returns
    -------
    predicted_labels : np.ndarray
        n x c matrix, where n is the number of data points in z and c is the number of clusterings.
    """
    z_rot = _rotate(z, V)
    if use_P:
        P = _get_P(betas=subspace_betas.detach(), centers=centers)
    labels = []
    for i, centers_i in enumerate(centers):
        if use_P:
            weighted_squared_diff = squared_euclidean_distance(z_rot[:, P[i]], centers_i[:, P[i]])
        else:
            weighted_squared_diff = squared_euclidean_distance(z_rot, centers_i, weights=subspace_betas[i, :])
        labels_sub = weighted_squared_diff.argmin(1)
        labels_sub = labels_sub.detach().cpu().numpy().astype(np.int32)
        labels.append(labels_sub)
    return np.stack(labels).transpose()


def enrc_predict_batchwise(V: torch.Tensor, centers: list, subspace_betas: torch.Tensor, model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader, device: torch.device = torch.device("cpu"),
                           use_P: bool = False) -> np.ndarray:
    """
    Predicts the labels for each clustering of a dataloader in a mini-batch manner.

    Parameters
    ----------
    V : torch.Tensor
        orthogonal rotation matrix
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    subspace_betas : torch.Tensor
        weights for each dimension per clustering. Calculated via softmax(beta_weights).
    model : torch.nn.Module
        the input model for encoding the data
    dataloader : torch.utils.data.DataLoader
        dataloader to be used for prediction
    device : torch.device
        device to be predicted on (default: torch.device('cpu'))
    use_P: bool
        if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: False)
    
    Returns
    -------
    predicted_labels : np.ndarray
        n x c matrix, where n is the number of data points in z and c is the number of clusterings.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[1].to(device)
            z = model.encode(batch_data)
            pred_i = enrc_predict(z=z, V=V, centers=centers, subspace_betas=subspace_betas, use_P=use_P)
            predictions.append(pred_i)
    return np.concatenate(predictions)


def enrc_encode_decode_batchwise_with_loss(V: torch.Tensor, centers: list, model: torch.nn.Module,
                                           dataloader: torch.utils.data.DataLoader,
                                           device: torch.device = torch.device("cpu"),
                                           ssl_loss_fn: torch.nn.modules.loss._Loss = None) -> np.ndarray:
    """
    Encode and Decode input data of a dataloader in a mini-batch manner with ENRC.

    Parameters
    ----------
    V : torch.Tensor
        orthogonal rotation matrix
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    model : torch.nn.Module
        the input model for encoding the data
    dataloader : torch.utils.data.DataLoader
        dataloader to be used for prediction
    device : torch.device
        device to be predicted on (default: torch.device('cpu'))
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: None)

    Returns
    -------
    enrc_encoding : np.ndarray
        n x d matrix, where n is the number of data points and d is the number of dimensions of z.
    enrc_decoding : np.ndarray
        n x D matrix, where n is the number of data points and D is the data dimensionality.
    reconstruction_error : flaot
        reconstruction error (will be None if ssl_loss_fn is not specified)
    """
    model.eval()
    reconstructions = []
    embeddings = []
    if ssl_loss_fn is None:
        loss = None
    else:
        loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[1].to(device)
            z = model.encode(batch_data)
            z_rot = _rotate(z=z, V=V)
            embeddings.append(z_rot.detach().cpu())
            z_rot_back = _rotate_back(z_rot=z_rot, V=V)
            reconstruction = model.decode(z_rot_back)
            if ssl_loss_fn is not None:
                loss += ssl_loss_fn(reconstruction, batch_data).item()
            reconstructions.append(reconstruction.detach().cpu())
    if ssl_loss_fn is not None:
        loss /= len(dataloader)
    embeddings = torch.cat(embeddings).numpy()
    reconstructions = torch.cat(reconstructions).numpy()
    return embeddings, reconstructions, loss


"""
===================== Initialization Strategies =====================
"""


def available_init_strategies() -> list:
    """
    Returns a list of strings of available initialization strategies for ENRC and ACeDeC.
    At the moment following strategies are supported: nrkmeans, random, sgd, auto
    """
    return ['nrkmeans', 'random', 'sgd', 'auto', 'subkmeans', 'acedec']


def optimal_beta(kmeans_loss: torch.Tensor, other_losses_mean_sum: torch.Tensor) -> torch.Tensor:
    """
    Calculate optimal values for the beta weight for each dimension.
    
    Parameters
    ----------
    kmeans_loss: torch.Tensor
        a 1 x d vector of the kmeans losses per dimension.
    other_losses_mean_sum: torch.Tensor
        a 1 x d vector of the kmeans losses of all other clusterings except the one in 'kmeans_loss'.
    
    Returns
    -------
    optimal_beta_weights: torch.Tensor
        a 1 x d vector containing the optimal weights for the softmax to indicate which dimensions are important for each clustering.
        Calculated via -torch.log(kmeans_loss/other_losses_mean_sum)
    """
    return -torch.log(kmeans_loss / other_losses_mean_sum)


def calculate_optimal_beta_weights_special_case(data: torch.Tensor, centers: list, V: torch.Tensor,
                                                batch_size: int = 32) -> torch.Tensor:
    """
    The beta weights have a closed form solution if we have two subspaces, so the optimal values given the data, centers and V can be computed.
    See supplement of Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Boehm, Claudia Plant: Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826-2832
    here: https://gitlab.cs.univie.ac.at/lukas/acedec_public/-/blob/master/supplement.pdf

    Parameters
    ----------
    data : torch.Tensor
        input data
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    V : torch.Tensor
        orthogonal rotation matrix
    batch_size : int
        size of the data batches (default: 32)

    Returns
    -------
    optimal_beta_weights: torch.Tensor
        a c x d vector containing the optimal weights for the softmax to indicate which dimensions d are important for each clustering c.
    """
    dataloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    device = V.device
    with torch.no_grad():
        # calculate kmeans losses for each clustering
        km_losses = [[] for _ in centers]
        for batch in dataloader:
            batch = batch[1].to(device)
            z_rot = torch.matmul(batch, V)
            for i, centers_i in enumerate(centers):
                centers_i = centers_i.to(device)
                weighted_squared_diff = squared_euclidean_distance(z_rot.unsqueeze(1), centers_i.unsqueeze(1))
                assignments = weighted_squared_diff.detach().sum(2).argmin(1)
                if len(set(assignments.tolist())) > 1:
                    one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                    weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask.unsqueeze(2)
                else:
                    weighted_squared_diff_masked = weighted_squared_diff

                km_losses[i].append(weighted_squared_diff_masked.detach().cpu())
                centers_i = centers_i.cpu()
        for i, km_loss in enumerate(km_losses):
            # Sum over samples and centers
            km_losses[i] = torch.cat(km_loss, 0).sum(0).sum(0)
        # calculate beta_weights for each dimension and clustering based on kmeans losses
        best_weights = []
        best_weights.append(optimal_beta(km_losses[0], km_losses[1]))
        best_weights.append(optimal_beta(km_losses[1], km_losses[0]))
        best_weights = torch.stack(best_weights)
    return best_weights


def beta_weights_init(P: list, n_dims: int, high_value: float = 0.9) -> torch.Tensor:
    """
    Initializes parameters of the softmax such that betas will be set to high_value in dimensions which form a cluster subspace according to P
    and set to (1 - high_value)/(len(P) - 1) for the other clusterings.
    
    Parameters
    ----------
    P : list
        list containing projections for each subspace
    n_dims : int
        dimensionality of the embedded data
    high_value : float
        value that should be initially used to indicate strength of assignment of a specific dimension to the clustering (default: 0.9)
    
    Returns
    ----------
    beta_weights : torch.Tensor
        initialized weights that are input in the softmax to get the betas.
    """
    weight_high = 1.0
    n_sub_clusterings = len(P)
    beta_hard = np.zeros((n_sub_clusterings, n_dims), dtype=np.float32)
    for sub_i, p in enumerate(P):
        for dim in p:
            beta_hard[sub_i, dim] = 1.0
    low_value = 1.0 - high_value
    weight_high_exp = np.exp(weight_high)
    # Because high_value = weight_high/(weight_high +low_classes*weight_low)
    n_low_classes = len(P) - 1
    weight_low_exp = weight_high_exp * (1.0 - high_value) / (high_value * n_low_classes)
    weight_low = np.log(weight_low_exp)
    beta_soft_weights = beta_hard * (weight_high - weight_low) + weight_low
    return torch.tensor(beta_soft_weights, dtype=torch.float32)


def calculate_beta_weight(data: torch.Tensor, centers: list, V: torch.Tensor, P: list,
                          high_beta_value: float = 0.9) -> torch.Tensor:
    """
    The beta weights have a closed form solution if we have two subspaces, so the optimal values given the data, centers and V can be computed.
    See supplement of Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Boehm, Claudia Plant: Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826-2832
    here: https://gitlab.cs.univie.ac.at/lukas/acedec_public/-/blob/master/supplement.pdf
    For number of subspaces > 2, we calculate the beta weight assuming that an assigned subspace should have a weight of 0.9.
    
    Parameters
    ----------
    data : torch.Tensor
        input data
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    V : torch.Tensor
        orthogonal rotation matrix
    P : list
        list containing projections for each subspace
    high_beta_value : float
        value that should be initially used to indicate strength of assignment of a specific dimension to the clustering (default: 0.9)

    Returns
    -------
    beta_weights: torch.Tensor
        a c x d vector containing the weights for the softmax to indicate which dimensions d are important for each clustering c.

    Raises
    -------
    ValueError: If number of clusterings is smaller than 2
    """
    n_clusterings = len(centers)
    if n_clusterings == 2:
        beta_weights = calculate_optimal_beta_weights_special_case(data=data, centers=centers, V=V)
    elif n_clusterings > 2:
        beta_weights = beta_weights_init(P=P, n_dims=data.shape[1], high_value=high_beta_value)
    else:
        raise ValueError(f"Number of clusterings is {n_clusterings}, but should be >= 2")
    return beta_weights


def nrkmeans_init(data: np.ndarray, n_clusters: list, rounds: int = 10, max_iter: int = 100, input_centers: list = None,
                  P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None, debug=True) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on the NrKmeans Algorithm. This strategy is preferred for small data sets, but the orthogonality
    constraint on V and subsequently for the clustered subspaces can be sometimes to limiting in practice, e.g., if clusterings are
    not perfectly non-redundant.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    rounds : int
        number of repetitions of the NrKmeans algorithm (default: 10)
    max_iter : int
        maximum number of iterations of NrKmeans (default: 100)
    input_centers : list
        list of np.ndarray, optional parameter if initial cluster centers want to be set (optional) (default: None)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace
        list containing projections for each subspace
        orthogonal rotation matrix
        weights for softmax function to get beta values.
    """
    best = None
    lowest = np.inf
    if max(n_clusters) >= data.shape[1]:
        mdl_for_noisespace = True
        if debug:
            print("mdl_for_noisespace=True, because number of clusters is larger then data dimensionality")
    else:
        mdl_for_noisespace = False
    for i in range(rounds):
        nrkmeans = NrKmeans(n_clusters=n_clusters, cluster_centers=input_centers, P=P, V=V, max_iter=max_iter,
                            random_state=random_state, mdl_for_noisespace=mdl_for_noisespace)
        nrkmeans.fit(X=data)
        centers_i, P_i, V_i, scatter_matrices_i = nrkmeans.cluster_centers, nrkmeans.P, nrkmeans.V, nrkmeans.scatter_matrices_
        if len(P_i) != len(n_clusters):
            if debug:
                print(
                    f"WARNING: Lost Subspace. Found only {len(P_i)} subspaces for {len(n_clusters)} clusterings. Try to increase the size of the embedded space or the number of iterations of nrkmeans to avoid this from happening.")
        else:
            cost = _get_total_cost_function(V=V_i, P=P_i, scatter_matrices=scatter_matrices_i)
            if lowest > cost:
                best = [centers_i, P_i, V_i, ]
                lowest = cost
            if debug:
                print(f"Round {i}: Found solution with: {cost} (current best: {lowest})")

    # Best parameters
    if best is None:
        centers, P, V = centers_i, P_i, V_i
        if debug:
            print(
                f"WARNING: No result with all subspaces was found. Will return last computed result with {len(P)} subspaces.")
    else:
        centers, P, V = best
    # centers are expected to be rotated for ENRC
    centers = [np.matmul(centers_sub, V) for centers_sub in centers]
    beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(),
                                         centers=[torch.from_numpy(centers_sub).float() for centers_sub in centers],
                                         V=torch.from_numpy(V).float(),
                                         P=P)
    beta_weights = beta_weights.detach().cpu().numpy()

    return centers, P, V, beta_weights


def random_nrkmeans_init(data: np.ndarray, n_clusters: list, rounds: int = 10, input_centers: list = None,
                         P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None,
                         debug: bool = True) -> (list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on the NrKmeans Algorithm. For documentation see nrkmeans_init function.
    Same as nrkmeans_init, but max_iter is set to 1, so the results will be faster and more random.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    rounds : int
        number of repetitions of the NrKmeans algorithm (default: 10)
    input_centers : list
        list of np.ndarray, optional parameter if initial cluster centers want to be set (optional) (default: None)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace
        list containing projections for each subspace
        orthogonal rotation matrix
        weights for softmax function to get beta values.
    """
    return nrkmeans_init(data=data, n_clusters=n_clusters, rounds=rounds, max_iter=1,
                         input_centers=input_centers, P=P, V=V, random_state=random_state, debug=debug)


def _determine_sgd_init_costs(enrc: _ENRC_Module, dataloader: torch.utils.data.DataLoader,
                              ssl_loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
                              return_rot: bool = False) -> float:
    """
    Determine the initial sgd costs.

    Parameters
    ----------
    enrc : _ENRC_Module
        The ENRC module
    dataloader : torch.utils.data.DataLoader
        dataloader to be used for the calculation of the costs
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    device : torch.device
        device to be trained on
    return_rot : bool
        if True rotated data from datalaoder will be returned (default: False)

    Returns
    -------
    cost : float
        the costs
    """
    cost = 0
    rotated_data = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[1].to(device)
            subspace_loss, z_rot, batch_rot_back, _ = enrc(batch)
            rotated_data.append(z_rot.detach().cpu())
            ssl_loss = ssl_loss_fn(batch_rot_back, batch)
            cost += (subspace_loss + ssl_loss)
        cost /= len(dataloader)
    if return_rot:
        rotated_data = torch.cat(rotated_data).numpy()
        return cost.item(), rotated_data
    else:
        return cost.item()


def sgd_init(data: np.ndarray, n_clusters: list, optimizer_params: dict, batch_size: int = 128,
             optimizer_class: torch.optim.Optimizer = None, rounds: int = 2, epochs: int = 10,
             random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
             V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ENRC's parameters V and beta in isolation from the neural network using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        number of repetitions of the initialization procedure (default: 2)
    epochs : int
        number of epochs for the actual clustering procedure (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    best = None
    lowest = np.inf
    dataloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    for round_i in range(rounds):
        random_state = check_random_state(random_state)
        # start with random initialization
        init_centers, P_init, V_init, _ = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=10,
                                                               input_centers=input_centers,
                                                               P=P, V=V, debug=debug)

        # Initialize betas with uniform distribution
        enrc_module = _ENRC_Module(init_centers, P_init, V_init, beta_init_value=1.0 / len(P_init)).to_device(device)
        enrc_module.to_device(device)
        optimizer_beta_params = optimizer_params.copy()
        optimizer_beta_params["lr"] = optimizer_beta_params["lr"] * 10
        param_dict = [dict({'params': [enrc_module.V]}, **optimizer_params),
                      dict({'params': [enrc_module.beta_weights]}, **optimizer_beta_params)
                      ]
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(param_dict)
        # Training loop
        # For the initialization we increase the weight for the rec error to enforce close to orthogonal V by setting fix_rec_error=True
        enrc_module.fit(data=data,
                        trainloader=None,
                        evalloader=None,
                        optimizer=optimizer,
                        max_epochs=epochs,
                        model=_IdentityAutoencoder(),
                        ssl_loss_fn=torch.nn.MSELoss(),
                        batch_size=batch_size,
                        device=device,
                        debug=False,
                        fix_rec_error=True)

        cost = _determine_sgd_init_costs(enrc=enrc_module, dataloader=dataloader, ssl_loss_fn=torch.nn.MSELoss(),
                                         device=device)
        if lowest > cost:
            best = [enrc_module.centers, enrc_module.P, enrc_module.V, enrc_module.beta_weights]
            lowest = cost
        if debug:
            print(f"Round {round_i}: Found solution with: {cost} (current best: {lowest})")

    centers, P, V, beta_weights = best
    beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers, V=V, P=P)
    centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
    beta_weights = beta_weights.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    return centers, P, V, beta_weights


def acedec_init(data: np.ndarray, n_clusters: list, optimizer_params: dict, batch_size: int = 128,
                optimizer_class: torch.optim.Optimizer = None, rounds: int = None, epochs: int = 10,
                random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
                V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ACeDeC's parameters V and beta in isolation from the neural network using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_params: dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        not used here (default: None)
    epochs : int
        epochs is automatically set to be close to 20.000 minibatch iterations as in the ACeDeC paper. If this determined value is smaller than the passed epochs, then epochs is used (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    best = None
    lowest = np.inf
    dataloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    # only use one repeat as in ACeDeC paper
    acedec_rounds = 1
    # acedec used 20.000 minibatch iterations for initialization. Thus we use a number of epochs corresponding to that
    epochs_estimate = int(20000 / (data.shape[0] / batch_size))
    max_epochs = np.max([epochs_estimate, epochs])
    if debug: print("Start ACeDeC init")
    for round_i in range(acedec_rounds):
        random_state = check_random_state(random_state)

        # start with random initialization
        if debug: print("Start with random init")
        init_centers, P_init, V_init, _ = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=10,
                                                               input_centers=input_centers,
                                                               P=P, V=V, debug=debug)
        # Recluster with KMeans to get better centroid estimate
        data_rot = np.matmul(data, V_init)
        kmeans = KMeans(n_clusters[0], n_init=10)
        kmeans.fit(data_rot)
        # cluster and shared space centers
        init_centers = [kmeans.cluster_centers_, data_rot.mean(0).reshape(1, -1)]

        # Initialize betas with uniform distribution
        enrc_module = _ENRC_Module(init_centers, P_init, V_init, beta_init_value=1.0 / len(P_init)).to_device(device)
        enrc_module.to_device(device)

        optimizer_beta_params = optimizer_params.copy()
        optimizer_beta_params["lr"] = optimizer_beta_params["lr"] * 10
        param_dict = [dict({'params': [enrc_module.V]}, **optimizer_params),
                      dict({'params': [enrc_module.beta_weights]}, **optimizer_beta_params)
                      ]
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(param_dict)
        # Training loop
        # For the initialization we increase the weight for the rec error to enforce close to orthogonal V by setting fix_rec_error=True
        if debug: print("Start pretraining parameters with SGD")
        enrc_module.fit(data=data,
                        trainloader=None,
                        evalloader=None,
                        optimizer=optimizer,
                        max_epochs=max_epochs,
                        model=_IdentityAutoencoder(),
                        ssl_loss_fn=torch.nn.MSELoss(),
                        batch_size=batch_size,
                        device=device,
                        debug=debug,
                        fix_rec_error=True)

        cost, z_rot = _determine_sgd_init_costs(enrc=enrc_module, dataloader=dataloader, ssl_loss_fn=torch.nn.MSELoss(),
                                                device=device, return_rot=True)

        # Recluster with KMeans to get better centroid estimate
        kmeans = KMeans(n_clusters[0], n_init=10)
        kmeans.fit(z_rot)
        # cluster and shared space centers
        enrc_rotated_centers = [kmeans.cluster_centers_, z_rot.mean(0).reshape(1, -1)]
        enrc_module.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in enrc_rotated_centers]

        if lowest > cost:
            best = [enrc_module.centers, enrc_module.P, enrc_module.V, enrc_module.beta_weights]
            lowest = cost
        if debug:
            print(f"Round {round_i}: Found solution with: {cost} (current best: {lowest})")

    centers, P, V, beta_weights = best

    beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers, V=V, P=P)
    centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
    beta_weights = beta_weights.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    return centers, P, V, beta_weights


def enrc_init(data: np.ndarray, n_clusters: list, init: str = "auto", rounds: int = 10, input_centers: list = None,
              P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None, max_iter: int = 100,
              optimizer_params: dict = None, optimizer_class: torch.optim.Optimizer = None, batch_size: int = 128,
              epochs: int = 10, device: torch.device = torch.device("cpu"), debug: bool = True,
              init_kwargs: dict = None) -> (list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy for the ENRC algorithm.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    init : str
        {'nrkmeans', 'random', 'sgd', 'auto'} or callable. Initialization strategies for parameters cluster_centers, V and beta of ENRC. (default='auto')

        'nrkmeans' : Performs the NrKmeans algorithm to get initial parameters. This strategy is preferred for small data sets,
        but the orthogonality constraint on V and subsequently for the clustered subspaces can be sometimes to limiting in practice,
        e.g., if clusterings in the data are not perfectly non-redundant.

        'random' : Same as 'nrkmeans', but max_iter is set to 10, so the performance is faster, but also less optimized, thus more random.

        'sgd' : Initialization strategy based on optimizing ENRC's parameters V and beta in isolation from the neural network using a mini-batch gradient descent optimizer.
        This initialization strategy scales better to large data sets than the 'nrkmeans' option and only constraints V using the reconstruction error (torch.nn.MSELoss),
        which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the 'sgd' strategy is that it can be less stable for small data sets.

        'auto' : Selects 'sgd' init if data.shape[0] > 100,000 or data.shape[1] > 1,000. For smaller data sets 'nrkmeans' init is used.

        If a callable is passed, it should take arguments data and n_clusters (additional parameters can be provided via the dictionary init_kwargs) and return an initialization (centers, P, V and beta_weights).
    
    rounds : int
        number of repetitions of the initialization procedure (default: 10)
    input_centers : list
        list of np.ndarray, optional parameter if initial cluster centers want to be set (optional) (default: None)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    max_iter : int
        maximum number of iterations of NrKmeans.  Only used for init='nrkmeans' (default: 100)
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate. Only used for init='sgd'
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used. Only used for init='sgd' (default: None)
    batch_size : int
        size of the data batches. Only used for init='sgd' (default: 128)
    epochs : int
        number of epochs for the actual clustering procedure. Only used for init='sgd' (default: 10)
    device : torch.device
        device on which should be trained on. Only used for init='sgd' (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)
    init_kwargs : dict
        additional parameters that are used if init is a callable (optional) (default: None)
    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace
        list containing projections for each subspace
        orthogonal rotation matrix
        weights for softmax function to get beta values.

    Raises
    ----------
    ValueError : if init variable is passed that is not implemented.
    """
    if init == "nrkmeans" or init == "subkmeans":
        centers, P, V, beta_weights = nrkmeans_init(data=data, n_clusters=n_clusters, rounds=rounds,
                                                    input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                    debug=debug)
    elif init == "random":
        centers, P, V, beta_weights = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=rounds,
                                                           input_centers=input_centers, P=P, V=V,
                                                           random_state=random_state, debug=debug)
    elif init == "sgd":
        centers, P, V, beta_weights = sgd_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                               rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                               optimizer_class=optimizer_class, batch_size=batch_size,
                                               random_state=random_state, device=device, debug=debug)
    elif init == "acedec":
        centers, P, V, beta_weights = acedec_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                                  rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                                  optimizer_class=optimizer_class, batch_size=batch_size,
                                                  random_state=random_state, device=device, debug=debug)
    elif init == "auto":
        if data.shape[0] > 100000 or data.shape[1] > 1000:
            init = "sgd"
        else:
            init = "nrkmeans"
        centers, P, V, beta_weights = enrc_init(data=data, n_clusters=n_clusters, device=device, init=init,
                                                rounds=rounds, input_centers=input_centers,
                                                P=P, V=V, random_state=random_state, max_iter=max_iter,
                                                optimizer_params=optimizer_params, optimizer_class=optimizer_class,
                                                epochs=epochs, debug=debug)
    elif callable(init):
        if init_kwargs is not None:
            centers, P, V, beta_weights = init(data, n_clusters, **init_kwargs)
        else:
            centers, P, V, beta_weights = init(data, n_clusters)
    else:
        raise ValueError(f"init={init} is not implemented.")
    return centers, P, V, beta_weights


"""
===================== Cluster Reinitialization Strategy =====================
"""


def _calculate_rotated_embeddings_and_distances_for_n_samples(enrc: _ENRC_Module, model: torch.nn.Module,
                                                              dataloader: torch.utils.data.DataLoader, n_samples: int,
                                                              center_id: int, subspace_id: int, device: torch.device,
                                                              calc_distances: bool = True) -> (
        torch.Tensor, torch.Tensor):
    """
    Helper function for calculating the distances and embeddings for n_samples in a mini-batch fashion.

    Parameters
    ----------
    enrc : _ENRC_Module
        The ENRC Module
    model : torch.nn.Module
        The neural network
    dataloader : torch.utils.data.DataLoader
        dataloader from which data is randomly sampled
    n_samples : int
        the number of samples
    center_id : int
        the id of the center
    subspace_id : int
        the id of the subspace
    device : torch.device
        device to be trained on
    calc_distances : bool
        specifies if the distances between all not lonely centers to embedded data points should be calculated

    Returns
    -------
    tuple : (torch.Tensor, torch.Tensor)
        the rotated embedded data points
        the distances (if calc_distancesis True)
    """
    changed = True
    sample_count = 0
    subspace_betas = enrc.subspace_betas()[subspace_id, :]
    subspace_centers = enrc.centers[subspace_id]
    embedding_rot = []
    dists = []
    for batch in dataloader:
        batch = batch[1].to(device)
        if (batch.shape[0] + sample_count) > n_samples:
            # Remove samples from the batch that are too many.
            # Assumes that dataloader randomly samples minibatches, 
            # so removing the last objects does not matter
            diff = (batch.shape[0] + sample_count) - n_samples
            batch = batch[:-diff]
        z_rot = enrc.rotate(model.encode(batch))
        embedding_rot.append(z_rot.detach().cpu())

        if calc_distances:
            # Calculate distance from all not lonely centers to embedded data points
            idx_other_centers = [i for i in range(subspace_centers.shape[0]) if i != center_id]
            weighted_squared_diff = squared_euclidean_distance(z_rot, subspace_centers[idx_other_centers],
                                                               weights=subspace_betas)
            dists.append(weighted_squared_diff.detach().cpu())

        # Increase sample_count by batch size. 
        sample_count += batch.shape[0]
        if sample_count >= n_samples:
            break
    embedding_rot = torch.cat(embedding_rot, 0)
    if calc_distances:
        dists = torch.cat(dists, 0)
    else:
        dists = None
    return embedding_rot, dists


def _split_most_expensive_cluster(distances: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Splits most expensive cluster calculated based on the k-means loss and returns a new centroid.
    The new centroid is the one which is worst represented (largest distance) by the most expensive cluster centroid.

    Parameters
    ----------
    distances : torch.Tensor
        n x n distance matrix, where n is the size of the mini-batch.
    z : torch.Tensor
        n x d embedded data point, where n is the size of the mini-batch and d is the dimensionality of the embedded space.

    Returns
    -------
    center : torch.Tensor
        new center
    """
    kmeans_loss = distances.sum(0)
    costly_centroid_idx = kmeans_loss.argmax()
    max_idx = distances[costly_centroid_idx, :].argmax()
    return z[max_idx]


def _random_reinit_cluster(embedded: torch.Tensor) -> torch.Tensor:
    """
    Reinitialize random cluster centers.

    Parameters
    ----------
    embedded : torch.Tensor
        The embedded data points

    Returns
    -------
    center : torch.Tensor
        The random center
    """
    rand_indices = np.random.randint(low=0, high=embedded.shape[0], size=1)
    random_perturbation = torch.empty_like(embedded[rand_indices]).normal_(mean=embedded.mean().item(),
                                                                           std=embedded.std().item())
    center = embedded[rand_indices] + 0.0001 * random_perturbation
    return center


def reinit_centers(enrc: _ENRC_Module, subspace_id: int, dataloader: torch.utils.data.DataLoader,
                   model: torch.nn.Module,
                   n_samples: int = 512, kmeans_steps: int = 10, split: str = "random", debug: bool = False) -> None:
    """
    Reinitializes centers that have been lost, i.e. if they did not get any data point assigned. Before a center is reinitialized,
    this method checks whether a center has not get any points assigned over several mini-batch iterations and if this count is higher than
    enrc.reinit_threshold the center will be reinitialized.
    
    Parameters
    ----------
    enrc : _ENRC_Module
        torch.nn.Module instance for the ENRC algorithm
    subspace_id : int
        integer which indicates which subspace the cluster to be checked are in.
    dataloader : torch.utils.data.DataLoader
        dataloader from which data is randomly sampled. Important shuffle=True needs to be set, because n_samples random samples are drawn.
    model : torch.nn.Module
        neural network used for the embedding
    n_samples : int
        number of samples that should be used for the reclustering (default: 512)
    kmeans_steps : int
        number of mini-batch kmeans steps that should be conducted with the new centroid (default: 10)
    split : str
        {'random', 'cost'}, default='random', select how clusters should be split for renitialization.
        'random' : split a random point from the random sample of size=n_samples.
        'cost' : split the cluster with max kmeans cost.
    debug : bool
        if True than training errors will be printed (default: True)
    """
    N = len(dataloader.dataset)
    if n_samples > N:
        if debug: print(
            f"WARNING: n_samples={n_samples} > number of data points={N}. Set n_samples=number of data points")
        n_samples = N
    # Assumes that enrc and model are on the same device
    device = enrc.V.device
    with torch.no_grad():
        k = enrc.centers[subspace_id].shape[0]
        subspace_betas = enrc.subspace_betas()
        for center_id, count_i in enumerate(enrc.lonely_centers_count[subspace_id].flatten()):
            if count_i > enrc.reinit_threshold:
                if debug: print(f"Reinitialize cluster {center_id} in subspace {subspace_id}")
                if split == "cost":
                    embedding_rot, dists = _calculate_rotated_embeddings_and_distances_for_n_samples(enrc, model,
                                                                                                     dataloader,
                                                                                                     n_samples,
                                                                                                     center_id,
                                                                                                     subspace_id,
                                                                                                     device)
                    new_center = _split_most_expensive_cluster(distances=dists, z=embedding_rot)
                elif split == "random":
                    embedding_rot, _ = _calculate_rotated_embeddings_and_distances_for_n_samples(enrc, model,
                                                                                                 dataloader, n_samples,
                                                                                                 center_id, subspace_id,
                                                                                                 device,
                                                                                                 calc_distances=False)
                    new_center = _random_reinit_cluster(embedding_rot)
                else:
                    raise NotImplementedError(f"split={split} is not implemented. Has to be 'cost' or 'random'.")
                enrc.centers[subspace_id][center_id, :] = new_center.to(device)

                embeddingloader = torch.utils.data.DataLoader(embedding_rot, batch_size=dataloader.batch_size,
                                                              shuffle=False, drop_last=False)
                # perform mini-batch kmeans steps
                batch_cluster_sums = 0
                mask_sum = 0
                for step_i in range(kmeans_steps):
                    for z_rot in embeddingloader:
                        z_rot = z_rot.to(device)
                        weighted_squared_diff = squared_euclidean_distance(z_rot, enrc.centers[subspace_id],
                                                                           weights=subspace_betas[subspace_id, :])
                        assignments = weighted_squared_diff.detach().argmin(1)
                        one_hot_mask = int_to_one_hot(assignments, k)
                        batch_cluster_sums += (z_rot.unsqueeze(1) * one_hot_mask.unsqueeze(2)).sum(0)
                        mask_sum += one_hot_mask.sum(0)
                    nonzero_mask = (mask_sum != 0)
                    enrc.centers[subspace_id][nonzero_mask] = batch_cluster_sums[nonzero_mask] / mask_sum[
                        nonzero_mask].unsqueeze(1)
                    # Reset mask_sum
                    enrc.mask_sum[subspace_id] = mask_sum.unsqueeze(1)
                # lonely_centers_count is reset
                enrc.lonely_centers_count[subspace_id][center_id] = 0


"""
===================== ENRC  =====================
"""


def _are_labels_equal(labels_new: np.ndarray, labels_old: np.ndarray, threshold: float = None) -> bool:
    """
    Check if the old labels and new labels are equal. Therefore check the nmi for each subspace_nr. If all are 1, labels
    have not changed.
    
    Parameters
    ----------
    labels_new: np.ndarray
        new labels list
    labels_old: np.ndarray
        old labels list
    threshold: float
        specifies how close the two labelings should match (default: None)

    Returns
    ----------
    changed : bool
        True if labels for all subspaces are the same
    """
    if labels_new is None or labels_old is None or labels_new.shape[1] != labels_old.shape[1]:
        return False

    if threshold is None:
        v = 1
    else:
        v = 1 - threshold
    return all(
        [normalized_mutual_info_score(labels_new[:, i], labels_old[:, i], average_method="arithmetic") >= v for i in
         range(labels_new.shape[1])])


def _enrc(X: np.ndarray, n_clusters: list, V: np.ndarray, P: list, input_centers: list, batch_size: int,
          pretrain_optimizer_params: dict, clustering_optimizer_params: dict, pretrain_epochs: int,
          clustering_epochs: int, optimizer_class: torch.optim.Optimizer, ssl_loss_fn: torch.nn.modules.loss._Loss,
          clustering_loss_weight: float, ssl_loss_weight: float, neural_network: torch.nn.Module | tuple,
          neural_network_weights: str, embedding_size: int, init: str, random_state: np.random.RandomState,
          device: torch.device, scheduler: torch.optim.lr_scheduler, scheduler_params: dict, tolerance_threshold: float,
          init_kwargs: dict, init_subsample_size: int, custom_dataloaders: tuple, augmentation_invariance: bool,
          final_reclustering: bool, debug: bool) -> (
        np.ndarray, list, np.ndarray, list, np.ndarray, list, list, torch.nn.Module):
    """
    Start the actual ENRC clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        input data
    n_clusters : list
        list containing number of clusters for each clustering
    V : np.ndarray
        orthogonal rotation matrix
    P : list
        list containing projections for each clustering
    input_centers : list
        list containing the cluster centers for each clustering
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate
    clustering_optimizer_params: dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network
    clustering_epochs : int
        maximum number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        optimizer for pretraining and training
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    clustering_loss_weight : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss
    neural_network : torch.nn.Module | tuple
        the input neural network.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network.
    embedding_size : int
        size of the embedding within the neural network. Only used if neural_network is None
    init : str
        strchoose which initialization strategy should be used. Has to be one of 'nrkmeans', 'random' or 'sgd'.
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    device : torch.device
        if device is None then it will be checked whether a gpu is available or not
    scheduler : torch.optim.lr_scheduler
        learning rate scheduler that should be used
    scheduler_params : dict
        dictionary of the parameters of the scheduler object
    tolerance_threshold : float
        tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
        for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
        will train as long as the labels are not changing anymore.
    init_kwargs : dict
        additional parameters that are used if init is a callable
    init_subsample_size : int
        specify if only a subsample of size 'init_subsample_size' of the data should be used for the initialization
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    final_reclustering : bool
        If True, the final embedding will be reclustered with the provided init strategy. (defaul: False)
    debug : bool
        if True additional information during the training will be printed

    Returns
    -------
    tuple : (np.ndarray, list, np.ndarray, list, np.ndarray, list, list, torch.nn.Module)
        the cluster labels,
        the cluster centers,
        the orthogonal rotation matrix,
        the dimensionalities of the subspaces,
        the betas,
        the projections of the subspaces,
        the final n_clusters,
        the final neural network
        the cluster labels before final_reclustering
    """
    # Set device to train on
    if device is None:
        device = detect_device()
    # Setup dataloaders
    trainloader, testloader, batch_size = get_train_and_test_dataloader(X, batch_size, custom_dataloaders)
    if custom_dataloaders is not None:
        if debug: print("Custom dataloaders are used, X will be overwritten with testloader return values.")
        _preprocessed = []
        for batch in testloader: _preprocessed.append(batch[1])
        X = torch.cat(_preprocessed)
    # Use subsample of the data if specified and subsample is smaller than dataset
    if init_subsample_size is not None and init_subsample_size > 0 and init_subsample_size < X.shape[0]:
        rand_idx = random_state.choice(X.shape[0], init_subsample_size, replace=False)
        subsampleloader = get_dataloader(X[rand_idx], batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        subsampleloader = testloader
    # Setup neural network
    neural_network = get_trained_network(trainloader, n_epochs=pretrain_epochs,
                                         optimizer_params=pretrain_optimizer_params, optimizer_class=optimizer_class,
                                         device=device, ssl_loss_fn=ssl_loss_fn, embedding_size=embedding_size,
                                         neural_network=neural_network, neural_network_weights=neural_network_weights,
                                         random_state=random_state)
    # Run ENRC init
    if debug:
        print("Run init: ", init)
        print("Start encoding")
    embedded_data = encode_batchwise(subsampleloader, neural_network)
    if debug: print("Start initializing parameters")
    # set init epochs proportional to clustering_epochs
    init_epochs = np.max([10, int(0.2 * clustering_epochs)])
    input_centers, P, V, beta_weights = enrc_init(data=embedded_data, n_clusters=n_clusters, device=device, init=init,
                                                  rounds=10, epochs=init_epochs, batch_size=batch_size, debug=debug,
                                                  input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                  max_iter=100, optimizer_params=clustering_optimizer_params,
                                                  optimizer_class=optimizer_class, init_kwargs=init_kwargs)
    # Setup ENRC Module
    enrc_module = _ENRC_Module(input_centers, P, V, clustering_loss_weight=clustering_loss_weight,
                               ssl_loss_weight=ssl_loss_weight,
                               beta_weights=beta_weights, augmentation_invariance=augmentation_invariance).to_device(
        device)
    if debug:
        print("Betas after init")
        print(enrc_module.subspace_betas().detach().cpu().numpy())
    # In accordance to the original paper we update the betas 10 times faster
    clustering_optimizer_beta_params = clustering_optimizer_params.copy()
    clustering_optimizer_beta_params["lr"] = clustering_optimizer_beta_params["lr"] * 10
    param_dict = [dict({'params': neural_network.parameters()}, **clustering_optimizer_params),
                  dict({'params': [enrc_module.V]}, **clustering_optimizer_params),
                  dict({'params': [enrc_module.beta_weights]}, **clustering_optimizer_beta_params)
                  ]
    optimizer = optimizer_class(param_dict)

    if scheduler is not None:
        scheduler = scheduler(optimizer, **scheduler_params)

    # Training loop
    if debug: print("Start training")
    enrc_module.fit(trainloader=trainloader,
                    evalloader=testloader,
                    max_epochs=clustering_epochs,
                    optimizer=optimizer,
                    ssl_loss_fn=ssl_loss_fn,
                    batch_size=batch_size,
                    model=neural_network,
                    device=device,
                    scheduler=scheduler,
                    tolerance_threshold=tolerance_threshold,
                    debug=debug)

    if debug:
        print("Betas after training")
        print(enrc_module.subspace_betas().detach().cpu().numpy())

    cluster_labels_before_reclustering = enrc_module.predict_batchwise(model=neural_network, dataloader=testloader,
                                                                       device=device, use_P=True)
    # Recluster
    if final_reclustering:
        if debug:
            print("Recluster")
        enrc_module.recluster(dataloader=subsampleloader, model=neural_network, device=device,
                              optimizer_params=clustering_optimizer_params,
                              optimizer_class=optimizer_class, reclustering_strategy=init, init_kwargs=init_kwargs)
        # Predict labels and transfer other parameters to numpy
        cluster_labels = enrc_module.predict_batchwise(model=neural_network, dataloader=testloader, device=device,
                                                       use_P=True)
        if debug:
            print("Betas after reclustering")
            print(enrc_module.subspace_betas().detach().cpu().numpy())
    else:
        cluster_labels = cluster_labels_before_reclustering
    cluster_centers = [centers_i.detach().cpu().numpy() for centers_i in enrc_module.centers]
    V = enrc_module.V.detach().cpu().numpy()
    betas = enrc_module.subspace_betas().detach().cpu().numpy()
    P = enrc_module.P
    m = enrc_module.m
    return cluster_labels, cluster_centers, V, m, betas, P, n_clusters, neural_network, cluster_labels_before_reclustering


class ENRC(_AbstractDeepClusteringAlgo):
    """
    The Embeddedn Non-Redundant Clustering (ENRC) algorithm.
        
    Parameters
    ----------
    n_clusters : list
        list containing number of clusters for each clustering
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    P : list
        list containing projections for each clustering (optional) (default: None)
    input_centers : list
        list containing the cluster centers for each clustering (optional) (default: None)
    batch_size : int
        size of the data batches (default: 128)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        maximum number of epochs for the actual clustering procedure (default: 150)
    tolerance_threshold : float
        tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
        for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
        will train as long as the labels are not changing anymore (default: None)
    optimizer_class : torch.optim.Optimizer
        optimizer for pretraining and training (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    clustering_loss_weight : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network. Only used if neural_network is None (default: 20)
    init : str
        choose which initialization strategy should be used. Has to be one of 'nrkmeans', 'random' or 'sgd' (default: 'nrkmeans')
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    scheduler : torch.optim.lr_scheduler
        learning rate scheduler that should be used (default: None)
    scheduler_params : dict
        dictionary of the parameters of the scheduler object (default: None)
    init_kwargs : dict
        additional parameters that are used if init is a callable (optional) (default: None)
    init_subsample_size: int
        specify if only a subsample of size 'init_subsample_size' of the data should be used for the initialization. If None, all data will be used. (default: 10,000)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    final_reclustering : bool
        If True, the final embedding will be reclustered with the provided init strategy. (defaul: False)
    debug: bool
        if True additional information during the training will be printed (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers
    neural_network : torch.nn.Module
        The final neural network

    Raises
    ----------
    ValueError : if init is not one of 'nrkmeans', 'random', 'auto' or 'sgd'.

    References
    ----------
    Miklautz, Lukas & Dominik Mautz et al. "Deep embedded non-redundant clustering."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.
    """

    def __init__(self, n_clusters: list, V: np.ndarray = None, P: list = None, input_centers: list = None,
                 batch_size: int = 128, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 tolerance_threshold: float = None, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 clustering_loss_weight: float = 1.0, ssl_loss_weight: float = 1.0,
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 20, init: str = "nrkmeans",
                 device: torch.device = None, scheduler: torch.optim.lr_scheduler = None,
                 scheduler_params: dict = None, init_kwargs: dict = None, init_subsample_size: int = 10000,
                 random_state: np.random.RandomState | int = None, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, final_reclustering: bool = True, debug: bool = False):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters = n_clusters.copy()
        self.pretrain_optimizer_params = {
            "lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {
            "lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.tolerance_threshold = tolerance_threshold
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.init_kwargs = init_kwargs
        self.init_subsample_size = init_subsample_size
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.final_reclustering = final_reclustering
        self.debug = debug

        if len(self.n_clusters) < 2:
            raise ValueError(f"n_clusters={n_clusters}, but should be <= 2.")

        if init in available_init_strategies():
            self.init = init
        else:
            raise ValueError(f"init={init} does not exist, has to be one of {available_init_strategies()}.")
        self.input_centers = input_centers
        self.V = V
        self.m = None
        self.P = P

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ENRC':
        """
        Cluster the input dataset with the ENRC algorithm. Saves the labels, centers, V, m, Betas, and P
        in the ENRC object.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            input data
        y : np.ndarray
            the labels (can be ignored)
            
        Returns
        ----------
        self : ENRC
            returns the ENRC object
        """
        super().fit(X, y)
        cluster_labels, cluster_centers, V, m, betas, P, n_clusters, neural_network, cluster_labels_before_reclustering = _enrc(
            X=X,
            n_clusters=self.n_clusters,
            V=self.V,
            P=self.P,
            input_centers=self.input_centers,
            batch_size=self.batch_size,
            pretrain_optimizer_params=self.pretrain_optimizer_params,
            clustering_optimizer_params=self.clustering_optimizer_params,
            pretrain_epochs=self.pretrain_epochs,
            clustering_epochs=self.clustering_epochs,
            tolerance_threshold=self.tolerance_threshold,
            optimizer_class=self.optimizer_class,
            ssl_loss_fn=self.ssl_loss_fn,
            clustering_loss_weight=self.clustering_loss_weight,
            ssl_loss_weight=self.ssl_loss_weight,
            neural_network=self.neural_network,
            neural_network_weights=self.neural_network_weights,
            embedding_size=self.embedding_size,
            init=self.init,
            random_state=self.random_state,
            device=self.device,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            init_kwargs=self.init_kwargs,
            init_subsample_size=self.init_subsample_size,
            custom_dataloaders=self.custom_dataloaders,
            augmentation_invariance=self.augmentation_invariance,
            final_reclustering=self.final_reclustering,
            debug=self.debug)
        # Update class variables
        self.labels_ = cluster_labels
        self.enrc_labels_ = cluster_labels_before_reclustering
        self.cluster_centers_ = cluster_centers
        self.V = V
        self.m = m
        self.P = P
        self.betas = betas
        self.n_clusters = n_clusters
        self.neural_network = neural_network
        return self

    def predict(self, X: np.ndarray = None, use_P: bool = True,
                dataloader: torch.utils.data.DataLoader = None) -> np.ndarray:
        """
        Predicts the labels for each clustering of X in a mini-batch manner.
        
        Parameters
        ----------
        X : np.ndarray
            input data
        use_P: bool
            if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: True)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used. Can be None if X is given (default: None)

        Returns
        -------
        predicted_labels : np.ndarray
            n x c matrix, where n is the number of data points in X and c is the number of clusterings.
        """
        if dataloader is None:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.neural_network.to(self.device)
        predicted_labels = enrc_predict_batchwise(V=torch.from_numpy(self.V).float().to(self.device),
                                                  centers=[torch.from_numpy(c).float().to(self.device) for c in
                                                           self.cluster_centers_],
                                                  subspace_betas=torch.from_numpy(self.betas).float().to(self.device),
                                                  model=self.neural_network,
                                                  dataloader=dataloader,
                                                  device=self.device,
                                                  use_P=use_P)
        return predicted_labels

    def transform_full_space(self, X: np.ndarray, embedded=False) -> np.ndarray:
        """
        Embedds the input dataset with the neural network and the matrix V from the ENRC object.
        Parameters
        ----------
        X : np.ndarray
            input data
        embedded : bool
            if True, then X is assumed to be already embedded (default: False)
        
        Returns
        -------
        rotated : np.ndarray
            The transformed data
        """
        if not embedded:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)
            emb = encode_batchwise(dataloader=dataloader, neural_network=self.neural_network)
        else:
            emb = X
        rotated = np.matmul(emb, self.V)
        return rotated

    def transform_subspace(self, X: np.ndarray, subspace_index: int = 0, embedded: bool = False) -> np.ndarray:
        """
        Embedds the input dataset with the neural network and with the matrix V projected onto a special clusterspace_nr.
        
        Parameters
        ----------
        X : np.ndarray
            input data
        subspace_index: int
            index of the subspace_nr (default: 0)
        embedded: bool
            if True, then X is assumed to be already embedded (default: False)
        
        Returns
        -------
        subspace : np.ndarray
            The transformed subspace
        """
        if not embedded:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)
            emb = encode_batchwise(dataloader=dataloader, neural_network=self.neural_network)
        else:
            emb = X
        cluster_space_V = self.V[:, self.P[subspace_index]]
        subspace = np.matmul(emb, cluster_space_V)
        return subspace

    def plot_subspace(self, X: np.ndarray, subspace_index: int = 0, labels: np.ndarray = None,
                      plot_centers: bool = False,
                      gt: np.ndarray = None, equal_axis: bool = False) -> None:
        """
        Plot the specified subspace_nr as scatter matrix plot.
       
        Parameters
        ----------
        X : np.ndarray
            input data
        subspace_index: int
            index of the subspace_nr (default: 0)
        labels: np.ndarray
            the labels to use for the plot (default: labels found by Nr-Kmeans) (default: None)
        plot_centers: bool
            plot centers if True (default: False)
        gt: np.ndarray
            of ground truth labels (default=None)
        equal_axis: bool
            equalize axis if True (default: False)
        Returns
        -------
        scatter matrix plot of the input data
        """
        if self.labels_ is None:
            raise Exception("The ENRC algorithm has not run yet. Use the fit() function first.")
        if labels is None:
            labels = self.labels_[:, subspace_index]
        if X.shape[0] != labels.shape[0]:
            raise Exception("Number of data objects must match the number of labels.")
        plot_scatter_matrix(self.transform_subspace(X, subspace_index), labels,
                            self.cluster_centers_[subspace_index] if plot_centers else None,
                            true_labels=gt, equal_axis=equal_axis)

    def reconstruct_subspace_centroids(self, subspace_index: int = 0) -> np.ndarray:
        """
        Reconstructs the centroids in the specified subspace_nr.

        Parameters
        ----------
        subspace_index: int
            index of the subspace_nr (default: 0)

        Returns
        -------
        centers_rec : centers_rec
            reconstructed centers as np.ndarray
        """
        cluster_space_centers = self.cluster_centers_[subspace_index]
        # rotate back as centers are in the V-rotated space
        centers_rot_back = np.matmul(cluster_space_centers, self.V.transpose())
        centers_rec = self.neural_network.decode(torch.from_numpy(centers_rot_back).float().to(self.device))
        return centers_rec.detach().cpu().numpy()


class ACeDeC(ENRC):
    """
    Autoencoder Centroid-based Deep Cluster (ACeDeC) can be seen as a special case of ENRC where we have one
    cluster space and one shared space with a single cluster.
  
    Parameters
    ----------
    n_clusters : int
        number of clusters
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    P : list
        list containing projections for clusters in clustered space and cluster in shared space (optional) (default: None)
    input_centers : list
        list containing the cluster centers for clusters in clustered space and cluster in shared space (optional) (default: None)
    batch_size : int
        size of the data batches (default: 128)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        maximum number of epochs for the actual clustering procedure (default: 150)
    tolerance_threshold : float
        tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
        for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
        will train as long as the labels are not changing anymore (default: None)
    optimizer_class : torch.optim.Optimizer
        optimizer for pretraining and training (default: torch.optim.Adam)
    ssl_loss_fn : torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: torch.nn.MSELoss())
    clustering_loss_weight : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network. Only used if neural_network is None (default: 20)
    init : str
        choose which initialization strategy should be used. Has to be one of 'acedec', 'subkmeans', 'random' or 'sgd' (default: 'acedec')
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    scheduler : torch.optim.lr_scheduler
        learning rate scheduler that should be used (default: None)
    scheduler_params : dict
        dictionary of the parameters of the scheduler object (default: None)
    init_kwargs : dict
        additional parameters that are used if init is a callable (optional) (default: None)
    init_subsample_size: int
        specify if only a subsample of size 'init_subsample_size' of the data should be used for the initialization. If None, all data will be used. (default: 10,000)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    final_reclustering : bool
        If True, the final embedding will be reclustered with the provided init strategy. (default: True)
    debug: bool
        if True additional information during the training will be printed (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers
    neural_network : torch.nn.Module
        The final neural_network

    Raises
    ----------
    ValueError : if init is not one of 'acedec', 'subkmeans', 'random', 'auto' or 'sgd'.

    References
    ----------
    Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Böhm, Claudia Plant:
    Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826-2832
    """

    def __init__(self, n_clusters: int, V: np.ndarray = None, P: list = None, input_centers: list = None,
                 batch_size: int = 128, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 tolerance_threshold: float = None, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 clustering_loss_weight: float = 1.0, ssl_loss_weight: float = 1.0,
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 20, init: str = "acedec",
                 device: torch.device = None, scheduler: torch.optim.lr_scheduler = None,
                 scheduler_params: dict = None, init_kwargs: dict = None, init_subsample_size: int = 10000,
                 random_state: np.random.RandomState | int = None, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False,
                 final_reclustering: bool = True, debug: bool = False):
        super().__init__([n_clusters, 1], V, P, input_centers,
                         batch_size, pretrain_optimizer_params, clustering_optimizer_params, pretrain_epochs,
                         clustering_epochs, tolerance_threshold, optimizer_class, ssl_loss_fn, clustering_loss_weight,
                         ssl_loss_weight, neural_network, neural_network_weights,
                         embedding_size, init, device, scheduler, scheduler_params, init_kwargs,
                         init_subsample_size, random_state, custom_dataloaders, augmentation_invariance,
                         final_reclustering, debug)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ACeDeC':
        """
        Cluster the input dataset with the ACeDeC algorithm. Saves the labels, centers, V, m, Betas, and P
        in the ACeDeC object.
        The resulting cluster labels will be stored in the labels_ attribute.
        Parameters
        ----------
        X : np.ndarray
            input data
        y : np.ndarray
            the labels (can be ignored)
        Returns
        ----------
        self : ACeDeC
            returns the AceDeC object
        """
        super().fit(X, y)
        self.labels_ = self.labels_[:, 0]
        self.acedec_labels_ = self.enrc_labels_[:, 0]
        return self

    def predict(self, X: np.ndarray, use_P: bool = True, dataloader: torch.utils.data.DataLoader = None) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data
        use_P: bool
            if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: True)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used. Can be None if X is given (default: None)

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        predicted_labels = super().predict(X, use_P, dataloader)
        return predicted_labels[:, 0]

import torch
import torch.nn.functional as F
from itertools import islice


class Simple_Autoencoder(torch.nn.Module):
    """A vanilla symmetric autoencoder.

    Args:
        input_dim: size of each input sample
        embedding_size: size of the inner most layer also called embedding

    Attributes:
        encoder: encoder part of the autoencoder, responsible for embedding data points
        decoder: decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    """

    def __init__(self, input_dim: int, embedding_size: int, small_network=False):
        super(Simple_Autoencoder, self).__init__()

        if small_network:
            # make a sequential list of all operations you want to apply for encoding a data point
            self.encoder = torch.nn.Sequential(
                # Linear layer (just a matrix multiplication)
                torch.nn.Linear(input_dim, 256),
                # apply an elementwise non-linear function
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(256, 128),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(64, embedding_size))

            # make a sequential list of all operations you want to apply for decoding a data point
            # In our case this is a symmetric version of the encoder
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, 64),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(64, 128),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(128, 256),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(256, input_dim),
            )
        else:
            # make a sequential list of all operations you want to apply for encoding a data point
            self.encoder = torch.nn.Sequential(
                # Linear layer (just a matrix multiplication)
                torch.nn.Linear(input_dim, 500),
                # apply an elementwise non-linear function
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, 500),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, 2000),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(2000, embedding_size))

            # make a sequential list of all operations you want to apply for decoding a data point
            # In our case this is a symmetric version of the encoder
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, 2000),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(2000, 500),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, 500),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(500, input_dim),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points

        Returns:
            embedded: the embedded data point with dimensionality embedding_size
        """
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedded: embedded data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Applies both encode and decode function.
        The forward function is automatically called if we call self(x).
        Args:
            x: input data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        for _ in range(n_epochs):
            for batch in trainloader:
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
                
class Stacked_Autoencoder(torch.nn.Module):
    'stacked AE'
    def __init__(self, feature_dim, layer_dims, weight_initalizer,
                 loss_fn=lambda x, y: torch.mean((x - y) ** 2),
                 optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.001),
                 tied_weights=False, activation_fn=None, bias_init=0.0, linear_embedded=True, linear_decoder_last=True
                 ):
        """
        :param feature_dim:
        :param layer_dims:
        :param weight_initalizer: a one parameter function which given a tensor initializes it, e.g. a function from torch.nn.init
        :param tied_weights:
        :param loss_fn: The loss function that should be used for pretraining and fine tuning accepting as first
        :param optimizer_fn: A function which returns an torch optimizer for the given parameters (given as parameters ;-)
         parameter the original value and as the second the reconstruction
        :param activation_fn:
        :param bias_init:
        :param linear_decoder_last: If True the last layer does not have the activation function
        """
    
        #super(Autoencoder, self).__init__()
        super().__init__()
        self.tied_weights = tied_weights
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        self.linear_decoder_last = linear_decoder_last
        self.linear_embedded = linear_embedded

        # [torch.nn.Parameter(, requires_grad=True) for
        #                               feature_dim, node_dim in ]

        self.n_layers = len(layer_dims)

        self.param_bias_encoder = []
        self.param_bias_decoder = []
        self.param_weights_encoder = []
        if tied_weights:
            self.param_weights_decoder = None
        else:
            self.param_weights_decoder = []

        layer_params = list(window([feature_dim] + layer_dims, 2))

        for l in range(self.n_layers):
            feature_dim, node_dim = layer_params[l]
            encoder_weight = torch.empty(node_dim, feature_dim)
            weight_initalizer(encoder_weight)
            encoder_weight = torch.nn.Parameter(encoder_weight, requires_grad=True)
            self.register_parameter(f"encoder_weight_{l}", encoder_weight)
            self.param_weights_encoder.append(encoder_weight)
            encoder_bias = torch.empty(node_dim)
            encoder_bias.fill_(bias_init)
            encoder_bias = torch.nn.Parameter(encoder_bias, requires_grad=True)
            self.register_parameter(f"encoder_bias_{l}", encoder_bias)
            self.param_bias_encoder.append(encoder_bias)

            if not tied_weights:
                decoder_weight = torch.empty(feature_dim, node_dim)
                weight_initalizer(decoder_weight)
                decoder_weight = torch.nn.Parameter(decoder_weight, requires_grad=True)
                self.register_parameter(f"decoder_weight_{l}", decoder_weight)
                self.param_weights_decoder.append(decoder_weight)
            decoder_bias = torch.empty(feature_dim)
            decoder_bias.fill_(bias_init)
            decoder_bias = torch.nn.Parameter(decoder_bias, requires_grad=True)
            self.register_parameter(f"decoder_bias_{l}", decoder_bias)
            self.param_bias_decoder.append(decoder_bias)
        if not tied_weights:
            self.param_weights_decoder.reverse()
        self.param_bias_decoder.reverse()
        self.activation_fn = activation_fn
        
    
    def forward_pretrain(self, input_data, stack, use_dropout=True, dropout_rate=0.2,
                         dropout_is_training=True):
        encoded_data = input_data
        if stack < 1 or stack > self.n_layers:
            raise RuntimeError(
                f"stack number {stack} is out or range (0,{self.n_layers})")
        for l in range(stack):
            weights = self.param_weights_encoder[l]
            bias = self.param_bias_encoder[l]
            # print(f"encoder stack: { l} weights-shape:{weights.shape} bias-shape:{bias.shape}")
            encoded_data = F.linear(encoded_data, weights, bias)

            if self.activation_fn is not None:
                # print(f"{self.linear_embedded} is False or ({stack} < {self.n_layers} and {l} < {stack - 1})")
                if self.linear_embedded is False or not (l == stack - 1 and stack == self.n_layers):
                    # print("\tuse activation function")
                    encoded_data = self.activation_fn(encoded_data)
                else:
                    # print("\t use linear activation")
                    pass
            if use_dropout:
                if not (
                        l == stack - 1 and stack == self.n_layers):  # The embedded space is linear and we do not want dropout
                    # print("\tapply dropout")
                    encoded_data = F.dropout(
                        encoded_data, p=dropout_rate, training=dropout_is_training)
        reconstructed_data = encoded_data

        for ll in range(stack - 1, -1, -1):
            l = self.n_layers - ll - 1
            # print(f"decoder layer ll:{ll} l:{l}")
            if self.tied_weights:
                # print("\ttied weights")
                weights = self.param_weights_encoder[self.n_layers - l - 1].t()
            else:
                weights = self.param_weights_decoder[l]
            bias = self.param_bias_decoder[l]
            # print(f"\t weight-shape: {weights.shape} bias-shape:{bias.shape}")
            reconstructed_data = F.linear(reconstructed_data, weights, bias)
            if self.activation_fn is not None:
                if self.linear_decoder_last is False or self.linear_decoder_last and ll > 0:
                    # print(f"\t apply activation function")
                    reconstructed_data = self.activation_fn(reconstructed_data)
            if use_dropout and ll > 0:
                # print(f"\t apply dropout")
                reconstructed_data = F.dropout(
                    reconstructed_data, p=dropout_rate, )

        return encoded_data, reconstructed_data

    def encode(self, input_data):
        encoded_data = input_data
        for l in range(self.n_layers):
            weights = self.param_weights_encoder[l]
            bias = self.param_bias_encoder[l]
            encoded_data = F.linear(encoded_data, weights, bias)
            if self.activation_fn is not None and not (self.linear_embedded and l == self.n_layers - 1):
                encoded_data = self.activation_fn(encoded_data)
        return encoded_data

    def decode(self, encoded_data):
        reconstructed_data = encoded_data

        for l in range(self.n_layers):
            if self.tied_weights:
                weights = self.param_weights_encoder[self.n_layers - l - 1].t()
            else:
                weights = self.param_weights_decoder[l]
            bias = self.param_bias_decoder[l]
            reconstructed_data = F.linear(reconstructed_data, weights, bias)
            if self.activation_fn is not None and not (self.linear_decoder_last and l == self.n_layers - 1):
                reconstructed_data = self.activation_fn(reconstructed_data)
        return reconstructed_data

    def forward(self, input_data):
        encoded_data = self.encode(input_data)
        reconstructed_data = self.decode(encoded_data)

        return encoded_data, reconstructed_data

    def parameters_pretrain(self, stack):
        parameters = []
        for l in range(stack):
            parameters.append(self.param_weights_encoder[l])
            parameters.append(self.param_bias_encoder[l])
        for ll in range(stack - 1, -1, -1):
            l = self.n_layers - ll - 1
            if not self.tied_weights:
                parameters.append(self.param_weights_decoder[l])
            parameters.append(self.param_bias_decoder[l])
        return parameters

    def pretrain(self, dataset, device, rounds_per_layer=1000, dropout_rate=0.2, corruption_fn=None):
        """
        Uses Adam to pretrain the model layer by layer
        :param rounds_per_layer:
        :param corruption_fn: Can be used to corrupt the input data for an denoising autoencoder
        :return:
        """

        for layer in range(1, self.n_layers + 1):
            print(f"Pretrain layer {layer}")
            optimizer = self.optimizer_fn(self.parameters_pretrain(layer))
            round = 0
            while True:  # each iteration is equal to an epoch
                for batch_data in dataset:

                    round += 1
                    if round > rounds_per_layer:
                        break

                    batch_data = batch_data[0]

                    batch_data = batch_data.to(device) # cuda()
                    if corruption_fn is not None:
                        corrupted_batch = corruption_fn(batch_data)
                        _, reconstruced_data = self.forward_pretrain(corrupted_batch, layer, use_dropout=True,
                                                                     dropout_rate=dropout_rate,
                                                                     dropout_is_training=True)
                    else:
                        _, reconstruced_data = self.forward_pretrain(batch_data, layer, use_dropout=True,
                                                                     dropout_rate=dropout_rate,
                                                                     dropout_is_training=True)
                    loss = self.loss_fn(batch_data, reconstruced_data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if round % 100 == 0:
                        print(f"Round {round} current loss: {loss.item()}")
                else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
                    continue
                break  # Break while loop here

    def refine_training(self, dataset, rounds, device, corruption_fn=None, optimizer_fn=None):
        print(f"Refine training")
        if optimizer_fn is None:
            optimizer = self.optimizer_fn(self.parameters())
        else:
            optimizer = optimizer_fn(self.parameters())

        index = 0
        while True:  # each iteration is equal to an epoch
            for batch_data in dataset:
                index += 1
                if index > rounds:
                    break
                batch_data = batch_data[0]

                batch_data = batch_data.to(device) #cuda()

                # Forward pass
                if corruption_fn is not None:
                    embeded_data, reconstruced_data = self.forward(
                        corruption_fn(batch_data))
                else:
                    embeded_data, reconstruced_data = self.forward(batch_data)

                loss = self.loss_fn(reconstruced_data, batch_data)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    print(f"Round {index} current loss: {loss.item()}")

            else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
                continue
            break  # Break while loop here   


def squared_euclidean_distance(centers, embedded, weights=None):
    ta = centers.unsqueeze(0)
    tb = embedded.unsqueeze(1)
    squared_diffs = (ta - tb)
    if weights is not None:
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).sum(2)  # .mean(2) # TODO Evaluate this change
    return squared_diffs


def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    for batch in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def predict_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch.to(device)
        prediction = cluster_module.prediction_hard(model.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()

def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
def add_noise(batch):
    mask = torch.empty(
        batch.shape, device=batch.device).bernoulli_(0.8)
    return batch * mask        

def int_to_one_hot(label_tensor, n_labels):
    onehot = torch.zeros([label_tensor.shape[0], n_labels], dtype=torch.float, device=label_tensor.device)
    onehot.scatter_(1, label_tensor.unsqueeze(1).long(), 1.0)
    return onehot

def get_trained_autoencoder(trainloader, learning_rate, n_epochs, device, optimizer_class, loss_fn,
                            input_dim, embedding_size, ae_stacked = False, autoencoder_class=Simple_Autoencoder):
    if embedding_size > input_dim:
        print(
            "WARNING: embedding_size is larger than the dimensionality of the input dataset. Setting embedding_size to",
            input_dim)
        embedding_size = input_dim
    # Pretrain Autoencoder
    # adjusted here
    if ae_stacked is False:
        autoencoder = autoencoder_class(input_dim=input_dim, embedding_size=embedding_size).to(device)
        optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
        autoencoder.start_training(trainloader, n_epochs, device, optimizer, loss_fn)
    else: ### added this
        #n_features = data.shape[1]
        ### die eventuell mal als Parameter
        ae_layout = [500, 500, 2000, embedding_size]
        steps_per_layer = 10000
        refine_training_steps = 20000
        ###
        autoencoder = autoencoder_class(input_dim, ae_layout, weight_initalizer=torch.nn.init.xavier_normal_,
        activation_fn=lambda x: F.relu(x), loss_fn=loss_fn, optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=learning_rate)).to(device)
        # train and testloader
        #trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
        autoencoder.pretrain(trainloader, device, rounds_per_layer=steps_per_layer, dropout_rate=0.2, corruption_fn=add_noise)
        autoencoder.refine_training(trainloader, refine_training_steps, device, corruption_fn=add_noise)
        #total_loss = get_total_loss(autoencoder,trainloader) <--- Hat mir einen Fehler geschmissen
        #print(total_loss)
    return autoencoder




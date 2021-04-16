import torch
#import torch.nn.functional as F
#from itertools import islice

class stacked_Autoencoder(torch.nn.Module):
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

    def pretrain(self, dataset, rounds_per_layer=1000, dropout_rate=0.2, corruption_fn=None):
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

                    #batch_data = batch_data.cuda()
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

    def refine_training(self, dataset, rounds, corruption_fn=None, optimizer_fn=None):
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

                #batch_data = batch_data.cuda()

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

#def count_values_in_sequence(seq):
#    from collections import defaultdict
#    res = defaultdict(lambda : 0)
#    for key in seq:
#        res[key] += 1
#    return dict(res)
######
    

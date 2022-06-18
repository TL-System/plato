"""
Implement the model, including the encoder and the projection, for the contrastive adaptation method.

"""

import torch
from torch import nn
from plato.config import Config

from plato.models import encoders_register
from plato.models import general_mlps_register

from moving_average import ModelEMA


class ProjectionMLP(nn.Module):
    """ The implementation of ContrasAdap's projection layer. """

    def __init__(self, in_dim):
        super().__init__()

        # we reuse the byol's projection structure
        self.layers = general_mlps_register.Model.get_model(
            model_type="byol_prediction_mlp", input_dim=in_dim)

    def forward(self, x):
        """ Forward the projection layer. """
        for layer in self.layers:
            x = layer(x)

        return x

    def output_dim(self):
        """ Obtain the output dimension. """
        return self.layers[-1][-1].out_features


class EncoderwithProjection(nn.Module):
    """ The module combining the encoder and the projection. """

    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # define the encoder based on the model_name in config
        if encoder is None:
            self.encoder, self.encode_dim = encoders_register.get()
        # utilize the custom model
        else:
            self.encoder, self.encode_dim = encoder, encoder_dim

        # build the projector proposed in the bylo net
        self.projector = ProjectionMLP(in_dim=self.encode_dim)

        self.projection_dim = self.projector.output_dim()

    def forward(self, x):
        """ Forward the encoder and the projection """
        x = self.encoder(x)
        x = self.projector(x)
        return x

    def get_projection_dim(self):
        return self.projection_dim


class ContrasAdap(nn.Module):
    """ The implementation of ContrasAdap method.

    """

    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # combine online encoder and online projector to be
        # the online network
        self.online_network = EncoderwithProjection(encoder, encoder_dim)

        projection_dim = self.online_network.get_projection_dim()
        # set the required online predictor
        # we reuse the structure of the BYOL
        # default use the same structure with its predictor
        # otherwise, according to BYOL implementations from other
        # researchers, the siasiam's MLP can also be used as the
        # structure of the BYOL's online predictor
        if not hasattr(Config.trainer, "predictor_type"):
            self.online_predictor = general_mlps_register.Model.get_model(
                model_type="byol_prediction_mlp", input_dim=projection_dim)
        else:
            self.online_predictor = general_mlps_register.Model.get_model(
                model_type=Config.trainer.predictor_type,
                input_dim=projection_dim)

        # define the target network
        # note that the target network will not be optimized, it update
        # only depends on the moving average of the previous target network
        # and the newly online network.
        self.target_network = EncoderwithProjection(encoder, encoder_dim)

        # set the moving average
        moving_average_decay = Config(
        ).trainer.onlient_target_moving_average_decay
        self.target_ema_updater = ModelEMA(moving_average_decay)

        self._initializes_target_network()

    @torch.no_grad()
    def _initializes_target_network(self):
        """ Initialize the tareget network. """
        for param_q, param_k in zip(self.online_network.parameters(),
                                    self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_target_network(self):
        """ Update the target network based on the EMA. """
        self.target_ema_updater.update_model_moving_average(
            previous_model=self.target_network,
            current_model=self.online_network)

    def forward(self, augmented_samples):
        """ Forward two batch of contrastive samples. """
        samples1, samples2 = augmented_samples
        online_proj_one = self.online_network(samples1)
        online_proj_two = self.online_network(samples2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        # forward the target network but stops the gradients
        with torch.no_grad():
            self._update_target_network()
            target_proj_one = self.target_network(samples1)
            target_proj_two = self.target_network(samples2)

        return (online_pred_one, online_pred_two), (target_proj_one,
                                                    target_proj_two)

    @property
    def encoder(self):
        """ Obtain the target network's encoder. """
        target_network_encoder = self.target_network.encoder
        return target_network_encoder

    @property
    def encode_dim(self):
        """ Obtain the target network's encoder. """

        return self.online_network.encode_dim

    @staticmethod
    def get_model():
        """Obtaining an instance of this model provided that the name is valid."""

        return ContrasAdap()
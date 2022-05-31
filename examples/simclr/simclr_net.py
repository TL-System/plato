"""
Implement the model, including the encoder and the projection, for the SimCLR method.

"""

from torch import nn

from plato.models import encoders_register
from plato.models import general_mlps_register


class ProjectionMLP(nn.Module):
    """ The implementation of SimCLR's projection part. """

    def __init__(self, in_dim):
        super().__init__()

        self.layers = general_mlps_register.Model.get_model(
            model_type="simclr_projection_mlp", input_dim=in_dim)

    def forward(self, x):
        """ Forward the projection block. """
        for layer in self.layers:
            x = layer(x)

        return x


class SimCLR(nn.Module):
    """ The implementation of SimCLR method. """

    def __init__(self):
        super().__init__()

        self.encoder, self.encode_dim = encoders_register.get()

        # build the projector proposed in the simclr net
        self.projector = ProjectionMLP(in_dim=self.encode_dim)

    def forward(self, samples1, samples2):
        """ Inference of two contrastive samples"""
        encoded_x1 = self.encoder(samples1)
        encoded_x2 = self.encoder(samples2)

        projected_z1 = self.projector(encoded_x1)
        projected_z2 = self.projector(encoded_x2)
        return projected_z1, projected_z2

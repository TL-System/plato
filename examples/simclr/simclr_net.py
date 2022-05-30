"""
Implement the model, including the encoder and the projection, for the SimCLR method.

"""

import torch.nn as nn

from plato.models import encoders_register
from plato.models import general_mlps_register


class projection_MLP(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.layers = general_mlps_register.Model.get_model(
            model_type="simclr_projection_mlp", input_dim=in_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class SimCLR(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder, self.encode_dim = encoders_register.get()

        # build the projector proposed in the simclr net
        self.projector = projection_MLP(in_dim=self.encode_dim)

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z1 = self.projector(h2)
        return z1, z1

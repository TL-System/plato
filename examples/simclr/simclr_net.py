"""
Implement the trainer for the SimCLR method.

"""

import torch.nn as nn

from plato.config import Config

from general_MLP import build_mlp_from_config
from encoder_register import register_encoder


class projection_MLP(nn.Module):

    def __init__(self, in_dim, out_dim=256):
        super().__init__()

        self.layers = build_mlp_from_config(
            dict(
                type='FullyConnectedHead',
                output_dim=out_dim,
                input_dim=in_dim,
                hidden_layers_dim=[in_dim],
                batch_norms=[None, None],
                activations=["relu", None],
                dropout_ratios=[0, 0],
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class SimCLR(nn.Module):

    def __init__(self):
        super().__init__()

        # use resnet18 by default
        defined_model_name = "resnet18"
        # dimension after the projection
        projection_dim = 128

        if hasattr(Config.trainer, "model_name"):
            defined_model_name = Config.trainer.model_name
        if hasattr(Config.trainer, "projection_dim"):
            projection_dim = Config.trainer.projection_dim

        self.encoder, encode_dim = register_encoder(
            base_model_name=defined_model_name)

        # build the
        self.projector = projection_MLP(in_dim=encode_dim,
                                        out_dim=projection_dim)

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z1 = self.projector(h2)
        return z1, z1

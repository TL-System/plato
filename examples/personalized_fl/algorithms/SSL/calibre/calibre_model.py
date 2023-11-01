"""
Implementation of Net used in calibre.
"""

import torch

from lightly.models.modules.heads import SimCLRProjectionHead

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class CalibreNet(torch.nn.Module):
    """The model structure of Calibre."""

    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # extract hyper-parameters
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim

        # define the encoder based on the model_name in config
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim
        self.projector = SimCLRProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )

    def forward(self, multiview_samples):
        """Forward two batch of contrastive samples."""
        samples1, samples2 = multiview_samples
        encoded_h1 = self.encoder(samples1)
        encoded_h2 = self.encoder(samples2)

        projected_z1 = self.projector(encoded_h1)
        projected_z2 = self.projector(encoded_h2)
        return (encoded_h1, encoded_h2), (projected_z1, projected_z2)

    @staticmethod
    def get():
        """Get the defined CalibreNet model."""
        return CalibreNet()

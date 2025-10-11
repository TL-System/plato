"""
Implementation of Net used in calibre.
"""

import torch
from lightly.models.modules.heads import SimCLRProjectionHead

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class CalibreNet(torch.nn.Module):
    """The model structure of Calibre."""

    def __init__(self, encoder=None):
        super().__init__()

        # extract hyper-parameters
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # define the encoder based on the model_name in config
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.projector = SimCLRProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )

    def forward(self, multiview_samples):
        """Forward two batch of contrastive samples."""
        sample1, sample2 = multiview_samples
        encoded_sample1 = self.encoder(sample1)
        encoded_sample2 = self.encoder(sample2)

        projected_sample1 = self.projector(encoded_sample1)
        projected_sample2 = self.projector(encoded_sample2)
        return (encoded_sample1, encoded_sample2), (
            projected_sample1,
            projected_sample2,
        )

    @staticmethod
    def get():
        """Get the defined CalibreNet model."""
        return CalibreNet()

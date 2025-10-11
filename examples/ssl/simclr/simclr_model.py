"""
Model for the SimCLR algorithm.
"""

from lightly.models.modules.heads import SimCLRProjectionHead
from torch import nn

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class SimCLRModel(nn.Module):
    """The model structure of SimCLR."""

    def __init__(self, encoder=None):
        super().__init__()

        # Extract hyper-parameters.
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # Define the encoder based on the model_name in config.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        self.projector = SimCLRProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )

    def forward(self, multiview_samples):
        """Forward the contrastive samples."""
        view_sample1, view_sample2 = multiview_samples
        encoded_sample1 = self.encoder(view_sample1)
        encoded_sample2 = self.encoder(view_sample2)

        projected_sample1 = self.projector(encoded_sample1)
        projected_sample2 = self.projector(encoded_sample2)
        return projected_sample1, projected_sample2

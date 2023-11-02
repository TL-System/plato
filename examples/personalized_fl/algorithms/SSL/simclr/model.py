"""
Model speicifcally used in SimCLR algorithm.
"""
from torch import nn
from lightly.models.modules.heads import SimCLRProjectionHead
from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class SimCLR(nn.Module):
    """The model structure of SimCLR."""

    def __init__(self, encoder=None):
        super().__init__()

        # Extract hyper-parameters.
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim

        # Define the encoder based on the model_name in config.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        self.projector = SimCLRProjectionHead(
            self.encoder.encoding_dim, projection_hidden_dim, projection_out_dim
        )

    def forward(self, multiview_samples):
        """Forward two batch of contrastive samples."""
        samples1, samples2 = multiview_samples
        encoded_h1 = self.encoder(samples1)
        encoded_h2 = self.encoder(samples2)

        projected_z1 = self.projector(encoded_h1)
        projected_z2 = self.projector(encoded_h2)
        return projected_z1, projected_z2

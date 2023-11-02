"""
A model for the SwAV method.
"""

from torch import nn

from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class SwaV(nn.Module):
    """The model structure for the SwaV."""

    def __init__(self, encoder=None):
        super().__init__()

        # Define the encoder.
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # Define the encoder.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        # Define the projector.
        self.projector = SwaVProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )
        self.prototypes = SwaVPrototypes(
            Config().trainer.projection_out_dim,
            n_prototypes=Config().trainer.n_prototypes,
        )

    def forward_view(self, view_sample):
        """Foward views of the samples"""
        encoded_sample = self.encoder(view_sample).flatten(start_dim=1)
        projected_sample = self.projector(encoded_sample)
        normalized_sample = nn.functional.normalize(projected_sample, dim=1, p=2)
        outputs = self.prototypes(normalized_sample)
        return outputs

    def forward(self, multiview_samples):
        """Forward multiview samples."""
        self.prototypes.normalize()
        multi_crop_features = [self.forward_view(views) for views in multiview_samples]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]

        return high_resolution, low_resolution

"""
A model for the SwAV method.
"""

from torch import nn

from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class SwaV(nn.Module):
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

        self.encoding_dim = self.encoder.encoding_dim

        # Define the projector.
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        n_prototypes = Config().trainer.n_prototypes

        self.projection_head = SwaVProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )
        self.prototypes = SwaVPrototypes(projection_out_dim, n_prototypes=n_prototypes)

    def forward_direct(self, samples):
        encoded_samples = self.encoder(samples).flatten(start_dim=1)
        encoded_samples = self.projection_head(encoded_samples)
        encoded_samples = nn.functional.normalize(encoded_samples, dim=1, p=2)
        outputs = self.prototypes(encoded_samples)
        return outputs

    def forward(self, multiview_samples):
        self.prototypes.normalize()
        multi_crop_features = [
            self.forward_direct(sample) for sample in multiview_samples
        ]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]

        return high_resolution, low_resolution

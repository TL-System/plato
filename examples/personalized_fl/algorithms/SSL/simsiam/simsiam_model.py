"""
The model of the SimSiam method.
"""

from torch import nn

from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class SimSiam(nn.Module):
    """A SimSiam model."""

    def __init__(self, encoder=None):
        super().__init__()

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

        self.projection_head = SimSiamProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )
        self.prediction_head = SimSiamPredictionHead(
            Config().trainer.projection_out_dim,
            Config().trainer.prediction_hidden_dim,
            Config().trainer.prediction_out_dim,
        )

    def forward_direct(self, samples):
        encoded_samples = self.encoder(samples).flatten(start_dim=1)
        projected_samples = self.projection_head(encoded_samples)
        output = self.prediction_head(projected_samples)
        projected_samples = projected_samples.detach()
        return projected_samples, output

    def forward(self, multiview_samples):
        samples1, samples2 = multiview_samples
        projected_samples1, output1 = self.forward_direct(samples1)
        projected_samples2, output2 = self.forward_direct(samples2)
        return (projected_samples1, output2), (projected_samples2, output1)

"""
The model of the SimSiam method.
"""

from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from torch import nn

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


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
        # A projector projects higher dimension encodings to projections
        self.projector = SimSiamProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )
        # A predictor predicts the output from the projected features
        self.predictor = SimSiamPredictionHead(
            Config().trainer.projection_out_dim,
            Config().trainer.prediction_hidden_dim,
            Config().trainer.prediction_out_dim,
        )

    def forward_view(self, view_sample):
        """Foward one view sample to get the output."""
        encoded_view = self.encoder(view_sample).flatten(start_dim=1)
        projected_view = self.projector(encoded_view)
        output = self.predictor(projected_view)
        projected_view = projected_view.detach()
        return projected_view, output

    def forward(self, multiview_samples):
        """Main forward function of the model."""
        view_sample1, view_sample2 = multiview_samples
        projected_view1, output1 = self.forward_view(view_sample1)
        projected_view2, output2 = self.forward_view(view_sample2)
        return (projected_view1, output2), (projected_view2, output1)

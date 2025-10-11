"""
The model for the FedEMA algorithm.
"""

import copy

from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from torch import nn

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class BYOLModel(nn.Module):
    """The model structure of FedEMA."""

    def __init__(self, encoder=None):
        super().__init__()

        # Define the encoder.
        # An encoder encode a sample to a higher dimension
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        # A projector projects higher dimension features to
        # output dimensions
        self.projector = BYOLProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )
        # A predictor predicts the output from the projected features
        self.predictor = BYOLPredictionHead(
            Config().trainer.projection_out_dim,
            Config().trainer.prediction_hidden_dim,
            Config().trainer.prediction_out_dim,
        )

        # The momentum encoder and projector, which are work in
        # a momentum manner
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)

        # Deactivate the requires_grad flag for all parameters
        deactivate_requires_grad(self.momentum_encoder)
        deactivate_requires_grad(self.momentum_projector)

    def forward_view(self, view_sample):
        """Foward one view to get the output."""
        encoded_view = self.encoder(view_sample).flatten(start_dim=1)
        projected_view = self.projector(encoded_view)
        output = self.predictor(projected_view)
        return output

    def forward_momentum(self, view_sample):
        """Foward one view to get the output in a momentum manner."""
        encoded_view = self.momentum_encoder(view_sample).flatten(start_dim=1)
        projected_view = self.momentum_projector(encoded_view)
        projected_view = projected_view.detach()
        return projected_view

    def forward(self, multiview_samples):
        """Main forward function of the model."""
        view_sample1, view_sample2 = multiview_samples
        output1 = self.forward_view(view_sample1)
        momentum1 = self.forward_momentum(view_sample1)
        output2 = self.forward_view(view_sample2)
        momentum2 = self.forward_momentum(view_sample2)
        return (output1, momentum2), (output2, momentum1)

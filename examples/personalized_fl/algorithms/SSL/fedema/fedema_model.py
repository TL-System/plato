"""
The model for the FedEMA algorithm.
"""

import copy

from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class BYOLModel(nn.Module):
    """The model structure of BYOL."""

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

        self.encoding_dim = self.encoder.encoding_dim

        # A projector project higher dimension features to output dimensions.
        self.projector = BYOLProjectionHead(
            self.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )
        self.predictor = BYOLPredictionHead(
            Config().trainer.projection_out_dim,
            Config().trainer.prediction_hidden_dim,
            Config().trainer.prediction_out_dim,
        )

        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)

        deactivate_requires_grad(self.momentum_encoder)
        deactivate_requires_grad(self.momentum_projector)

    def forward_direct(self, sample):
        """Foward one sample to get the output."""
        encoded_sample = self.encoder(sample).flatten(start_dim=1)
        projected_sample = self.projector(encoded_sample)
        output = self.predictor(projected_sample)
        return output

    def forward_momentum(self, sample):
        """Foward one sample to get the output in a momentum manner."""
        encoded_example = self.momentum_encoder(sample).flatten(start_dim=1)
        projected_example = self.momentum_projector(encoded_example)
        projected_example = projected_example.detach()
        return projected_example

    def forward(self, multiview_samples):
        """Main forward function of the model."""
        sample1, sample2 = multiview_samples
        output1 = self.forward_direct(sample1)
        projected_sample1 = self.forward_momentum(sample1)
        output2 = self.forward_direct(sample2)
        projected_sample2 = self.forward_momentum(sample2)
        return (output1, projected_sample2), (output2, projected_sample1)

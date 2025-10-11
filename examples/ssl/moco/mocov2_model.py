"""
A model for the MoCoV2 method.
"""

import copy

from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from torch import nn

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class MoCoV2(nn.Module):
    """A model structure for the MoCoV2 method."""

    def __init__(self, encoder=None):
        super().__init__()

        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        # Define the encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        # Define the projector
        self.projector = MoCoProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projector_momentum = copy.deepcopy(self.projector)

        # Deactivate the requires_grad flag for all parameters
        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projector_momentum)

    def forward_view(self, view_sample):
        """Foward one view sample to get the output."""
        query = self.encoder(view_sample).flatten(start_dim=1)
        query = self.projector(query)
        return query

    def forward_momentum(self, view_sample):
        """Foward one view sample to get the output in a momentum manner."""
        key = self.encoder_momentum(view_sample).flatten(start_dim=1)
        key = self.projector_momentum(key).detach()
        return key

    def forward(self, multiview_samples):
        """Main forward function of the model."""
        view_sample1, view_sample2 = multiview_samples
        query = self.forward_view(view_sample1)
        key = self.forward_momentum(view_sample2)

        return query, key

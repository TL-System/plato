"""
Model for the SimCLR algorithm.
"""

from typing import Any, cast

from lightly.models.modules.heads import SimCLRProjectionHead
from torch import nn

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class SimCLRModel(nn.Module):
    """The model structure of SimCLR."""

    def __init__(self, encoder: nn.Module | None = None) -> None:
        super().__init__()

        # Extract hyper-parameters.
        encoder_name = Config().trainer.encoder_name
        params: dict[str, Any] = Config().params
        encoder_params_obj = params.get("encoder")
        if isinstance(encoder_params_obj, dict):
            encoder_params: dict[str, Any] = dict(encoder_params_obj)
        else:
            encoder_params = {}

        # Define the encoder based on the model_name in config.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        encoding_dim = cast(int, getattr(self.encoder, "encoding_dim"))

        self.projector = SimCLRProjectionHead(
            encoding_dim,
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

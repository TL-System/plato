"""
The implementation of the framework of the maskedFL

"""

import torch
from torch import nn
from transformers import ViTMAEModel, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEEmbeddings,
    ViTMAEEncoder,
    ViTMAEDecoder,
)

from vgbase.models.masking import (
    square_masking,
    blockwise_masking,
    blockwise_gmm_masking,
    random_masking,
)


class VisualMaksedEmbeddings(ViTMAEEmbeddings):
    """The masked embeddings.

    The masking mechanism is required to be reimplemented.
    """

    def __init__(self, config):
        super().__init__(config)

        self.supported_masking_methods = [
            "random_masking",
            "blockwise_masking",
            "blockwise_gmm_masking",
            "random_masking",
        ]

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        masking_name = self.config.masking_name
        if (
            hasattr(self.config, "masking_name")
            and self.config.masking_name in self.supported_masking_methods
        ):
            masking_method = getattr(self, masking_name)
        else:
            # utilize random masking by default
            masking_method = getattr(self, "random_masking")
        embeddings, mask, ids_restore = masking_method(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class MaskedGlobalEncoder(ViTMAEModel):
    """The global encoder that will be exchanged between the server and clients.

    It will perform two steps,
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = VisualMaksedEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_model():

        # set hyper-parameters
        configuration = ViTMAEConfig()
        return ViTMAEModel(configuration)


class PersonalizedDecoder(ViTMAEDecoder):
    """The personalized decoder utilized to reconstruct the masked image."""

    @staticmethod
    def get_model():

        # set hyper-parameters
        configuration = ViTMAEConfig()
        return ViTMAEModel(configuration)

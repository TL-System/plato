"""
Obtaining a ViT model for ImageClassification from the PyTorch Hub.
"""

from torch import nn
from transformers import AutoModelForImageClassification, AutoConfig
from plato.config import Config


class ModelChangeResolution(nn.Module):
    """
    Ttansform image resolution to assigned resolution of loaded pretrain model.
    """

    def __init__(self, model_name, config) -> nn.Module:
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )
        self.resolution = config.image_size

    def forward(self, image):
        """
        Forward function will first fit image resolution and finally output the logits.
        """
        if image.size(-1) != self.resolution:
            image = nn.functional.interpolate(
                image, size=self.resolution, mode="bicubic"
            )
        outputs = self.model(image)
        return outputs.logits


class Model:
    """
    The Transformer and other models loaded from HuggingFace.
    Supported by HuggingFace
    https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel
    """

    @staticmethod
    def get(model_name=None, **kwargs):  # pylint: disable=unused-argument
        """Returns a named model from HuggingFace."""
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        model_name = model_name.replace("@", "/")
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)

        return ModelChangeResolution(model_name, config)

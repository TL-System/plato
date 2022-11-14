"""
Obtaining a ViT model for image classification from HuggingFace.

The reference to T2T-ViT model:
https://github.com/yitu-opensource/T2T-ViT.

The reference to Deep-ViT model:
https://github.com/zhoudaquan/dvit_repo.

"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageClassification

from plato.config import Config
from plato.models import dvit, t2tvit


class ResolutionAdjustedModel(nn.Module):
    """
    Transforms the image resolution to the assigned resolution of a pretrained model.
    """

    def __init__(self, model_name, config) -> nn.Module:
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )

        if (
            hasattr(Config().parameters, "model")
            and hasattr(Config().parameters.model, "pretrained")
            and not Config().parameters.model.pretrained
        ):
            self.model.init_weights()
        self.resolution = config.image_size

    def forward(self, image):
        """
        Adjusts the image resolution and outputs the logits.
        """
        if image.size(-1) != self.resolution:
            image = nn.functional.interpolate(
                image, size=self.resolution, mode="bicubic"
            )
        outputs = self.model(image)
        return outputs.logits


class T2TVIT(nn.Module):
    """Wrap up t2t-vit."""

    def __init__(self, name) -> nn.Module:
        super().__init__()

        model_name = getattr(t2tvit.models, name)
        t2t = model_name(num_classes=Config().trainer.num_classes)

        if (
            hasattr(Config().parameters, "model")
            and hasattr(Config().parameters.model, "pretrained")
            and Config().parameters.model.pretrained
        ):
            t2tvit.utils.load_for_transfer_learning(
                t2t,
                Config().parameters.model.pretrain_path,
                use_ema=True,
                strict=False,
                num_classes=Config().trainer.num_classes,
            )
        self.model = t2t
        self.resolution = 224

    def forward(self, x):
        """forward function"""
        if x.size(-1) != self.resolution:
            x = nn.functional.interpolate(x, size=self.resolution, mode="bicubic")
        return self.model(x)


class DeepViT(nn.Module):
    """Wrap up deep vit."""

    def __init__(self, name) -> nn.Module:
        super().__init__()

        model_name = getattr(dvit.models.deep_vision_transformer, name)
        deepvit = model_name(
            pretrained=False, num_classes=Config.trainer.num_classes, in_chans=3
        )
        if (
            hasattr(Config().parameters, "model")
            and hasattr(Config().parameters.model, "pretrained")
            and Config().parameters.model.pretrained
        ):
            state_dict = torch.load(
                Config().parameters.model.pretrain_path, map_location="cpu"
            )
            del state_dict["head.weight"]
            del state_dict["head.bias"]
            deepvit.load_state_dict(state_dict)
        self.model = deepvit
        self.resolution = 224

    def forward(self, x):
        """forward function"""
        if x.size(-1) != self.resolution:
            x = nn.functional.interpolate(x, size=self.resolution, mode="bicubic")
        return self.model(x)


class Model:
    """
    The Transformer and other models loaded from HuggingFace.
    Supported by HuggingFace
    https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel
    """

    @staticmethod
    def get(model_name=None, **kwargs):  # pylint: disable=unused-argument
        """Returns a named model from HuggingFace."""
        if "t2t" in model_name:
            return T2TVIT(model_name)
        if "deepvit" in model_name:
            return DeepViT(model_name)

        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        model_name = model_name.replace("@", "/")
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)

        return ResolutionAdjustedModel(model_name, config)

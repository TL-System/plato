"""
Obtaining a Vision Transformer (ViT) model for image classification from HuggingFace.

Reference to the Tokens-to-Token ViT (T2T-ViT) model:
https://github.com/yitu-opensource/T2T-ViT

Reference to the Deep Vision Transformer (DeepViT) model:
https://github.com/zhoudaquan/dvit_repo

"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageClassification


from plato.config import Config


class ResolutionAdjustedModel(nn.Module):
    """
    Transforms the image resolution to the assigned resolution of a pretrained model.
    """

    def __init__(self, model_name, config) -> nn.Module:
        super().__init__()

        if (
            hasattr(Config().parameters, "model")
            and hasattr(Config().parameters.model, "pretrained")
            and not Config().parameters.model.pretrained
        ):
            ignore_mismatched_sizes = True
        else:
            ignore_mismatched_sizes = False

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
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
    """Wrapper for the T2T-ViT model."""

    def __init__(self, name) -> nn.Module:
        super().__init__()
        # pylint:disable=import-outside-toplevel
        from plato.models import t2tvit
        from plato.models.t2tvit.models import t2t_vit

        model_name = getattr(t2t_vit, name)
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

    def forward(self, feature):
        """The forward pass."""
        if feature.size(-1) != self.resolution:
            feature = nn.functional.interpolate(
                feature, size=self.resolution, mode="bicubic"
            )
        return self.model(feature)


class DeepViT(nn.Module):
    """Wrapper for the DeepViT model."""

    def __init__(self, name) -> nn.Module:
        super().__init__()
        # pylint:disable=import-outside-toplevel
        from plato.models.dvit.models import deep_vision_transformer

        model_name = getattr(deep_vision_transformer, name)
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

    def forward(self, feature):
        """The forward pass."""
        if feature.size(-1) != self.resolution:
            feature = nn.functional.interpolate(
                feature, size=self.resolution, mode="bicubic"
            )

        return self.model(feature)


class Model:
    """
    The Transformer and other models loaded from HuggingFace.
    Supported by HuggingFace
    https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel
    """

    # pylint:disable=too-few-public-methods
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
        config_kwargs.update(kwargs)

        model_name = model_name.replace("@", "/")
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)

        return ResolutionAdjustedModel(model_name, config)

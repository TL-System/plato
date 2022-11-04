from distutils.command.config import config
from re import T
from statistics import mode
from t2tvit.models import t2t_vit_14
from t2tvit.utils import load_for_transfer_learning
from dvit.models import deep_vision_transformer
from plato.config import Config

import torch
from torch import nn
from timm import create_model
from plato.config import Config


class T2T_VIT_14(nn.Module):
    """wrap up t2t-vit-14"""

    def __init__(self) -> nn.Module:
        super().__init__()

        t2t = t2t_vit_14(num_classes=Config().trainer.num_classes)
        load_for_transfer_learning(
            t2t,
            Config().parameters.architect.pretrain_path,
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


class Deep_ViT(nn.Module):
    """wrap up deep vit"""

    def __init__(self) -> nn.Module:
        super().__init__()

        deepvit = deep_vision_transformer.deepvit_S(
            pretrained=False, num_classes=Config.trainer.num_classes, in_chans=3
        )
        if hasattr(Config().parameters.architect, "pretrain_pth"):
            state_dict = torch.load(
                Config().parameters.architect.pretrain_path, map_location="cpu"
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


def get_vits():
    name = Config().trainer.model_name
    if name == "t2t_vit_14":
        return T2T_VIT_14
    if name == "deep_vit_16":
        return Deep_ViT
    return None


if __name__ == "__main__":
    model = get_vits()
    model = model()
    print(model)
